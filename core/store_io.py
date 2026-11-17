"""
索引持久化 —— 把内存中的向量与文本落盘，重启后无需重新上传

学习要点：
- 向量索引不是数据库：FAISS 只在进程内有效，要持久化必须自己保存原始向量
- 三份文件各司其职：向量矩阵、文本与元数据、指纹（manifest）
- 原子写：先写 .tmp 再 os.replace()，避免写到一半崩溃留下损坏文件
- 指纹校验：换了 embedding 模型或分块参数后，旧索引必须拒绝加载，
  否则会用 A 模型建的向量去匹配 B 模型的查询，结果全是噪音且很难排查

存储布局（默认 data/index/，可用环境变量 INDEX_DIR 覆盖）：

    data/index/
      embeddings.npy   float32 [n, dim]，第 i 行 ↔ id_order[i]
      meta.json        id_order / contents_map / metadatas_map
      manifest.json    指纹：schema 版本、模型名、维度、分块参数
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import CHUNK_OVERLAP, CHUNK_SIZE
from core.embeddings import EMBED_MODEL_NAME

SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"

EMB_FILE = "embeddings.npy"
META_FILE = "meta.json"
MANIFEST_FILE = "manifest.json"


class StoreError(Exception):
    """索引读写失败"""


class StoreIncompatible(StoreError):
    """索引与当前配置不兼容，需要重建"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 路径
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def index_dir(path: Optional[Any] = None) -> Path:
    """返回索引目录。显式传参优先，其次环境变量 INDEX_DIR，最后默认 data/index"""
    if path is not None:
        return Path(path)
    env_dir = os.getenv("INDEX_DIR")
    return Path(env_dir) if env_dir else DEFAULT_INDEX_DIR


def store_exists(path: Optional[Any] = None) -> bool:
    """三份文件齐全才算存在（缺一份说明上次写入没完成）"""
    target = index_dir(path)
    return all((target / name).exists() for name in (EMB_FILE, META_FILE, MANIFEST_FILE))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 写
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _atomic_write_bytes(target: Path, writer) -> None:
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as handle:
        writer(handle)
    os.replace(tmp, target)


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    def _dump(handle):
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))

    _atomic_write_bytes(target, _dump)


def build_manifest(num_chunks: int, dimension: int) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "embed_model": EMBED_MODEL_NAME,
        "dimension": int(dimension),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_chunks": int(num_chunks),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_store(store, path: Optional[Any] = None) -> Path:
    """
    把 VectorStore 的当前状态写入磁盘

    Args:
        store: 具备 embeddings / id_order / contents_map / metadatas_map 属性的对象

    Raises:
        StoreError: 向量缺失或行数与 id 数量不一致
    """
    state = {
        "embeddings": getattr(store, "embeddings", None),
        "id_order": list(store.id_order),
        "contents_map": dict(store.contents_map),
        "metadatas_map": dict(store.metadatas_map),
    }
    return save_state(state, path)


def save_state(state: Dict[str, Any], path: Optional[Any] = None) -> Path:
    """
    把状态字典写入磁盘，结构需与 load_store() 的返回值一致

    删除文档后拿到的是一个新的 state（而不是 VectorStore 对象），走这个入口落盘。
    """
    embeddings = state.get("embeddings")
    if embeddings is None:
        raise StoreError("向量数组缺失：状态里没有 embeddings，无法落盘")

    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    id_order = list(state.get("id_order", []))

    if embeddings.ndim != 2:
        raise StoreError(f"向量矩阵维度异常: {embeddings.shape}")
    if embeddings.shape[0] != len(id_order):
        raise StoreError(
            f"向量行数({embeddings.shape[0]})与 id 数量({len(id_order)})不一致，拒绝写入"
        )

    target = index_dir(path)
    target.mkdir(parents=True, exist_ok=True)

    meta = {
        "id_order": id_order,
        "contents_map": dict(state.get("contents_map", {})),
        "metadatas_map": dict(state.get("metadatas_map", {})),
    }
    manifest = build_manifest(len(id_order), embeddings.shape[1])

    # 顺序有讲究：先写数据，最后写 manifest。manifest 存在即代表数据完整。
    _atomic_write_bytes(target / EMB_FILE, lambda handle: np.save(handle, embeddings))
    _atomic_write_json(target / META_FILE, meta)
    _atomic_write_json(target / MANIFEST_FILE, manifest)

    logging.info(f"索引已保存: {target}（{len(id_order)} 个文本块）")
    return target


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 读
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_compat(manifest: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """比对 manifest 与当前配置，返回 (是否兼容, 原因)"""
    if not manifest:
        return False, "缺少 manifest.json，无法判断索引版本"

    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        return False, (
            f"索引格式版本不兼容（文件 {manifest.get('schema_version')} ≠ 当前 {SCHEMA_VERSION}）"
        )

    if manifest.get("embed_model") != EMBED_MODEL_NAME:
        return False, (
            f"向量化模型已变更（索引 {manifest.get('embed_model')} ≠ 当前 {EMBED_MODEL_NAME}），"
            "旧向量与新查询不在同一空间"
        )

    if int(manifest.get("chunk_size", -1)) != CHUNK_SIZE:
        return False, (
            f"分块大小已变更（索引 {manifest.get('chunk_size')} ≠ 当前 {CHUNK_SIZE}）"
        )

    if int(manifest.get("chunk_overlap", -1)) != CHUNK_OVERLAP:
        return False, (
            f"分块重叠已变更（索引 {manifest.get('chunk_overlap')} ≠ 当前 {CHUNK_OVERLAP}）"
        )

    return True, ""


def load_store(path: Optional[Any] = None, strict: bool = True) -> Optional[Dict[str, Any]]:
    """
    读取磁盘上的索引快照

    Args:
        strict: True 时，配置不兼容直接抛 StoreIncompatible；
                False 时只记录警告并返回 None，由调用方决定降级策略

    Returns:
        dict(embeddings, id_order, contents_map, metadatas_map, documents, manifest)
        索引不存在时返回 None
    """
    target = index_dir(path)
    if not store_exists(target):
        return None

    try:
        manifest = json.loads((target / MANIFEST_FILE).read_text(encoding="utf-8"))
        meta = json.loads((target / META_FILE).read_text(encoding="utf-8"))
        embeddings = np.load(target / EMB_FILE, allow_pickle=False)
    except Exception as exc:  # 文件损坏 / JSON 解析失败
        raise StoreError(f"索引文件读取失败: {exc}") from exc

    compatible, reason = check_compat(manifest)
    if not compatible:
        if strict:
            raise StoreIncompatible(reason)
        logging.warning(f"⚠️ 忽略已有索引：{reason}")
        return None

    id_order = list(meta.get("id_order", []))
    if embeddings.shape[0] != len(id_order):
        raise StoreError(
            f"索引不一致：向量 {embeddings.shape[0]} 行，id {len(id_order)} 个"
        )
    if embeddings.shape[1] != int(manifest.get("dimension", embeddings.shape[1])):
        raise StoreError(
            f"索引不一致：向量维度 {embeddings.shape[1]}，manifest 声明 {manifest.get('dimension')}"
        )

    contents_map = dict(meta.get("contents_map", {}))
    metadatas_map = dict(meta.get("metadatas_map", {}))

    return {
        "embeddings": np.ascontiguousarray(embeddings, dtype="float32"),
        "id_order": id_order,
        "contents_map": contents_map,
        "metadatas_map": metadatas_map,
        "documents": derive_documents(metadatas_map),
        "manifest": manifest,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文档级操作
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def derive_documents(metadatas_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从块级元数据反推文档清单（不额外存一份，避免两个真相来源）"""
    docs: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas_map.values():
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        entry = docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "source": meta.get("source", "未知文件"),
                "chunks": 0,
                "added_at": meta.get("added_at", ""),
            },
        )
        entry["chunks"] += 1
        if not entry["added_at"]:
            entry["added_at"] = meta.get("added_at", "")
    return sorted(docs.values(), key=lambda item: item.get("added_at", ""))


def document_exists(state: Dict[str, Any], doc_id: str) -> bool:
    """该文档是否已在库中"""
    return any(
        meta.get("doc_id") == doc_id for meta in state["metadatas_map"].values()
    )


def delete_document(state: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """
    返回移除指定文档后的新状态（纯函数，不改原对象）

    删除后需要重建 FAISS 索引：行号会整体前移，局部删除会让 id 映射错位。
    """
    keep_rows = [
        i
        for i, chunk_id in enumerate(state["id_order"])
        if state["metadatas_map"].get(chunk_id, {}).get("doc_id") != doc_id
    ]

    keep_ids = [state["id_order"][i] for i in keep_rows]
    new_state = {
        "embeddings": state["embeddings"][keep_rows],
        "id_order": keep_ids,
        "contents_map": {cid: state["contents_map"][cid] for cid in keep_ids},
        "metadatas_map": {cid: state["metadatas_map"][cid] for cid in keep_ids},
        "manifest": state.get("manifest", {}),
    }
    new_state["documents"] = derive_documents(new_state["metadatas_map"])
    logging.info(f"已从索引中移除文档 {doc_id}，剩余 {len(keep_ids)} 个文本块")
    return new_state


def clear_store(path: Optional[Any] = None) -> None:
    """删除磁盘上的索引文件"""
    target = index_dir(path)
    for name in (EMB_FILE, META_FILE, MANIFEST_FILE):
        (target / name).unlink(missing_ok=True)
    logging.info(f"索引文件已清除: {target}")
