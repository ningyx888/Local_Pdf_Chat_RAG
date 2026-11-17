"""
索引持久化自检 —— 不加载任何模型，验证 store_io 的写入/读取/删除/校验

用法：
    .venv\\Scripts\\python.exe tools\\check_store.py

它会构造一个假的向量库（6 个块 / 2 个文档），走一遍完整生命周期：
保存 → 读取 → 反推文档清单 → 删除一个文档 → 指纹不兼容 → 行数不一致
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK，直接打印 ✅/❌ 会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import store_io
from core.vector_store import VectorStore

DOCS = {
    "a1b2c3d4e5f6": ("手册A.pdf", 4),
    "9f8e7d6c5b4a": ("手册B.pdf", 2),
}


def make_store() -> VectorStore:
    store = VectorStore()
    rng = np.random.default_rng(0)
    chunks, ids, metas = [], [], []
    for doc_id, (source, count) in DOCS.items():
        for i in range(count):
            chunks.append(f"{source} 的第 {i} 段内容")
            ids.append(f"{doc_id}_chunk_{i}")
            metas.append(
                {"source": source, "doc_id": doc_id, "added_at": "2026-11-17T10:00:00"}
            )
    embeddings = rng.random((len(chunks), 384)).astype("float32")
    store.build_index(chunks, ids, metas, embeddings)
    return store


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        raise SystemExit(1)


def main() -> None:
    # 用工作区内的临时目录，避免系统临时目录在沙箱/受限环境下不可写
    workdir = Path(__file__).resolve().parent.parent / "data" / "_check"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        store = make_store()
        check("构建内存索引", store.total_chunks == 6, f"{store.total_chunks} 个块")
        check("向量已保留", store.has_embeddings and store.embeddings.shape == (6, 384))

        # 1. 保存
        store_io.save_store(store, workdir)
        check(
            "三份文件齐全",
            store_io.store_exists(workdir),
            ", ".join(sorted(p.name for p in workdir.iterdir())),
        )

        # 2. 读取
        state = store_io.load_store(workdir)
        check("读回向量矩阵", state["embeddings"].shape == (6, 384))
        check("向量内容一致", np.allclose(state["embeddings"], store.embeddings))
        check("id 顺序一致", state["id_order"] == store.id_order)
        check("文本一致", state["contents_map"] == store.contents_map)

        # 3. 反推文档清单
        docs = state["documents"]
        check("文档清单", len(docs) == 2, f"{[d['source'] for d in docs]}")
        check("块数统计", [d["chunks"] for d in docs] == [4, 2])

        # 4. 去重判断
        check("已入库文档可识别", store_io.document_exists(state, "a1b2c3d4e5f6"))
        check("新文档可识别", not store_io.document_exists(state, "000000000000"))

        # 5. 删除一个文档
        trimmed = store_io.delete_document(state, "a1b2c3d4e5f6")
        check("删除后剩余块数", len(trimmed["id_order"]) == 2)
        check("删除后向量行数", trimmed["embeddings"].shape == (2, 384))
        check("删除后文档数", len(trimmed["documents"]) == 1)
        check("删除的是对的文档", trimmed["documents"][0]["source"] == "手册B.pdf")

        # 6. 指纹不兼容
        manifest_path = workdir / store_io.MANIFEST_FILE
        original = manifest_path.read_text(encoding="utf-8")
        tampered = json.loads(original)
        tampered["embed_model"] = "some-other-model"
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            store_io.load_store(workdir)
            check("指纹不兼容应报错", False, "没有抛出异常")
        except store_io.StoreIncompatible as exc:
            check("指纹不兼容被拦截", True, str(exc)[:40] + "...")
        manifest_path.write_text(original, encoding="utf-8")

        # 7. 行数与 id 数量不一致
        meta_path = workdir / store_io.META_FILE
        original_meta = meta_path.read_text(encoding="utf-8")
        broken = json.loads(original_meta)
        broken["id_order"].append("多出来的一个id")
        meta_path.write_text(json.dumps(broken), encoding="utf-8")
        try:
            store_io.load_store(workdir)
            check("行数不一致应报错", False, "没有抛出异常")
        except store_io.StoreError as exc:
            check("行数不一致被拦截", True, str(exc)[:40] + "...")
        meta_path.write_text(original_meta, encoding="utf-8")

        # 8. 清除
        store_io.clear_store(workdir)
        check("清除后索引不存在", not store_io.store_exists(workdir))

        print("\n全部通过：写入 / 读取 / 去重 / 删除 / 校验 均正常")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
