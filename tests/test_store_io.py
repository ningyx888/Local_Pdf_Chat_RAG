"""
索引持久化的单元测试

只测 core/store_io.py 与 VectorStore 的持久化接口，不加载 embedding 模型：
向量用 numpy 直接造，这样测试能在秒级跑完。
"""

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from core import store_io
from core.vector_store import VectorStore

DIM = 8


@pytest.fixture()
def index_path():
    """工作区内的临时索引目录（系统 temp 在沙箱下不可写）"""
    base = Path(__file__).resolve().parent.parent / "data" / "_pytest"
    path = base / f"case_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def make_state(docs=(("docA", "a.txt", 2), ("docB", "b.txt", 3))):
    """构造一个与 load_store() 同结构的状态字典"""
    id_order, contents, metas = [], {}, {}
    rows = []
    for doc_id, source, count in docs:
        for i in range(count):
            chunk_id = f"{doc_id}_chunk_{i}"
            id_order.append(chunk_id)
            contents[chunk_id] = f"{source} 的第 {i} 段内容"
            metas[chunk_id] = {"source": source, "doc_id": doc_id, "added_at": "2026-11-17T10:00:00"}
            rows.append(np.ones(DIM, dtype="float32") * (len(rows) + 1))
    return {
        "embeddings": np.vstack(rows),
        "id_order": id_order,
        "contents_map": contents,
        "metadatas_map": metas,
    }


# ━━━ 写读往返 ━━━
def test_save_and_load_roundtrip(index_path):
    state = make_state()
    store_io.save_state(state, index_path)

    loaded = store_io.load_store(index_path)

    assert loaded["id_order"] == state["id_order"]
    assert loaded["contents_map"] == state["contents_map"]
    assert loaded["metadatas_map"] == state["metadatas_map"]
    np.testing.assert_array_equal(loaded["embeddings"], state["embeddings"])
    assert loaded["manifest"]["num_chunks"] == len(state["id_order"])


def test_documents_are_derived_from_chunk_metadata(index_path):
    store_io.save_state(make_state(), index_path)

    loaded = store_io.load_store(index_path)
    docs = {doc["doc_id"]: doc for doc in loaded["documents"]}

    assert set(docs) == {"docA", "docB"}
    assert docs["docA"]["chunks"] == 2
    assert docs["docB"]["chunks"] == 3
    assert docs["docA"]["source"] == "a.txt"


def test_load_returns_none_when_nothing_saved(index_path):
    assert store_io.store_exists(index_path) is False
    assert store_io.load_store(index_path) is None


def test_atomic_write_leaves_no_tmp_files(index_path):
    store_io.save_state(make_state(), index_path)

    names = sorted(p.name for p in index_path.iterdir())

    assert names == sorted([store_io.EMB_FILE, store_io.META_FILE, store_io.MANIFEST_FILE])


# ━━━ 一致性校验 ━━━
def test_save_rejects_row_count_mismatch(index_path):
    state = make_state()
    state["embeddings"] = state["embeddings"][:-1]

    with pytest.raises(store_io.StoreError):
        store_io.save_state(state, index_path)


def test_save_rejects_missing_embeddings(index_path):
    state = make_state()
    state["embeddings"] = None

    with pytest.raises(store_io.StoreError):
        store_io.save_state(state, index_path)


def test_load_rejects_row_count_mismatch(index_path):
    store_io.save_state(make_state(), index_path)

    meta_file = index_path / store_io.META_FILE
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["id_order"] = meta["id_order"][:-1]
    meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(store_io.StoreError):
        store_io.load_store(index_path)


# ━━━ 指纹校验 ━━━
def test_manifest_records_current_config(index_path):
    store_io.save_state(make_state(), index_path)
    manifest = json.loads((index_path / store_io.MANIFEST_FILE).read_text(encoding="utf-8"))

    assert manifest["schema_version"] == store_io.SCHEMA_VERSION
    assert manifest["dimension"] == DIM
    assert manifest["embed_model"]
    assert store_io.check_compat(manifest) == (True, "")


def test_changed_embed_model_is_rejected(index_path):
    store_io.save_state(make_state(), index_path)

    manifest_file = index_path / store_io.MANIFEST_FILE
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["embed_model"] = "some-other-model"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(store_io.StoreIncompatible):
        store_io.load_store(index_path)


def test_non_strict_load_degrades_to_none(index_path):
    store_io.save_state(make_state(), index_path)

    manifest_file = index_path / store_io.MANIFEST_FILE
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["chunk_size"] = manifest["chunk_size"] + 100
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    assert store_io.load_store(index_path, strict=False) is None


# ━━━ 文档级操作 ━━━
def test_delete_document_keeps_remaining_rows_aligned(index_path):
    state = make_state()
    store_io.save_state(state, index_path)
    loaded = store_io.load_store(index_path)

    new_state = store_io.delete_document(loaded, "docA")

    assert new_state["id_order"] == ["docB_chunk_0", "docB_chunk_1", "docB_chunk_2"]
    assert new_state["embeddings"].shape[0] == 3
    # 删掉 A 的两行后，第 0 行应当正好是 B 的第 0 段
    np.testing.assert_array_equal(new_state["embeddings"][0], loaded["embeddings"][2])
    assert new_state["contents_map"]["docB_chunk_0"] == "b.txt 的第 0 段内容"
    assert [d["doc_id"] for d in new_state["documents"]] == ["docB"]


def test_delete_document_does_not_mutate_original():
    state = make_state()

    store_io.delete_document(state, "docA")

    assert len(state["id_order"]) == 5


def test_deleting_last_document_leaves_empty_state(index_path):
    state = make_state(docs=(("docA", "a.txt", 2),))
    new_state = store_io.delete_document(state, "docA")

    assert new_state["id_order"] == []
    assert new_state["embeddings"].shape == (0, DIM)
    assert new_state["documents"] == []


def test_document_exists():
    state = make_state()

    assert store_io.document_exists(state, "docA") is True
    assert store_io.document_exists(state, "docX") is False


def test_clear_store_removes_all_files(index_path):
    store_io.save_state(make_state(), index_path)

    store_io.clear_store(index_path)

    assert store_io.store_exists(index_path) is False
    assert store_io.load_store(index_path) is None


# ━━━ VectorStore 侧接口 ━━━
def test_vector_store_append_keeps_row_order():
    store = VectorStore()
    store.build_index(
        ["a0", "a1"], ["a0", "a1"], [{"doc_id": "A"}, {"doc_id": "A"}],
        np.vstack([np.ones(DIM), np.ones(DIM) * 2]).astype("float32"),
    )
    store.append(
        ["b0"], ["b0"], [{"doc_id": "B"}],
        (np.ones(DIM) * 3).reshape(1, -1).astype("float32"),
    )

    assert store.id_order == ["a0", "a1", "b0"]
    assert store.embeddings.shape == (3, DIM)
    assert store.index.ntotal == 3
    assert store.has_document("B") is True
    assert store.has_document("C") is False


def test_vector_store_append_rejects_dimension_mismatch():
    store = VectorStore()
    store.build_index(["a0"], ["a0"], [{"doc_id": "A"}], np.ones((1, DIM), dtype="float32"))

    with pytest.raises(ValueError):
        store.append(["b0"], ["b0"], [{"doc_id": "B"}], np.ones((1, DIM + 1), dtype="float32"))


def test_vector_store_restore_from_state(index_path):
    state = make_state()
    store_io.save_state(state, index_path)
    loaded = store_io.load_store(index_path)

    store = VectorStore()
    store.restore(loaded)

    assert store.id_order == state["id_order"]
    assert store.index.ntotal == len(state["id_order"])
    assert store.has_embeddings is True
    docs, ids, _ = store.search(np.ones((1, DIM), dtype="float32"), k=1)
    assert ids[0] in state["id_order"]
    assert docs[0] == state["contents_map"][ids[0]]


def test_vector_store_clear_drops_embeddings():
    store = VectorStore()
    store.build_index(["a0"], ["a0"], [{"doc_id": "A"}], np.ones((1, DIM), dtype="float32"))

    store.clear()

    assert store.has_embeddings is False
    assert store.is_ready is False
    assert store.id_order == []
