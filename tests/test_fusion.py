"""RRF 融合的单元测试

覆盖点：
- 双路命中优先、单路名次保留
- 纯 BM25 命中能被抬进候选池（这是从加权求和改 RRF 的核心收益）
- 空路 / 畸形输入不抛异常
- k 与权重的调节效果
- 同分时顺序稳定（可复现）
- fuse_results 的分派
"""

import pytest

import core.retriever as retriever
from core.retriever import fuse_results, hybrid_merge, rrf_merge


def make_semantic(ids, prefix="doc"):
    """构造 vector_store.search 返回的结构"""
    return {
        "ids": [list(ids)],
        "documents": [[f"content of {i}" for i in ids]],
        "metadatas": [[{"source": f"{i}.pdf"} for i in ids]],
    }


def make_bm25(ids, scores=None):
    if scores is None:
        scores = [10.0 - i for i in range(len(ids))]
    return [
        {"id": i, "score": s, "content": f"content of {i}"}
        for i, s in zip(ids, scores)
    ]


def test_rrf_prefers_document_hit_by_both_routes():
    semantic = make_semantic(["doc-a", "doc-b"])
    sparse = make_bm25(["doc-b", "doc-c"])

    merged = rrf_merge(semantic, sparse, k=50)
    merged_by_id = dict(merged)

    assert set(merged_by_id) == {"doc-a", "doc-b", "doc-c"}
    assert merged[0][0] == "doc-b"  # 两路都命中
    assert merged_by_id["doc-b"]["score"] > merged_by_id["doc-a"]["score"]


def test_rrf_preserves_semantic_order_when_bm25_empty():
    semantic = make_semantic(["doc-1", "doc-2", "doc-3"])

    merged = rrf_merge(semantic, [], k=50)

    assert [doc_id for doc_id, _ in merged] == ["doc-1", "doc-2", "doc-3"]


def test_rrf_preserves_bm25_order_when_semantic_empty():
    sparse = make_bm25(["doc-x", "doc-y", "doc-z"])

    merged = rrf_merge({}, sparse, k=50)

    assert [doc_id for doc_id, _ in merged] == ["doc-x", "doc-y", "doc-z"]


def test_rrf_promotes_bm25_only_document_above_low_rank_semantic():
    """核心回归：纯 BM25 命中的块能压过语义路的第 6 名。

    加权求和时它最多拿 1-alpha = 0.3 分，而语义第 6 名是 0.7×(1-5/10)=0.35，
    所以这类块只能排在语义前 6 名之后，很难进 RETRIEVAL_TOP_K=10 的候选池前列。
    """
    semantic = make_semantic([f"sem-{i}" for i in range(1, 11)])
    sparse = make_bm25(["bm-only", "bm-2"])

    merged = rrf_merge(semantic, sparse, k=50)
    order = [doc_id for doc_id, _ in merged]

    assert "bm-only" in order
    # k=50：纯 BM25 第 1 名 1/51 ≈ 0.0196，语义第 6 名 1/56 ≈ 0.0179
    assert order.index("bm-only") < order.index("sem-6")

    weighted = dict(hybrid_merge(semantic, sparse, alpha=0.7))
    assert weighted["bm-only"]["score"] < weighted["sem-6"]["score"]


def test_rrf_returns_empty_when_both_routes_empty():
    assert rrf_merge({}, [], k=50) == []
    assert rrf_merge({"ids": [[]], "documents": [[]], "metadatas": [[]]}, [], k=50) == []


def test_rrf_k_controls_head_compression():
    semantic = make_semantic(["doc-1", "doc-2"])

    sharp = dict(rrf_merge(semantic, [], k=0))
    flat = dict(rrf_merge(semantic, [], k=60))

    assert [d for d, _ in rrf_merge(semantic, [], k=0)] == [d for d, _ in rrf_merge(semantic, [], k=60)]
    sharp_ratio = sharp["doc-1"]["score"] / sharp["doc-2"]["score"]
    flat_ratio = flat["doc-1"]["score"] / flat["doc-2"]["score"]
    assert sharp_ratio == pytest.approx(2.0)   # k=0 → 1/1 vs 1/2
    assert flat_ratio < 1.05                   # k=60 → 几乎压平
    assert sharp_ratio > flat_ratio


def test_rrf_weight_can_promote_bm25_only_document_to_top():
    semantic = make_semantic(["sem-1", "sem-2"])
    sparse = make_bm25(["bm-only"])

    merged = rrf_merge(semantic, sparse, k=50, weights=(1.0, 3.0))

    assert merged[0][0] == "bm-only"


def test_rrf_is_deterministic_on_ties():
    semantic = make_semantic(["doc-x"])
    sparse = make_bm25(["doc-y"])  # 两路各 1/51，精确同分

    first = rrf_merge(semantic, sparse, k=50)
    second = rrf_merge(semantic, sparse, k=50)

    assert first[0][1]["score"] == pytest.approx(first[1][1]["score"])
    assert [d for d, _ in first] == ["doc-x", "doc-y"]  # 同分按 doc_id 升序
    assert first == second


def test_rrf_tolerates_malformed_input():
    # 语义结构缺字段
    assert [d for d, _ in rrf_merge({"ids": [["a"]]}, [], k=50)] == []
    # 语义整体不是 dict
    assert [d for d, _ in rrf_merge(None, make_bm25(["b"]), k=50)] == ["b"]
    # BM25 元素缺 content
    assert [d for d, _ in rrf_merge({}, [{"id": "c", "score": 1.0}], k=50)] == []
    # BM25 整体不是 list
    assert [d for d, _ in rrf_merge({}, "oops", k=50)] == []
    # 行长度不一致
    broken = {"ids": [["a", "b"]], "documents": [["x"]], "metadatas": [[{}, {}]]}
    assert rrf_merge(broken, [], k=50) == []


def test_rrf_backfills_metadata_for_bm25_only_document(monkeypatch):
    monkeypatch.setattr(
        retriever.vector_store,
        "metadatas_map",
        {"bm-only": {"doc_id": "d1", "source": "manual.pdf"}},
    )

    merged = dict(rrf_merge({}, make_bm25(["bm-only", "unknown"]), k=50))

    assert merged["bm-only"]["metadata"] == {"doc_id": "d1", "source": "manual.pdf"}
    assert merged["unknown"]["metadata"] == {}


def test_fuse_results_dispatches_by_method():
    semantic = make_semantic(["doc-a", "doc-b"])
    sparse = make_bm25(["doc-b", "doc-c"])

    assert fuse_results(semantic, sparse, method="rrf") == rrf_merge(semantic, sparse)
    assert fuse_results(semantic, sparse, method="weighted") == hybrid_merge(semantic, sparse)
    assert fuse_results(semantic, sparse, method=" RRF ") == rrf_merge(semantic, sparse)
    # 默认走 config.FUSION_METHOD
    assert fuse_results(semantic, sparse) == rrf_merge(semantic, sparse)
