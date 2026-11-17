"""
建库 + 检索对比实验（不调 LLM）

做四件事：
    1. 解析 PDF -> 分块 -> 向量化 -> 建 FAISS + BM25
    2. 同一个 query 分别走 FAISS(L2) 和 BM25，并排打印
    3. 额外算余弦相似度，验证 "未归一化 + L2" 与余弦是否一致
    4. 跑融合（默认 rrf 与 weighted 并排），看两路怎么被揉起来

用法：
    uv run python tools/inspect_index.py "挖掘机维修案例(样例）.pdf" --query "液压泵有异响是什么原因"
    uv run python tools/inspect_index.py "某.pdf" --query "..." --top 8 --k 0
    uv run python tools/inspect_index.py "某.pdf" --query "..." --fusion weighted --alpha 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                    # noqa: E402

from config import FUSION_METHOD, HYBRID_ALPHA, RRF_K, RRF_WEIGHT_BM25, RRF_WEIGHT_SEMANTIC  # noqa: E402
from core.document_loader import extract_text         # noqa: E402
from core.text_splitter import split_text             # noqa: E402
from core.embeddings import encode_texts, encode_query  # noqa: E402
from core.vector_store import vector_store            # noqa: E402
from core.bm25_index import bm25_manager              # noqa: E402
from core.retriever import hybrid_merge, rrf_merge    # noqa: E402


def preview(text, n=60):
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def main():
    ap = argparse.ArgumentParser(description="FAISS vs BM25 对比")
    ap.add_argument("path")
    ap.add_argument("--query", required=True, help="用来检索的问题")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None, help="weighted 融合的语义权重，默认读 config.HYBRID_ALPHA(0.7)")
    ap.add_argument("--fusion", choices=["rrf", "weighted", "both"], default="both",
                    help="融合方式，默认 both 并排对比（线上默认由 config.FUSION_METHOD 决定）")
    ap.add_argument("--k", type=int, default=None, help=f"RRF 平滑常数，默认读 config.RRF_K({RRF_K})")
    ap.add_argument("--w-sem", type=float, default=None, help=f"RRF 语义权重，默认 {RRF_WEIGHT_SEMANTIC}")
    ap.add_argument("--w-bm25", type=float, default=None, help=f"RRF BM25 权重，默认 {RRF_WEIGHT_BM25}")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[错误] 文件不存在: {args.path}")
        return 1

    text = extract_text(args.path)
    if not text:
        print("[错误] 提取文本为空（扫描件或不支持的格式）")
        return 1

    chunks = split_text(text)
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    metas = [{"source": os.path.basename(args.path), "chunk_index": i} for i in range(len(chunks))]

    print(f"文件: {os.path.basename(args.path)}  块数: {len(chunks)}")
    print("正在向量化...（首次会加载模型，稍慢）")
    embeddings = encode_texts(chunks, show_progress=False)

    vector_store.build_index(chunks, chunk_ids, metas, embeddings)
    bm25_manager.build_index(chunks, chunk_ids)

    info = vector_store.index.get_index_info()
    norms = np.linalg.norm(embeddings, axis=1)
    print()
    print("== 索引概况 ==")
    print(f"  向量数 {info['size']} | 维度 {info['dimension']} | 类型 {info['index_type']}")
    print(f"  向量模长: 最小 {norms.min():.3f} 最大 {norms.max():.3f} 平均 {norms.mean():.3f}"
          f"  -> {'未归一化' if abs(norms.mean() - 1.0) > 0.05 else '近似归一化'}")
    print()

    q = encode_query(args.query)

    # --- FAISS ---
    D, I = vector_store.index.search(q, k=args.top)
    qn = q[0] / (np.linalg.norm(q[0]) + 1e-12)
    sem_ids, sem_docs, sem_metas = [], [], []
    print(f"== FAISS Top{args.top}（L2 距离，越小越近）==")
    print(f"  {'#':<3}{'chunk':<12}{'L2':>8}{'cos':>8}  预览")
    for rank, (dist, row) in enumerate(zip(D[0], I[0])):
        if row < 0 or row >= len(chunks):
            continue
        v = embeddings[row]
        cos = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-12)))
        cid = chunk_ids[row]
        sem_ids.append(cid)
        sem_docs.append(chunks[row])
        sem_metas.append(metas[row])
        print(f"  {rank:<3}{cid:<12}{dist:>8.4f}{cos:>8.4f}  {preview(chunks[row])}")
    print()

    # --- BM25 ---
    bm25_hits = bm25_manager.search(args.query, top_k=args.top)
    print(f"== BM25 Top{args.top}（原始分，无上界）==")
    if not bm25_hits:
        print("  （无命中：query 中的词在语料里一次都没出现）")
    for rank, r in enumerate(bm25_hits):
        print(f"  {rank:<3}{r['id']:<12}{r['score']:>8.3f}  {preview(r['content'])}")
    print(f"  BM25 返回 {len(bm25_hits)} 条（分数为 0 的已被过滤）")
    print()

    bm25_ids = {r["id"] for r in bm25_hits}
    overlap = bm25_ids & set(sem_ids)
    union = bm25_ids | set(sem_ids)
    print("== 两路对比 ==")
    print(f"  重合 {len(overlap)} 条 / 各自 Top{args.top}；并集 {len(union)} 条")
    print(f"  仅语义命中: {sorted(set(sem_ids) - bm25_ids)}")
    print(f"  仅关键词命中: {sorted(bm25_ids - set(sem_ids))}")
    print()

    # --- 融合 ---
    sem = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas]}
    sem_rank = {cid: i + 1 for i, cid in enumerate(sem_ids)}
    bm25_rank = {r["id"]: i + 1 for i, r in enumerate(bm25_hits)}

    k_used = RRF_K if args.k is None else args.k
    w_sem = RRF_WEIGHT_SEMANTIC if args.w_sem is None else args.w_sem
    w_bm25 = RRF_WEIGHT_BM25 if args.w_bm25 is None else args.w_bm25
    alpha_used = HYBRID_ALPHA if args.alpha is None else args.alpha
    limit = max(args.top, 8)

    def show_table(title, merged):
        print(f"== {title} ==")
        print(f"  {'#':<3}{'chunk':<12}{'得分':>8}{'语义':>6}{'bm25':>6}  预览")
        for rank, (cid, item) in enumerate(merged[:limit], start=1):
            sem_r = sem_rank.get(cid, "-")
            bm_r = bm25_rank.get(cid, "-")
            print(f"  {rank:<3}{cid:<12}{item['score']:>8.4f}{str(sem_r):>6}{str(bm_r):>6}  {preview(item['content'])}")
        print()

    orders = {}
    if args.fusion in ("rrf", "both"):
        rrf_sorted = rrf_merge(sem, bm25_hits, k=k_used, weights=(w_sem, w_bm25))
        show_table(f"RRF 融合（k={k_used}, w={w_sem}/{w_bm25}）", rrf_sorted)
        orders["rrf"] = [cid for cid, _ in rrf_sorted[:limit]]
    if args.fusion in ("weighted", "both"):
        weighted_sorted = hybrid_merge(sem, bm25_hits, alpha=alpha_used)
        show_table(f"加权分数融合（alpha={alpha_used}）", weighted_sorted)
        orders["weighted"] = [cid for cid, _ in weighted_sorted[:limit]]

    if len(orders) == 2:
        rrf_order, weighted_order = orders["rrf"], orders["weighted"]
        if rrf_order == weighted_order:
            print("两种融合方式的 Top 顺序完全一致。")
        else:
            print("两种融合方式排序不同：")
            for pos in range(limit):
                a = rrf_order[pos] if pos < len(rrf_order) else "-"
                b = weighted_order[pos] if pos < len(weighted_order) else "-"
                if a != b:
                    print(f"  第{pos + 1}位: rrf={a}  weighted={b}")
        print()

    print(f"线上默认融合方式: FUSION_METHOD={FUSION_METHOD}")
    print("提示：只被 BM25 捞到的块若出现在上面，说明关键词那一路在起作用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())