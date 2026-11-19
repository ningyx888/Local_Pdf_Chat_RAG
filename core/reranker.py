"""
重排序器 —— 对检索结果进行二次精排

学习要点：
- 两阶段检索（Recall + Rerank）是工业界常用的范式
- Recall 阶段用高效检索（FAISS/BM25）从大量文档中召回候选
- Rerank 阶段用更精确的模型（交叉编码器/LLM）对候选精排
- 交叉编码器比双塔模型更精确，但速度更慢（适合对少量候选精排）

两种重排方式（由 RERANK_METHOD 决定，默认 cross_encoder）：
- cross_encoder：本地模型，一次批量前向给所有候选打分，几十毫秒
- llm：调用大模型做 listwise 排序，一次请求排全部候选（早期是逐条 pointwise
  打分，N 条候选要发 N 次请求，延迟和成本都高一个量级，已废弃）
"""

import logging
import re
import threading

from config import (
    DEFAULT_MODEL_CHOICE,
    LLM_RERANK_DOC_CHARS,
    LLM_RERANK_MAX_CANDIDATES,
    LLM_RERANK_TEMPERATURE,
    RERANK_METHOD,
    RERANK_MODEL_NAME,
)

# 交叉编码器（懒加载 + 线程安全）
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """懒加载交叉编码器模型（双重检查锁定，线程安全）"""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    logging.info(f"加载重排序模型: {RERANK_MODEL_NAME}")
                    _cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
                    logging.info("交叉编码器加载成功")
                except Exception as e:
                    logging.error(f"加载交叉编码器失败: {str(e)}")
                    _cross_encoder = None
    return _cross_encoder


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """使用交叉编码器对检索结果进行重排序"""
    if not docs:
        return []

    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("交叉编码器不可用，跳过重排序")
        return _fallback_results(doc_ids, docs, metadata_list)

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = encoder.predict(cross_inputs)
        results = [
            (doc_id, {'content': doc, 'metadata': meta, 'score': float(score)})
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        return results[:top_k]
    except Exception as e:
        logging.error(f"交叉编码器重排序失败: {str(e)}")
        return _fallback_results(doc_ids, docs, metadata_list)


# ━━━ LLM 重排：listwise 批量排序 ━━━

_LISTWISE_TEMPLATE = """你是一个检索结果排序助手。请判断下列候选片段与问题的相关性，并按相关性从高到低排序。

问题：{query}

候选片段：
{candidates}

输出要求：
1. 只输出候选编号，用英文逗号分隔，从最相关到最不相关
2. 必须包含全部 {n} 个编号，每个编号只出现一次
3. 不要输出解释、标题或任何多余文字

排序结果："""


def _truncate(text, limit):
    """压缩空白并截断，避免单条候选撑爆提示词。"""
    return " ".join(str(text).split())[:limit]


def _build_listwise_prompt(query, docs, max_candidates=None, doc_chars=None):
    """把所有候选编号后拼进一个提示词，返回 (prompt, 参与排序的候选数)。

    只取前 max_candidates 条：listwise 提示词长度随候选数线性增长，
    候选太多会超上下文，也会拖慢延迟。剩下的候选保持原序接在后面。
    """
    max_candidates = LLM_RERANK_MAX_CANDIDATES if max_candidates is None else max_candidates
    doc_chars = LLM_RERANK_DOC_CHARS if doc_chars is None else doc_chars

    selected = list(docs[:max(1, max_candidates)])
    lines = [f"[{i}] {_truncate(doc, doc_chars)}" for i, doc in enumerate(selected, start=1)]
    prompt = _LISTWISE_TEMPLATE.format(
        query=query, candidates="\n".join(lines), n=len(selected)
    )
    return prompt, len(selected)


def _parse_ranking(text, n):
    """从 LLM 输出解析候选顺序，返回 (order, ok)。

    order 是 0-based 索引列表，长度恒为 n：
    - 越界编号、重复编号直接忽略
    - 缺失的编号按原始顺序补在末尾（宁可顺序不准，也不能丢候选）
    ok 表示是否解析出至少一个有效编号；False 时调用方应退回原始顺序。
    """
    order, seen = [], set()
    for token in re.findall(r"\d+", str(text or "")):
        idx = int(token) - 1
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    if not order:
        return list(range(n)), False
    order.extend(i for i in range(n) if i not in seen)
    return order, True


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5, model_choice=None):
    """用 LLM 做 listwise 重排：一次请求给全部候选排序。

    相比逐条 pointwise 打分，请求数从 N 次降到 1 次，延迟和成本都低一个量级；
    代价是候选片段要截断后塞进提示词，模型看不到完整段落。
    任何失败都退回原始顺序并打 warning，不静默假装排过序。
    """
    if not docs:
        return []

    # 延迟导入：core.generator 依赖 retriever，模块级导入会形成循环
    from core.generator import call_llm_simple

    prompt, n = _build_listwise_prompt(query, docs)
    try:
        raw = call_llm_simple(
            prompt,
            model_choice or DEFAULT_MODEL_CHOICE,
            temperature=LLM_RERANK_TEMPERATURE,
        )
    except Exception as e:
        logging.warning(f"LLM 重排失败，退回原始顺序: {str(e)}")
        return _fallback_results(doc_ids, docs, metadata_list)

    order, ok = _parse_ranking(raw, n)
    if not ok:
        logging.warning(f"LLM 重排输出无法解析，退回原始顺序: {str(raw)[:80]!r}")
        return _fallback_results(doc_ids, docs, metadata_list)

    results = []
    for rank, idx in enumerate(order):
        results.append((doc_ids[idx], {
            'content': docs[idx],
            'metadata': metadata_list[idx],
            'score': 1.0 - rank / max(1, n),
        }))
    # 超出参与排序范围的候选，保持原序接在后面
    for idx in range(n, len(docs)):
        results.append((doc_ids[idx], {
            'content': docs[idx],
            'metadata': metadata_list[idx],
            'score': 0.0,
        }))
    return results[:top_k]


def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5,
                   model_choice=None):
    """对检索结果进行重排序（统一入口）。

    method 缺省时读 RERANK_METHOD；未知取值退回原始顺序。
    model_choice 只对 llm 方式有意义（决定用哪个后端打分）。
    """
    if method is None:
        method = RERANK_METHOD

    if method == "llm":
        return rerank_with_llm(
            query, docs, doc_ids, metadata_list, top_k=top_k, model_choice=model_choice
        )
    if method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=top_k)
    return _fallback_results(doc_ids, docs, metadata_list)


def _fallback_results(doc_ids, docs, metadata_list):
    """回退方案：按原始顺序返回"""
    return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
            for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
