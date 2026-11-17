# 02 · 建索引：两套索引 + 一层 id 映射

> 目标：搞清 `list[str]` 怎么变成「FAISS 里的 (N, 384) 矩阵」和「BM25 里的词频表」，
> 以及为什么两套完全不同的东西能回答同一个问题。

## 三个角色

| 文件 | 产物 | 一句话 |
|---|---|---|
| `core/embeddings.py:35` `encode_texts()` | `(N, 384) float32` 矩阵 | 每段文字 → 一个语义向量 |
| `core/vector_store.py:99` `build_index()` | FAISS 索引 + 3 个 dict | 向量进索引，原文和元数据留在 Python 里 |
| `core/bm25_index.py:31` `build_index()` | BM25Okapi + 分词语料 | 每段文字 → 一堆词 |

## 一、向量化：`encode_texts`

```python
# core/embeddings.py:18 / :21-32 / :47-48
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
@lru_cache(maxsize=1)          # 进程内只加载一次
model = SentenceTransformer(...)
embeddings = model.encode(texts)   # 返回 (N, 384)，未归一化
```

三个事实：

1. **384 维**。模型是英文语料上训练的，对中文能用但不算强（`embeddings.py:6` 的注释自己写了这句）。缓存里已有 `BAAI/bge-small-zh-v1.5`，换它只需改 `EMBED_MODEL_NAME` 一行。
2. **`lru_cache(maxsize=1)`** 保证模型只加载一次；这也是为什么第一次查询要等十几秒、之后秒回。
3. **输出未归一化**。`model.encode()` 默认 `normalize_embeddings=False`，向量长度不为 1。而 FAISS 用的是 **L2 距离**（见下），不是余弦——两者对未归一化向量**不等价**。这是本项目一个隐蔽的小偏差：排序大体正确，但和"余弦相似度"并不严格一致。

## 二、FAISS：`AutoFaissIndex`

`core/vector_store.py:41-62` 按向量数量自动选类型：

| 数量 | 类型 | 说明 |
|---|---|---|
| ≤ 10,000 | `IndexFlatL2` | 暴力精确搜索，我们的场景永远走这里 |
| 1万–10万 | `IndexIVFFlat` | 先聚类再搜，需要 `train()` |
| > 10万 | `IndexIVFPQ` | 乘积量化，牺牲精度换内存 |

**对你现在的 PDF 来说，永远是 FlatL2**——几百个块，精确搜索，没有近似误差。

`search()` 返回两个数组（`vector_store.py:73-76`）：

```python
D, I = index.search(query_vectors, k)   # D=距离(越小越近)，I=行号
```

### 关键：`VectorStore.search` 把 D 丢了

```python
# core/vector_store.py:134-143
D, I = self.index.search(query_embedding, k=k)
for faiss_idx in I[0]:
    original_id = self.id_order[faiss_idx]   # 行号 → chunk_id
    docs.append(self.contents_map[original_id])
```

它只返回 `(docs, doc_ids, metadatas)`，**距离 D 没有传出去**。后果在下一节显现：混合检索拿不到真实相似度，只能用"排第几"当分数。

## 三、BM25：`BM25IndexManager`

```python
# core/bm25_index.py:33-36
self.raw_corpus      = documents
self.doc_mapping     = {i: doc_id}                     # 行号 → chunk_id
self.tokenized_corpus= [list(jieba.cut(d)) for d in documents]
self.bm25_index      = BM25Okapi(self.tokenized_corpus)
```

- **jieba 分词**：中文没有空格，必须先切词。"液压泵异响" → `['液压泵', '异响']`。专业术语若不在 jieba 词典里，会被切碎（"液压泵"可能变成"液压"+"泵"），这是中文 BM25 的常见坑。
- **BM25 分数无上界**，可能在 0–20 之间；且和语义相似度**完全不可比**——这就是为什么混合前必须归一化。
- `search()` 里 `if bm25_scores[idx] > 0`（`bm25_index.py:51`）：**分数为 0 的结果直接丢弃**。所以 BM25 那一路可能只返回 2–3 条，而不是 `top_k` 条。

## 四、两套索引怎么对齐

答案是 **`chunk_id`**，两边各存了一份"行号 → chunk_id"的映射：

```
FAISS:  row 0 ──┐
        row 1 ──┤ id_order[0..N-1]  ──→ chunk_id ──→ contents_map / metadatas_map
        row 2 ──┘

BM25:   doc 0 ──┐
        doc 1 ──┤ doc_mapping{0..N-1} ──→ chunk_id
        doc 2 ──┘
```

之所以必须有这一层，是因为 **FAISS 只认行号**（整数），它不知道"第 3 行是哪个文件哪一段"。`id_order` 就是那本字典（`vector_store.py:97`、`:118`）。

> ⚠️ `build_index` 对 `id_order` 是 `append` 而不是覆盖（`vector_store.py:118`）。如果不清空就调第二次，行号会错位——正确性完全依赖调用方先 `clear()`。`rag_demo.py:55-56` 确实先清了，所以没问题；但这是个"靠约定维持"的设计。

## 五、混合打分公式（下一站的主角，先看个大概）

`core/retriever.py:19` 的 `hybrid_merge`：

```
语义分  = 1.0 - (名次 / 结果数)        # retriever.py:49  ← 注意：用的是名次，不是 D
BM25分 = 原始分 / 该路最高分           # retriever.py:59,65
最终   = 0.7 × 语义分 + 0.3 × BM25分   # alpha 来自 config.py:82
```

- 名次第 0 → 语义分 1.0；第 9 → 0.1。**距离信息在这一步彻底丢失**，只剩排序。
- BM25 只在这条路上出现过时，最高只能拿 0.3；语义路排第 0 的能拿 0.7。所以**语义路天然占优**，这与 `alpha=0.7` 的意图一致，但意味着 BM25 很难翻盘。

## 实验：让两路检索各跑一次，看它们打架

`tools/inspect_index.py`：建索引（不调 LLM）→ 同一个问题分别用 FAISS 和 BM25 查 → 并排对比。

```powershell
uv run python tools/inspect_index.py "挖掘机维修案例(样例）.pdf" --query "液压泵有异响是什么原因"
```

它会打印：

1. 索引概况：块数、维度、索引类型、向量平均模长（验证"未归一化"）
2. **FAISS Top5**：名次、L2 距离、余弦相似度（我另算了余弦，用来对比 L2 的排序是否一致）、前 60 字
3. **BM25 Top5**：名次、原始分、前 60 字
4. **两路重合数**：`x/5`
5. **混合后的 Top8**：直接调 `hybrid_merge`，看 alpha 如何把两路揉在一起

### 观察点

- **L2 排序 vs 余弦排序**：如果两者顺序不同，就验证了"未归一化 + L2"确实和余弦不等价。
- **BM25 返回了几条**：如果不足 5 条，说明大部分块的 BM25 分为 0（关键词完全没命中）。
- **重合数**：中文语料上通常只有 1–3 条重合。重合越低，混合检索的价值越大——这正是这套架构存在的理由。
- **混合 Top8 里有没有"只被 BM25 捞到"的块**：如果有，看它是不是包含精确术语（型号、故障码），那就是关键词检索不可替代的地方。

### 进阶：换中文向量模型

```powershell
# 改 core/embeddings.py:18 -> EMBED_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
uv run python tools/inspect_index.py "挖掘机维修案例(样例）.pdf" --query "液压泵有异响是什么原因"
```

模型已在缓存里，不用下载。对比 FAISS Top5 的变化——中文语料上通常能明显感到"语义命中"变准。改完记得改回来（或者就这么用着，它比 MiniLM 更适合中文）。

## 自测题（01 的答案 + 02 的新题）

**01-1 答案**：扫描件返回空字符串 `""`；`rag_demo.py:68-69` 抛 `ValueError("文档内容为空或无法提取文本")`，被 `process_multiple_files` 的 `try` 捕获，该文件在结果里标记 `❌ 处理失败`，**其它文件继续处理**。

**01-2 答案**：overlap 是把上一块末尾复制到下一块，同一段文字因此拥有两个不同的 `chunk_id`；两路检索都可能命中，在 `hybrid_merge` 里按 `chunk_id` 合并时**不会去重**（id 不同），于是 Top5 里出现两份几乎一样的内容。当 `top_k` 较小、候选本来就少时，这会挤掉真正相关的内容；但如果这段文字本身信息密度高、需要从不同角度召回，重复反而无害。

**02-1**：为什么 `VectorStore.search` 不返回距离也能工作？代价是什么？
**02-2**：如果一个块的 BM25 分是 8，该路最高分是 20，alpha=0.7，它最终得分是多少？如果它同时被语义路排在第 4（共 10 条），最终得分又是多少？

## 下一站

`03-retrieval.md`：`recursive_retrieval` 的 3 轮改写到底在改什么，重排怎么接进来，Top10 怎么变成 Top5。
