# 00 · 全景：一次提问的一生

> 这一页只做一件事：给你一张能挂住后面所有细节的地图。代码细节留给 01–04。

## 两条时间线

整个系统其实只有两条线，它们**不共享执行过程**，只共享内存里的 3 份状态：

```
建库线（上传时跑一次）                     问答线（每次提问跑一次）
─────────────────────────                ─────────────────────────
PDF ─→ 文本 ─→ 分块 ─→ 向量 ─┐           问题 ─→ 向量 ─┐
                              ├─→ FAISS 索引 ───────────┤
             分块 ─→ jieba ───┴─→ BM25  索引 ───────────┤
                                                         ↓
                                              混合检索 → 重排 → Top5
                                                         ↓
                                              Prompt 拼装 → LLM → 答案
```

## 建库线（`rag_demo.py:414` 上传按钮）

```
upload_btn.click → process_multiple_files(files)          rag_demo.py:48
  ├─ 1. vector_store.clear() + bm25_manager.clear()       rag_demo.py:55-56   ← 每次都清空
  ├─ 2. 逐文件：extract_text(file.name)                   rag_demo.py:67      → 纯文本
  │           split_text(text)                            rag_demo.py:71      → list[str]
  │           生成 doc_id / chunk_id                      rag_demo.py:72-74
  ├─ 3. 所有分块一次性 encode_texts(all_chunks)            rag_demo.py:87      → (N, 384) 矩阵
  ├─ 4. vector_store.build_index(块, ids, metas, 向量)     rag_demo.py:90
  └─ 5. bm25_manager.build_index(块, ids)                 rag_demo.py:93
```

注意第 3 步：**不是边切边算**，而是把所有文件的所有块攒在 `all_chunks` 里一次性编码（`rag_demo.py:60`）。所以日志里那句 `Batches: 1/1` 就是这一次批量编码。

## 问答线（`rag_demo.py:415` 提问按钮）

```
ask_btn.click → process_chat(question, history, web_search, model)   rag_demo.py:337
  └─ query_answer(question, enable_web_search, model_choice)         rag_demo.py:356
       └─ core/generator.py:221 内部按顺序做：
          ① recursive_retrieval()   检索 + 最多 3 轮 LLM 改写
          ② hybrid_merge()          语义/关键词两路融合（alpha=0.7）
          ③ rerank_results()        重排取 Top 5
          ④ _build_prompt()         拼上下文 + 引用编号 + 防幻觉规则
          ⑤ _call_openai_compatible_api()   → 智谱 / SiliconFlow / Ollama
          ⑥ 拼「答案 + 来源列表」返回
```

`process_chat` 只做两件事：把问题和答案 append 进 `history`，清空输入框（`rag_demo.py:361-363`）。真正的活全在 `query_answer` 里。

## 三份内存状态（系统唯一的"记忆"）

| 状态 | 定义位置 | 存什么 | 写 | 读 |
|---|---|---|---|---|
| `vector_store` | `core/vector_store.py` | FAISS 索引 + `contents_map`(id→原文) + `metadatas_map`(id→来源) + `id_order`(行号→id) | 建库 | 检索、分块可视化页 |
| `bm25_manager` | `core/bm25_index.py:31` | jieba 分词后的语料 + BM25Okapi | 建库 | 检索 |
| `chunk_data_cache` | `rag_demo.py:107` | 分块可视化页的表格缓存 | 点「刷新分块」 | 点击某一行看详情 |

**没有任何持久化**。关掉进程 = 全部丢失，下次要重新上传。这也解释了为什么重启后必须先传文件才能提问。

## 关键参数（`config.py:80-84`）

| 参数 | 值 | 在哪用 | 调大了会怎样 |
|---|---|---|---|
| `CHUNK_SIZE` | 400 字符 | `text_splitter.py:30` | 块更少更整,但检索粒度粗、易混入无关内容 |
| `CHUNK_OVERLAP` | 40 字符 | `text_splitter.py:31` | 跨块信息更完整,但块数↑、索引↑ |
| `HYBRID_ALPHA` | 0.7 | `retriever.py` 融合 | →1 更信语义,→0 更信关键词 |
| `RETRIEVAL_TOP_K` | 10 | 初检候选数 | 候选更多,重排负担↑ |
| `RERANK_TOP_K` | 5 | 送进 prompt 的段数 | 上下文更长,可能超长/更贵 |

## 现在只需记住的三个坑

1. **每次上传都清空索引**（`rag_demo.py:55-56`）。要问多份文档，必须在**一次上传里选多个文件**，不能传完 A 再传 B。
2. **重排器的分类头是随机初始化的**（上一轮日志的 `MISSING` × 4）。检索照常，但 Top5 顺序目前不可信。
3. **无鉴权、无持久化、无 OCR**。扫描版 PDF 会得到空文本 → `文档内容为空或无法提取文本`。

> 仓库自带的 `openwiki/` 里有一套官方文档（英文），结构和本笔记一致但更偏"模块清单"。本笔记补的是**执行顺序 + 参数含义 + 可验证的实验**。

## 下一站

`01-ingest.md`：文本是怎么被切开的，400/40 到底切出了什么。
