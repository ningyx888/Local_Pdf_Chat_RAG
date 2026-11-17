# 01 · 数据进：解析与分块

> 目标：搞清"一份 PDF"是怎么变成"N 个带 id 的字符串"的。
> 这一步决定后面所有环节的上限——切坏了，再好的模型也救不回来。

## 三个参与者

| 文件 | 函数 | 职责 |
|---|---|---|
| `core/document_loader.py:14` | `extract_text(filepath)` | 任意格式 → 纯文本 `str` |
| `core/text_splitter.py:14` | `split_text(text)` | 纯文本 → `list[str]` |
| `rag_demo.py:48` | `process_multiple_files(files)` | 串起来 + 编号 + 送进索引 |

## 第一步：extract_text

`core/document_loader.py:26` 按扩展名分派，**没有统一抽象层**：

| 扩展名 | 实现 | 关键行 |
|---|---|---|
| `.pdf` | `pdfminer.six` 的 `extract_text_to_fp` | `document_loader.py:29-33` |
| `.txt` / `.md` | 直接 `read()`，UTF-8 | `:35-37` |
| `.docx` | `python-docx`，只取 `paragraphs` | `:39-46` |
| `.xlsx` | `pandas`，每个 sheet `to_string()` | `:48-60` |
| `.pptx` | `python-pptx`，遍历 shape.text | `:62-74` |
| 其它 | 打 warning 返回 `""` | `:76-78` |

三个必须知道的后果：

1. **PDF 只提文字层**。`pdfminer` 不做 OCR，扫描件/图片型 PDF 返回空字符串 → 上层在 `rag_demo.py:68-69` 抛 `文档内容为空或无法提取文本`。
2. **表格会散架**。PDF 里的表格被抽成一串按行排列的文字，列对齐信息丢失。
3. **页码丢失**。`extract_text` 返回的是一个纯字符串，页边界信息在函数内部就不存在了。所以后面 `metadata` 里**只有 `source` 和 `doc_id`，没有页码**（`rag_demo.py:73`）。这就是为什么答案引用只能标到"哪个文件"，标不到"第几页"。

> 想加页级引用，唯一路径是换解析器（如 `pymupdf` 的 `page.get_text()`）并在切分前按页保留元数据——这是本项目一个真实的可改进点。

## 第二步：split_text

`core/text_splitter.py:29-33`：

```python
RecursiveCharacterTextSplitter(
    chunk_size=400,          # config.py:80
    chunk_overlap=40,        # config.py:81
    separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
)
```

### "递归"是什么意思

不是递归函数，而是**逐级降级**：先按 `\n\n`（段落）切；切出来的片段如果还超过 400 字符，就按 `\n`（行）再切；还超就按 `。`（句号）切；再不行就按逗号、分号、冒号、空格；最后 `""` 表示按字符硬切。

所以**优先在语义边界断开**，实在没法断才硬切。`separators` 里放了中文标点，是这份实现为中文文档做的唯一适配。

### 400 是"字符"不是 token

中文里 400 字符 ≈ 200–300 token。这个数字偏小但安全：小块召回更精准，代价是上下文碎片化、块数变多。

### overlap 的真相

`chunk_overlap=40` 不是"多切出来一段"，而是**把上一块的末尾 40 个字符复制到下一块的开头**。结果：

- 好处：一句话被 400 的边界劈开时，两边都能看到完整语义
- 代价：同一段文字在索引里出现两次 → 检索时可能召回两个几乎相同的块，挤占 Top5 名额

这就是为什么块总数不是 `总字符/400`，而是略多。

## 第三步：编号与元数据

`rag_demo.py:72-74`：

```python
doc_id    = f"doc_{int(time.time())}_{idx}"        # 例：doc_1763001234_1
metadatas = [{"source": 文件名, "doc_id": doc_id} for _ in chunks]
chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
```

- `idx` 是文件序号（从 1 开始），所以同一秒上传多个文件不会撞 id
- 一个文件的所有块共享同一个 `doc_id` → 这是后面"同一文档去重/聚合"的唯一依据
- `chunk_id` 是全局唯一键，FAISS 的行号 ↔ `chunk_id` 的映射就存在 `vector_store.id_order` 里（`vector_store.py:99` 附近）

## 实验：把分块拆开看

我写了 `tools/inspect_chunks.py`，不碰 LLM、不建索引，只做"读文件 → 切 → 打印统计"：

```powershell
# 用项目自带的样例 PDF
uv run python tools/inspect_chunks.py "挖掘机维修案例(样例）.pdf"

# 对比不同 chunk_size
uv run python tools/inspect_chunks.py "挖掘机维修案例(样例）.pdf" --sizes 200 400 800

# 多打印几块原文
uv run python tools/inspect_chunks.py "挖掘机维修案例(样例）.pdf" --show 3
```

预期看到：

```
提取字符数: 12,345
段落数(空行分隔): 87

--- chunk_size=200, overlap=40 ---
块数: 82 | 最短 12 | 最长 200 | 平均 151 字符
--- chunk_size=400, overlap=40 ---
块数: 44 | 最短 12 | 最长 400 | 平均 281 字符
--- chunk_size=800, overlap=40 ---
块数: 24 | 最短 12 | 最长 800 | 平均 515 字符
```

### 三个观察点

1. **块数不是线性反比**。800 的块数往往不到 400 的一半——因为切分点受段落边界约束，不是均匀填充。
2. **有很短的块**（十几字符）。这些通常是标题、页眉页脚、孤立的一行。它们会污染检索：用户问"液压泵异响"，一个只有"故障排除"四个字的块也可能被召回。
3. **重叠验证**。脚本最后会检查相邻两块的首尾是否真的重叠 40 字符。如果显示 `未检测到重叠`，说明这段文本里 400 字符以内就有天然段落边界，overlap 根本没触发。

### 进阶实验（可选）

把 `config.py:80` 的 `CHUNK_SIZE` 改成 200 或 800，重启、重传同一个 PDF，看：
- 分块可视化页签里的块数变化
- 问同一个问题时，答案引用了几段、是否更容易出现"答非所问"

改完记得改回来。

## 自测题

1. 如果一份 PDF 是扫描件，`extract_text` 会返回什么？系统在哪个环节报错、报什么错？
2. 为什么 `chunk_overlap` 会让检索出现"同一段内容被召回两次"？在什么情况下这不是问题？

（答案我放在 02 的末尾，你先自己想一想。）

## 下一站

`02-index.md`：这些字符串怎么变成 FAISS 里的向量和 BM25 里的词频，以及两套索引怎么对齐同一个 `chunk_id`。