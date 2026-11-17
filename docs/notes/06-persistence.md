# 06 · 索引持久化：让知识库活过重启

> 目标：搞清"上传一次，之后每次启动都还在"是怎么做到的，
> 以及为什么这件事不能只靠 `faiss.write_index()` 一行解决。

## 问题从哪来

改之前的状态：**索引全在内存里，进程一退就没了。**

```python
# core/vector_store.py（改之前）
self.index = None          # FAISS 对象，进程内有效
self.contents_map = {}     # chunk_id -> 文本
self.metadatas_map = {}    # chunk_id -> 元数据
self.id_order = []         # 行号 -> chunk_id
```

三个地方印证了这一点：

- `rag_demo.py` 旧版每次上传都先 `vector_store.clear()`，等于"这次上传就是全部知识"。
- `core/embeddings.py` 的向量矩阵是 `encode_texts()` 的局部变量，`build_index()` 之后就没人引用了。
- 全仓库搜不到 `write_index` / `read_index` / `save_local`。

所以重启一次就要重新上传、重新向量化——一份 100 页 PDF 大概要等十几秒到一分钟。

## 一、为什么不能只存 FAISS

`faiss.write_index(index, path)` 确实存在，但它只保存**向量本身**，不保存：

| 缺什么 | 后果 |
|---|---|
| `contents_map` | 搜到了行号，却取不出原文 |
| `metadatas_map` | 不知道这段文字来自哪个文件 |
| `id_order` | 行号无法还原成 `chunk_id` |

而且更根本的问题：**FAISS 把数据拷进自己的 C++ 缓冲区后，就不再对外暴露原始数组**。
`index.reconstruct(i)` 能一行行取回，但那是逐条调用，几百个块还行，几万个块就慢了；
而且 IVF/PQ 类型的索引是**有损**的，reconstruct 出来的已经不是原向量。

结论：**落盘的对象是"原始向量 + 三张映射表"，不是 FAISS 对象**。
启动时用这些原始数据重新 `add()` 一次，FAISS 索引就回来了——重建几百个块的 FlatL2 索引是毫秒级的事。

```python
# core/vector_store.py:98-100
# 向量本身也保留一份：FAISS 把数据拷进自己的 C++ 缓冲区后就不再对外暴露，
# 落盘时必须有原始数组可用，否则索引无法持久化。
self.embeddings = None      # numpy float32, 形状 (n, dim)，第 i 行对应 id_order[i]
```

## 二、存储布局

```
data/index/                 ← 已被 .gitignore 忽略
  embeddings.npy   float32 [n, dim]，第 i 行 ↔ id_order[i]
  meta.json        { id_order, contents_map, metadatas_map }
  manifest.json    指纹：schema 版本、模型名、维度、分块参数、块数、更新时间
```

三份文件的分工是刻意的：**npy 存二进制大块，json 存可读的结构，manifest 存"这份索引是在什么配置下生成的"**。

路径可用环境变量覆盖（`core/store_io.py:52-57`）：

```python
INDEX_DIR=data/my_index    # 测试和多实例隔离时用
```

## 三、三个必须处理的坑

### 坑 1：写到一半崩溃 → 半截文件

`np.save()` 直接写目标文件，如果这时进程被杀，留下的是**损坏的 npy**，下次启动读取直接炸。

解法是原子写（`core/store_io.py:69-73`）：

```python
def _atomic_write_bytes(target: Path, writer) -> None:
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as handle:
        writer(handle)
    os.replace(tmp, target)      # 同一文件系统内，这一步是原子的
```

`os.replace()` 在 Windows 和 POSIX 上都是原子的：要么是旧文件，要么是新文件，不存在"半个"。

写顺序也有讲究（`core/store_io.py:130-132`）：

```
1. embeddings.npy    ← 数据
2. meta.json         ← 文本与映射
3. manifest.json     ← 最后写，它存在即代表前两份完整
```

所以 `store_exists()` 要求三份齐全，缺一份就当作"没有索引"。

### 坑 2：换了模型还用旧向量

用 A 模型建的向量，配 B 模型的查询向量去算距离——**结果全是噪音，而且不会报错**。
这种 bug 极难排查，因为一切"看起来在正常工作"。

所以 manifest 里记了四个指纹（`core/store_io.py:83-92`），加载前逐一比对（`:141-167`）：

| 字段 | 为什么记它 |
|---|---|
| `schema_version` | 存储格式变了，旧文件解析会错 |
| `embed_model` | 向量空间变了，旧向量全部作废 |
| `chunk_size` / `chunk_overlap` | 分块变了，`chunk_id` 的语义就变了 |

不匹配时**拒绝加载**并说明原因，而不是静默降级：

```python
# rag_demo.py load_persisted_index()
except store_io.StoreIncompatible as exc:
    return f"⚠️ 已有索引与当前配置不兼容，未加载（{exc}）"
```

启动日志会直接告诉你"向量化模型已变更（索引 all-MiniLM-L6-v2 ≠ 当前 BAAI/bge-small-zh-v1.5），
旧向量与新查询不在同一空间"。这时正确做法是删掉 `data/index/` 重新上传。

> 反过来，这个机制也让"换中文模型"变得安全：换完自动作废旧索引，不会悄悄混用。

### 坑 3：多文档追加 → 行号错位

原来 `build_index()` 对 `id_order` 是 `append`，但每次上传前都 `clear()`，所以从没暴露问题。
现在要"第二份文档追加进已有索引"，行号就必须严格对齐：

```
id_order:  [A0, A1]   →   append B   →   [A0, A1, B0]
embeddings: [[..],[..]]                [[..],[..],[..]]
```

做法是把"重建索引"收敛成一个函数（`core/vector_store.py:120-133`）：

```python
def _rebuild(self, embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    auto_index = AutoFaissIndex(dimension=embeddings.shape[1])
    auto_index.select_index_type(len(embeddings))
    auto_index.add(embeddings)
    self.index = auto_index
    self.embeddings = embeddings
```

`build_index` / `append` / `restore` 三个入口最后都调它。
**行号一旦变化就整体重建**——几百个块重建是毫秒级，没必要为了省这点时间去做局部删除（FAISS 的 `remove_ids` 会让行号前移，映射表就得同步维护，容易出错）。

## 四、为什么 BM25 也要重建

BM25 索引没法序列化（`BM25Okapi` 对象里是 numpy 数组 + 词表，没有官方的存取格式），
但它的构建成本很低：**jieba 分词 + 统计词频**，几百个块是几十毫秒。

所以策略是"不存 BM25，每次从向量库的语料重建"（`rag_demo.py rebuild_bm25_from_store()`）：

```python
bm25_manager.build_index(
    [vector_store.contents_map[chunk_id] for chunk_id in vector_store.id_order],
    list(vector_store.id_order),
)
```

这里有个隐含要求：**BM25 的语料顺序必须和 `id_order` 一致**，
否则 `bm25_manager.doc_mapping[idx]` 会指向错误的 `chunk_id`。
按 `id_order` 遍历生成语料，就天然保证了这一点。

## 五、doc_id 为什么改成内容哈希

改之前：

```python
doc_id = f"doc_{int(time.time())}_{idx}"      # 秒级时间戳
```

两个问题：

1. **同一份文件重复上传 → 两个不同的 doc_id** → 索引里出现两份一模一样的内容，还占两倍空间。
2. **同一秒上传两个文件会撞 id**（虽然加了 `idx` 兜住了，但很脆弱）。

改成 SHA1 内容摘要（`rag_demo.py file_digest()`）：

```python
doc_id = file_digest(file.name)[:12]     # 前 12 位十六进制，碰撞概率可忽略
```

三个好处：重复上传得到同一个 id（可以据此跳过）、文件改名不影响 id、内容变了 id 就变（天然区分版本）。

去重就一行（`rag_demo.py`）：

```python
if PERSIST_INDEX and vector_store.has_document(doc_id):
    processed_results.append(f"⏭️ {file_name}: 已在知识库中，跳过")
    continue
```

## 六、删除文档为什么要全量重建

删掉一个文档 = 从 `id_order` 里去掉它的所有 `chunk_id`，对应的向量行也要去掉。
剩下的行会**整体前移**，所以不能只改映射表，必须重建索引。

`delete_document()` 写成了纯函数（`core/store_io.py:255-277`）：吃一个 state，吐一个新的 state，不改原对象。
好处是可以先算出结果、写盘成功后再替换内存，中途失败不会留下"内存删了、磁盘没删"的中间态。

删空的情况要单独处理——否则会留下一个 0 行的索引文件：

```python
if new_state["id_order"]:
    store_io.save_state(new_state)
    vector_store.restore(new_state)
else:
    store_io.clear_store()
    vector_store.clear()
```

## 七、一个刻意留下的限制：只让 Gradio 入口写盘

`api_router.py` 通过 `import rag_demo` 复用处理函数，但它是**另一个进程**。
如果两个进程都写同一份索引文件，后写的会覆盖先写的，而且彼此的 `id_order` 还不一样。

所以 `PERSIST_INDEX` 默认 `False`，只在 `rag_demo.py` 的 `__main__` 里显式打开：

```python
if __name__ == "__main__":
    enable_persistence()
    print(load_persisted_index())
```

这是本期的取舍：**单写者**。要做多进程共享，需要文件锁 + 版本号，或者干脆换真正的向量库。

## 八、验证方式

单测（不加载模型，秒级）：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_store_io.py -q     # 19 passed
```

端到端（真跑一遍上传 → 重启 → 检索 → 删除 → 清空）：

```powershell
.venv\Scripts\python.exe tools\e2e_persist.py                    # 30 passed
```

端到端脚本里有两处**故意放宽**的断言，原因值得记下来：

- **不校验向量检索的语义排序**。`all-MiniLM-L6-v2` 是英文模型，中文的相似度全挤在 0.2 附近
  （实测"付款周期是多少"对合同段 0.2593、对无关的设备巡检段 0.2596），排序基本是噪音。
  所以只校验**行序对齐**：FAISS 返回的 id 必须等于直接按向量算余弦的 argmax——
  这条不变量一旦破坏，就是"搜到 A 的向量、取出 B 的文本"，那才是持久化真正会引入的 bug。
- **不校验 BM25 是否命中**。语料只有几个块时，一个词项出现在半数以上文档里会让
  IDF 退化为 0，分数恒为 0，被 `bm25_scores[idx] > 0` 过滤掉（`core/bm25_index.py:51`）。
  这是 rank_bm25 的正常行为，不是持久化的 bug。

> 顺带一提：这两条也解释了为什么这个项目在中文上"能用但不够好"——
> 语义路基本靠不住，真正在干活的是 BM25 + 重排。

## 自测题

**06-1**：为什么不能只存 `faiss.write_index()` 的结果？至少说出两个理由。
**06-2**：manifest 里为什么必须记 `embed_model`？如果不记，出问题时会是什么现象？
**06-3**：删除一个文档后，为什么不能只从 `contents_map` 里删 key，而要重建 FAISS 索引？
**06-4**：`_atomic_write_bytes` 里 `os.replace()` 换成 `shutil.move()` 会有什么风险？
**06-5**：如果把 `PERSIST_INDEX` 打开在 `api_router` 里，会发生什么？怎么发现？

## 下一站

索引能存了，但"什么内容值得进索引"还没优化：
`03-retrieval.md` 讲 `recursive_retrieval` 的 3 轮改写、重排怎么接进来、Top10 怎么变成 Top5。
