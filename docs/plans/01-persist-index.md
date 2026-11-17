# 索引持久化实施计划

> 目标：文档上传一次即入库，重启后直接提问，不需要重新上传。

## 一、目标与非目标

**目标**

1. 上传 PDF → 解析、分块、向量化 → **写入磁盘**，成为"知识库"的一部分
2. 启动时自动加载已有索引，立即可提问
3. 多次上传累积，而不是互相覆盖
4. 同一文件重复上传不产生重复块
5. 可以查看已入库文档、删除单个文档、清空整个库

**非目标（本期不做）**

- 多用户 / 权限 / 鉴权
- 并发写入（多进程同时写同一索引）
- 云存储、对象存储
- OCR、扫描件解析
- 增量向量化（只给新文档编码，不重算全部）—— 留到后期

## 二、现状（为什么现在做不到）

| 事实 | 位置 |
|---|---|
| 索引是模块级单例，字段全在内存 | `core/vector_store.py:93-97`、`:164-165` |
| 全仓库没有任何 `write_index` / `pickle` / `save` | 无 |
| 每次上传先 `clear()` 再 `build_index()` | `rag_demo.py:56-57`、`:91` |
| doc_id 用秒级时间戳，重传即变 | `rag_demo.py:73` |
| 向量数组是局部变量，build 完即丢 | `rag_demo.py:88` |
| 文本只存在 `contents_map`，进程退出即失 | `core/vector_store.py:95` |

关键缺口：`VectorStore` **不保存向量本身**。`build_index()` 把 numpy 数组交给 FAISS 后就再没有引用，所以即使想落盘也无从取。这是第一步必须补的。

## 三、设计

### 3.1 存储布局

```
data/index/                      ← 新增，加入 .gitignore
  embeddings.npy                 ← float32 [n, 384]，与 id_order 行对齐
  meta.json                      ← contents_map / metadatas_map / id_order / docs
  manifest.json                  ← 指纹：模型名、维度、chunk_size、overlap、schema 版本
```

### 3.2 数据模型

```jsonc
// meta.json
{
  "id_order": ["a1b2c3d4e5f6_chunk_0", "a1b2c3d4e5f6_chunk_1"],
  "contents_map": { "a1b2c3d4e5f6_chunk_0": "文本内容..." },
  "metadatas_map": { "a1b2c3d4e5f6_chunk_0": { "source": "手册.pdf", "doc_id": "a1b2c3d4e5f6" } },
  "docs": [
    { "doc_id": "a1b2c3d4e5f6", "source": "手册.pdf", "chunks": 42,
      "sha1": "a1b2c3d4e5f6...", "added_at": "2026-11-17T10:00:00" }
  ]
}

// manifest.json
{
  "schema_version": 1,
  "embed_model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "chunk_size": 400,
  "chunk_overlap": 40
}
```

`doc_id = sha1(file_bytes)[:12]`：内容不变则 id 不变，天然支持去重与覆盖。

### 3.3 关键流程

**启动**

```
store_exists() ? load_store() : 空库
   ├─ 读 manifest.json → 与当前配置比对
   │    不一致 → 记录警告，不加载（避免用错模型查询）
   └─ 读 embeddings.npy + meta.json → build_index() 重建 FAISS
```

**上传**

```
for file in files:
    sha1 = 内容哈希
    if sha1 in docs: 跳过，提示"已在库中"
    else: 解析 → 分块 → 编码 → 追加到现有向量矩阵

重建 FAISS（追加即可，FlatL2 支持 add）
→ 原子写盘（先写 .tmp，再 os.replace）
```

**删除**

```
保留的 chunk = 所有 id 中 doc_id != 目标
重建 embeddings 矩阵与 FAISS 索引（数据量小，全量重建最简单）
→ 写盘
```

### 3.4 一致性

- **原子写**：`embeddings.npy.tmp` → `os.replace()`，避免写一半崩溃导致文件损坏
- **指纹校验**：`embed_model` / `dimension` / `chunk_size` / `chunk_overlap` 任一变化 → 拒绝加载并提示重建
- **双进程冲突**：`rag_demo.py` 与 `api_router.py` 是两个独立进程，若都启用持久化会互相覆盖。本期**只让 Gradio 入口启用**，API 入口保持内存模式（或后续加文件锁）

## 四、分阶段

| 阶段 | 内容 | 产出 | 预计 |
|---|---|---|---|
| **P0** | doc_id 改内容哈希；`VectorStore` 保留 `self.embeddings` | 改动 `rag_demo.py:73`、`vector_store.py:99-122` | 0.5h |
| **P1** | 新增 `core/store_io.py`：`save/load/exists/delete_doc/check_compat`，原子写 | 新模块 ~180 行 | 1.5h |
| **P2** | 启动自动加载；上传改为追加合并（去掉 `clear()`） | 改 `rag_demo.py:48-101` | 1h |
| **P3** | 文档列表 / 删除单个 / 清空库；UI 按钮 | 改 UI + 2 个处理函数 | 1h |
| **P4** | 指纹校验接入启动流程；`tests/test_store_io.py` | 新测试 ~10 个用例 | 1h |
| **P5** | `docs/notes/06-persistence.md` 记录设计与坑 | 1 篇笔记 | 0.5h |

合计约 5.5 小时，可分批提交：P0+P1 一个提交、P2 一个、P3+P4 一个。

## 五、验收标准

1. 上传 A.pdf → 关闭程序 → 重启 → 直接问 A 的内容，**不重新上传**即可答出
2. 再上传 B.pdf → 问 A、问 B 都能答（累积生效）
3. 重复上传 A.pdf → 提示"已在库中"，向量数不增加
4. 删除 A.pdf → 问 A 返回"知识库中没有相关内容"，问 B 正常
5. 把 `.env` 的 `CHUNK_SIZE` 改成 500 → 启动时给出明确警告，不静默使用旧索引
6. 上传过程中强制结束进程 → 重启后索引仍可用（原子写生效）

## 六、风险

| 风险 | 影响 | 对策 |
|---|---|---|
| `embeddings.npy` 与 `id_order` 行数不一致 | 检索结果错位 | 加载时校验长度，不一致即拒绝加载 |
| 删除后行号映射错乱 | 答非所问 | 删除一律全量重建，不做局部删除 |
| 双进程同时写 | 索引损坏 | 本期只允许 Gradio 入口写 |
| 旧库 doc_id 是时间戳 | 与哈希 id 混存 | 升级时提示清空重建（教学项目数据可弃） |

## 七、实施记录（2026-11-17）

P0–P5 全部完成，验收标准 1–4 已由 `tools/e2e_persist.py` 自动覆盖（30 项断言全过），
验收标准 5 由 `tests/test_store_io.py::test_changed_embed_model_is_rejected` 覆盖，
验收标准 6（强制结束进程）未自动化，靠原子写机制保证。

| 文件 | 改动 |
|---|---|
| `core/vector_store.py` | 新增 `self.embeddings`、`_rebuild()`、`append()`、`restore()`、`has_document()`、`has_embeddings` |
| `core/store_io.py` | 新增，约 300 行：原子写、manifest 指纹、load/save/delete/clear |
| `rag_demo.py` | `file_digest()`、`PERSIST_INDEX` 开关、`load_persisted_index()`、`rebuild_bm25_from_store()`、知识库标签页（列表/删除/清空） |
| `tests/test_store_io.py` | 新增 19 个用例 |
| `tools/check_store.py` | 新增，18 项 store_io 自检 |
| `tools/e2e_persist.py` | 新增，端到端 30 项断言 |
| `docs/notes/06-persistence.md` | 新增设计与坑的记录 |
| `.gitignore` | 新增 `data/` |

实施中发现的额外事实：

- `all-MiniLM-L6-v2` 对中文的语义区分度极低（相关段 0.2593 vs 无关段 0.2596），
  向量路在中文语料上接近随机；真正起作用的是 BM25 + 重排。换 `BAAI/bge-small-zh-v1.5`
  （已在 HF 缓存里）能显著改善，且 manifest 指纹会自动作废旧索引，切换是安全的。
- 语料很小（几个块）时 BM25 的 IDF 会退化为 0，`bm25_scores[idx] > 0` 的过滤会吞掉全部结果，
  这是 rank_bm25 的正常行为，不是 bug。
