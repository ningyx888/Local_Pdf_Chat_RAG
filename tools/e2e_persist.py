"""
端到端验证索引持久化

流程：上传文档 → 落盘 → 重复上传去重 → 追加第二个文档 → 清空内存（模拟重启）
     → 从磁盘加载 → 向量检索 + BM25 检索仍可用

用法：.venv\\Scripts\\python.exe tools\\e2e_persist.py [--keep]
"""

import os
import shutil
import sys

import jieba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

WORK = os.path.join(ROOT, "data", "_e2e")
os.environ["INDEX_DIR"] = os.path.join(WORK, "index")
os.environ["RERANK_METHOD"] = "none"  # 避免加载重排模型

if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK, exist_ok=True)

import rag_demo  # noqa: E402
from core.vector_store import vector_store  # noqa: E402
from core.bm25_index import bm25_manager  # noqa: E402
from core.embeddings import encode_texts  # noqa: E402
from core import store_io  # noqa: E402

passed = failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}  ->  {detail}")


class FakeFile:
    """模拟 Gradio 上传对象，只需要 .name"""

    def __init__(self, path):
        self.name = path


def noop(*args, **kwargs):
    return None


DOC_A = os.path.join(WORK, "甲方合同.txt")
DOC_B = os.path.join(WORK, "乙方说明.txt")
with open(DOC_A, "w", encoding="utf-8") as f:
    for i in range(1, 21):
        if i % 4 == 0:
            f.write(f"第{i}条 甲方应当在验收后三十日内支付款项，付款周期为三十个工作日。\n")
        else:
            f.write(f"第{i}条 交付地点由双方另行约定，验收标准见附件清单。\n")
with open(DOC_B, "w", encoding="utf-8") as f:
    for i in range(1, 21):
        if i == 7:
            f.write(f"第{i}条 设备的保修期为二十四个月，保修期内免费更换损坏部件。\n")
        else:
            f.write(f"第{i}条 设备的日常巡检由乙方安排专人负责，记录归档备查。\n")

print("\n1) 未启用持久化")
check("返回未启用", rag_demo.load_persisted_index() == "未启用索引持久化")

print("2) 启用持久化，此时磁盘无索引")
rag_demo.enable_persistence()
msg = rag_demo.load_persisted_index()
check("提示知识库为空", "知识库为空" in msg, msg)

print("3) 上传第一个文档")
out, _ = rag_demo.process_multiple_files([FakeFile(DOC_A)], progress=noop)
check("提示已保存", "索引已保存" in out, out)
check("向量库就绪", vector_store.is_ready)
check("保留原始向量", vector_store.has_embeddings)
check("磁盘文件存在", store_io.store_exists())
first_chunks = len(vector_store.id_order)

print("4) 重复上传同一文件")
out, _ = rag_demo.process_multiple_files([FakeFile(DOC_A)], progress=noop)
check("提示已在库中", "已在知识库中" in out, out)
check("块数未增加", len(vector_store.id_order) == first_chunks, len(vector_store.id_order))

print("5) 上传第二个文档")
out, _ = rag_demo.process_multiple_files([FakeFile(DOC_B)], progress=noop)
total_chunks = len(vector_store.id_order)
check("块数增加", total_chunks > first_chunks, total_chunks)
check(
    "共两个文档",
    len({m.get("doc_id") for m in vector_store.metadatas_map.values()}) == 2,
)

print("6) 清空内存，模拟重启")
vector_store.clear()
bm25_manager.clear()
check("内存已空", not vector_store.is_ready)
msg = rag_demo.load_persisted_index()
check("加载成功", "已加载知识库" in msg and "2 个文档" in msg, msg)
check("块数完全恢复", len(vector_store.id_order) == total_chunks, len(vector_store.id_order))
check("BM25 语料恢复", len(bm25_manager.raw_corpus) == total_chunks)

print("7) 重启后的检索")
# 注意：all-MiniLM-L6-v2 是英文模型，对中文的语义区分度很低（相似度都在 0.2 附近），
# 所以这里不校验“排序是否符合语义”，只校验行序对齐——FAISS 返回的 id 必须与
# 直接按向量算余弦的 argmax 一致，否则就是“搜到 A 的向量、取出 B 的文本”。
query_vec = encode_texts(["付款周期是多少"], show_progress=False)
sims = vector_store.embeddings @ query_vec[0]
best_row = int(sims.argmax())
docs, ids, metas = vector_store.search(query_vec, k=3)
check("向量检索有结果", bool(docs), docs[:1])
check("FAISS 行序与 id_order 对齐", ids[0] == vector_store.id_order[best_row], (ids[:1], best_row))
check("取出的文本与 id 对应", docs[0] == vector_store.contents_map[ids[0]])
hits = bm25_manager.search("保修期", top_k=3)
check("BM25 命中保修段落", bool(hits) and "保修期" in hits[0]["content"], hits[:1])

print("8) 元数据完整性")
meta = vector_store.metadatas_map[vector_store.id_order[0]]
check("保留 source", bool(meta.get("source")), meta)
check("保留 added_at", bool(meta.get("added_at")), meta)

print("9) 知识库列表")
table = rag_demo.kb_overview_markdown()
check("列出两个文档", "甲方合同.txt" in table and "乙方说明.txt" in table, table)
check("含文本块统计", f"**{total_chunks}** 个文本块" in table, table)

print("10) 删除一个文档")
state = store_io.load_store()
doc_b_id = next(d["doc_id"] for d in state["documents"] if d["source"] == "乙方说明.txt")
doc_b_chunks = next(d["chunks"] for d in state["documents"] if d["doc_id"] == doc_b_id)
msg = rag_demo.kb_delete_document(doc_b_id)
check("提示删除成功", "已删除" in msg, msg)
check("内存块数减少", len(vector_store.id_order) == total_chunks - doc_b_chunks, len(vector_store.id_order))
check("BM25 同步重建", len(bm25_manager.raw_corpus) == len(vector_store.id_order))
check("磁盘同步更新", len(store_io.load_store()["id_order"]) == len(vector_store.id_order))
# 这里只校验 BM25 能对新语料打分，不校验命中：语料只有几个块时，
# 词项出现在半数以上文档里会让 IDF 退化为 0，分数恒为 0（rank_bm25 的已知行为）。
scores = bm25_manager.bm25_index.get_scores(list(jieba.cut("付款周期")))
check("BM25 对新语料可打分", len(scores) == len(vector_store.id_order), len(scores))

print("11) 清空知识库")
msg = rag_demo.kb_clear_all()
check("提示已清空", "已清空" in msg, msg)
check("磁盘文件已删除", not store_io.store_exists())
check("内存已空", not vector_store.is_ready and not bm25_manager.raw_corpus)

if "--keep" not in sys.argv:
    shutil.rmtree(WORK, ignore_errors=True)

print(f"\n结果：{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
