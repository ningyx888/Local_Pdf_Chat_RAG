"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型
"""

import logging
import numpy as np
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ


class AutoFaissIndex:
    """
    自动选择 FAISS 索引类型的封装类

    根据数据量自动选择最优索引类型：
    - 小数据集（<1万）: FlatL2（精确搜索）
    - 中等数据集（1万-10万）: IVFFlat（近似搜索）
    - 大数据集（>10万）: IVFPQ（高效近似搜索）
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None
        self.m = None
        self.nprobe = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0

    def select_index_type(self, num_vectors):
        """根据向量数量自动选择最优索引类型"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logging.info(f"选择索引类型: {self.index_type}，向量数: {num_vectors}")
        return self.index_type

    def train(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.train(vectors)

    def add(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    解决原代码中 4 个全局变量的管理问题。
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表
        # 向量本身也保留一份：FAISS 把数据拷进自己的 C++ 缓冲区后就不再对外暴露，
        # 落盘时必须有原始数组可用，否则索引无法持久化。
        self.embeddings = None      # numpy float32, 形状 (n, dim)，第 i 行对应 id_order[i]

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        构建 FAISS 索引

        Args:
            chunks: 文本片段列表
            chunk_ids: 片段 ID 列表
            metadatas: 元数据列表
            embeddings: 向量数组 (numpy, float32)
        """
        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        self._rebuild(embeddings)
        logging.info(f"FAISS 索引构建完成，共 {self.index.ntotal} 个文本块，类型: {self.index.index_type}")

    def _rebuild(self, embeddings):
        """
        用完整向量矩阵重建 FAISS 索引

        追加、删除、从磁盘恢复都复用这一条路径：
        FAISS 的行号就是 id_order 的下标，任何行序变化都必须整体重建，
        否则会出现“搜到的是 A 的向量，取出来却是 B 的文本”。
        """
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        auto_index = AutoFaissIndex(dimension=embeddings.shape[1])
        auto_index.select_index_type(len(embeddings))
        auto_index.add(embeddings)
        self.index = auto_index
        self.embeddings = embeddings

    def append(self, chunks, chunk_ids, metadatas, embeddings):
        """把新文档的块追加到现有索引（多文档累积，而不是覆盖）"""
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        if self.has_embeddings and embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"向量维度不一致：已有 {self.embeddings.shape[1]} 维，新数据 {embeddings.shape[1]} 维"
            )

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        merged = np.vstack([self.embeddings, embeddings]) if self.has_embeddings else embeddings
        self._rebuild(merged)
        logging.info(f"索引追加完成，共 {len(self.id_order)} 个文本块")
        return len(self.id_order)

    def restore(self, state):
        """从持久化快照恢复，state 来自 core/store_io.load_store()"""
        self.clear()
        self.contents_map.update(state["contents_map"])
        self.metadatas_map.update(state["metadatas_map"])
        self.id_order.extend(state["id_order"])
        self._rebuild(state["embeddings"])
        logging.info(f"索引已从磁盘恢复，共 {len(self.id_order)} 个文本块")

    def has_document(self, doc_id):
        """该文档是否已在索引中（上传时用于去重）"""
        return any(meta.get("doc_id") == doc_id for meta in self.metadatas_map.values())

    def search(self, query_embedding, k=10):
        """
        搜索最相似的向量

        Returns:
            (docs, doc_ids, metadatas)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas = [], [], []
            for faiss_idx in I[0]:
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
            return docs, doc_ids, metadatas
        except Exception as e:
            logging.error(f"FAISS 检索错误: {str(e)}")
            return [], [], []

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def has_embeddings(self):
        """是否保留了可落盘的原始向量"""
        return self.embeddings is not None and len(self.embeddings) > 0

    @property
    def total_chunks(self):
        return self.index.ntotal if self.index is not None else 0

    def clear(self):
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        self.embeddings = None
        logging.info("向量存储已清空")


# 模块级单例
vector_store = VectorStore()
