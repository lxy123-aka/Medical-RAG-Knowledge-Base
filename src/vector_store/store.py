"""
向量存储模块 - Milvus 入库与检索封装

设计说明：
1. 封装 Milvus 操作，提供面向业务的高层接口
2. 文档分块 → 向量化 → 入库 一站式完成
3. 检索支持：普通向量检索 + 带摘要增强检索
4. 返回结果附带文档来源溯源信息

使用方法：
    store = VectorStore()
    await store.initialize()
    # 入库
    await store.ingest_chunks(chunks, embedding_model)
    # 检索
    results = await store.search(query_vector, top_k=5)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量存储封装

    基于现有 MilvusPool 提供面向业务的高层接口：
    - 文档分块批量入库（向量化 + 写入）
    - 语义检索（返回原文 + 来源溯源）
    """

    def __init__(self):
        self._milvus_pool = None
        self._collection_name = "doc_rag_chunks"
        self._initialized = False

    async def initialize(self) -> None:
        """初始化，获取 Milvus 连接"""
        if self._initialized:
            return
        from services.db.milvus_pool import get_milvus_pool
        self._milvus_pool = get_milvus_pool()
        if not self._milvus_pool._initialized:
            await self._milvus_pool.initialize()
            await self._milvus_pool.ensure_collections()
        self._initialized = True
        logger.info("VectorStore 初始化完成")

    async def ingest_chunks(
        self,
        chunks: list,
        embedding_model,
        doc_id: Optional[str] = None,
        source_type: str = "textbook",
        department: str = "",
        permission_level: int = 0,
    ) -> Dict[str, Any]:
        """
        批量入库文档分块

        Args:
            chunks: Chunk 对象列表（来自 MedicalSemanticChunker）
            embedding_model: EmbeddingModel 实例
            doc_id: 文档ID（可选，用于过滤）
            source_type: 来源类型 textbook/record/pharmacopoeia
            department: 所属科室
            permission_level: 权限等级

        Returns:
            入库统计信息
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return {"inserted": 0, "message": "无分块数据"}

        settings = get_settings()
        total_inserted = 0
        batch_size = 64  # 批量入库大小

        logger.info(f"开始入库: {len(chunks)} 个分块, 文档ID={doc_id}")

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]

            # 1. 批量生成向量（使用带摘要的内容，提升语义质量）
            texts = [chunk.content_with_summary for chunk in batch]
            embeddings = await embedding_model.embed_documents(texts)

            # 2. 构造入库数据
            import datetime
            now_str = datetime.datetime.now().isoformat()

            records = []
            for chunk, embedding in zip(batch, embeddings):
                records.append({
                    "doc_id": chunk.doc_id or doc_id or "unknown",
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": embedding,
                    "permission_level": permission_level,
                    "source_type": source_type,
                    "department": department,
                    "created_at": now_str,
                })

            # 3. 写入 Milvus
            try:
                result = await self._milvus_pool.insert(
                    collection_name=self._collection_name,
                    data=records,
                )
                insert_count = result.get("insert_count", len(records)) if isinstance(result, dict) else len(records)
                total_inserted += insert_count
                logger.info(f"  批次 {batch_start // batch_size + 1}: 入库 {insert_count} 条")
            except Exception as e:
                logger.error(f"  批次入库失败: {e}")
                raise

        stats = {
            "inserted": total_inserted,
            "total_chunks": len(chunks),
            "doc_id": doc_id,
        }
        logger.info(f"入库完成: {stats}")
        return stats

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        min_score: float = 0.3,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量语义检索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            min_score: 最低相似度阈值
            filter_expr: 过滤表达式（如权限过滤）

        Returns:
            检索结果列表，每条包含 content, score, source 等溯源信息
        """
        if not self._initialized:
            await self.initialize()

        settings = get_settings()
        start_time = time.time()

        try:
            results = await self._milvus_pool.search(
                collection_name=self._collection_name,
                query_vectors=[query_vector],
                top_k=top_k,
                min_score=min_score,
                filter_expr=filter_expr,
                output_fields=["content", "doc_id", "chunk_index", "source_type", "department", "created_at"],
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 格式化结果
            formatted = []
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    formatted.append({
                        "content": entity.get("content", ""),
                        "score": hit.get("distance", 0.0),
                        "source": {
                            "doc_id": entity.get("doc_id", ""),
                            "chunk_index": entity.get("chunk_index", 0),
                            "source_type": entity.get("source_type", ""),
                            "department": entity.get("department", ""),
                        },
                        "retrieval_method": "vector",
                    })

            logger.info(
                f"向量检索完成: 返回 {len(formatted)} 条结果, "
                f"耗时 {elapsed_ms:.1f}ms"
            )
            return formatted

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    async def search_with_queries(
        self,
        query_texts: List[str],
        embedding_model,
        top_k: int = 5,
        min_score: float = 0.3,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        文本查询 → 向量化 → 检索（一站式接口）

        Args:
            query_texts: 查询文本列表（支持多查询融合）
            embedding_model: EmbeddingModel 实例
            top_k: 每个查询返回数量
            min_score: 最低阈值
            filter_expr: 过滤表达式

        Returns:
            去重后的检索结果列表
        """
        all_results = []
        seen_content = set()

        for text in query_texts:
            # 向量化
            vec = await embedding_model.embed_query(text)
            # 检索
            results = await self.search(
                query_vector=vec,
                top_k=top_k,
                min_score=min_score,
                filter_expr=filter_expr,
            )
            # 去重（基于内容前100字符）
            for r in results:
                content_key = r["content"][:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_results.append(r)

        # 按分数排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """按文档ID删除向量数据"""
        if not self._initialized:
            await self.initialize()

        try:
            await self._milvus_pool.delete(
                collection_name=self._collection_name,
                filter_expr=f'doc_id == "{doc_id}"',
            )
            logger.info(f"已删除文档 {doc_id} 的向量数据")
        except Exception as e:
            logger.error(f"删除文档向量数据失败: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        if not self._initialized:
            await self.initialize()

        try:
            stats = await self._milvus_pool.get_collection_stats(self._collection_name)
            return {
                "collection": self._collection_name,
                "row_count": stats.get("row_count", 0),
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"collection": self._collection_name, "error": str(e)}
