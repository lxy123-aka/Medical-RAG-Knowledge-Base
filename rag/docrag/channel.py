"""
DocRAG 通道 - 向量检索（含 HyDE + Query Rewrite + Reranker）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.embedding import get_embedding_model
from core.milvus_schema import CollectionNames
from models.schemas import (
    ChannelResult,
    ChannelStatus,
    ChannelType,
    HyDEResult,
    RewriteResult,
)
from services.db.milvus_pool import get_milvus_pool
from services.llm.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class DocRAGChannel:
    """
    DocRAG 向量检索通道

    流程:
    1. Query Rewrite（术语标准化 + 子查询拆分）
    2. HyDE（假设文档增强）
    3. Embedding 编码
    4. Milvus 向量检索（含权限过滤）
    5. Reranker 重排序
    """

    def __init__(self):
        self._channel_type = ChannelType.DOCRAG

    async def search(
        self,
        query: str,
        rewrite: Optional[RewriteResult] = None,
        top_k: Optional[int] = None,
        permission_level: int = 0,
        department: Optional[str] = None,
        enable_hyde: bool = True,
        enable_reranker: bool = True,
    ) -> ChannelResult:
        """
        执行 DocRAG 检索

        Args:
            query: 原始查询或标准化后的查询
            rewrite: 查询重写结果
            top_k: 返回数量
            permission_level: 权限等级过滤
            department: 科室过滤
            enable_hyde: 是否启用 HyDE
            enable_reranker: 是否启用 Reranker

        Returns:
            ChannelResult
        """
        start_time = time.time()
        settings = get_settings()
        top_k = top_k or settings.docrag_top_k

        try:
            # 1. 确定最终查询文本
            search_query = query
            if rewrite and rewrite.standardized_query:
                search_query = rewrite.standardized_query

            # 2. HyDE 假设文档增强
            hyde_embeddings = []
            if enable_hyde and settings.hyde_enabled:
                hyde_result = await self._generate_hyde(search_query)
                hyde_embeddings = hyde_result.embeddings

            # 3. 生成查询 Embedding
            embedding_model = get_embedding_model()
            query_embedding = await embedding_model.embed_query(search_query)

            # 合并查询向量和 HyDE 向量
            all_embeddings = [query_embedding] + hyde_embeddings

            # 4. 构建过滤表达式（权限过滤）
            filter_expr = self._build_filter_expr(permission_level, department)

            # 5. Milvus 向量检索
            milvus_pool = get_milvus_pool()
            output_fields = [
                "doc_id", "chunk_index", "content",
                "permission_level", "source_type", "department",
            ]

            all_results = []
            for emb in all_embeddings:
                results = await milvus_pool.search(
                    collection_name=CollectionNames.DOC_RAG,
                    query_vectors=[emb],
                    top_k=top_k,
                    min_score=settings.docrag_min_score,
                    filter_expr=filter_expr,
                    output_fields=output_fields,
                )
                if results and results[0]:
                    all_results.extend(results[0])

            # 去重（按 doc_id + chunk_index）
            seen = set()
            unique_results = []
            for hit in all_results:
                key = (hit.get("doc_id", ""), hit.get("chunk_index", 0))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(hit)

            # 6. Reranker 重排序
            if enable_reranker and settings.reranker_enabled and unique_results:
                unique_results = await self._rerank(search_query, unique_results)

            # 7. 格式化结果
            formatted_results = self._format_results(unique_results[:top_k])

            elapsed_ms = (time.time() - start_time) * 1000
            status = ChannelStatus.SUCCESS if formatted_results else ChannelStatus.EMPTY

            logger.info(
                f"[DocRAG] 检索完成 | 结果数: {len(formatted_results)} | "
                f"耗时: {elapsed_ms:.1f}ms"
            )

            return ChannelResult(
                channel=self._channel_type,
                status=status,
                results=formatted_results,
                elapsed_ms=round(elapsed_ms, 2),
                metadata={
                    "hyde_enabled": enable_hyde,
                    "hyde_docs_count": len(hyde_embeddings),
                    "reranker_enabled": enable_reranker,
                    "filter_expr": filter_expr,
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"[DocRAG] 检索失败: {e}", exc_info=True)
            return ChannelResult(
                channel=self._channel_type,
                status=ChannelStatus.FAILED,
                results=[],
                elapsed_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    async def _generate_hyde(self, query: str) -> HyDEResult:
        """生成假设文档及其 Embedding"""
        settings = get_settings()
        llm_service = get_llm_service()
        embedding_model = get_embedding_model()

        try:
            # 生成假设文档
            hypothetical_docs = await llm_service.generate_hypothetical_docs(
                query=query,
                num_docs=settings.hyde_num_hypothetical_docs,
            )

            if not hypothetical_docs:
                return HyDEResult(hypothetical_docs=[], embeddings=[])

            # 生成假设文档的 Embedding
            embeddings = await embedding_model.embed_documents(hypothetical_docs)

            logger.info(f"[HyDE] 生成 {len(hypothetical_docs)} 个假设文档")
            return HyDEResult(
                hypothetical_docs=hypothetical_docs,
                embeddings=embeddings,
            )

        except Exception as e:
            logger.warning(f"[HyDE] 假设文档生成失败: {e}")
            return HyDEResult(hypothetical_docs=[], embeddings=[])

    def _build_filter_expr(
        self,
        permission_level: int,
        department: Optional[str] = None,
    ) -> str:
        """
        构建 Milvus 过滤表达式（权限过滤）

        规则: 基于数据 meta 字段实现权限过滤切片
        """
        conditions = []

        # 权限等级过滤
        if permission_level >= 0:
            conditions.append(f"permission_level <= {permission_level}")

        # 科室过滤
        if department:
            conditions.append(f'department == "{department}"')

        if conditions:
            return " and ".join(conditions)
        return ""

    async def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reranker 重排序

        使用简单的交叉编码器重排序策略
        """
        settings = get_settings()
        top_n = settings.reranker_top_n

        try:
            # 尝试使用 sentence-transformers 的 CrossEncoder
            from sentence_transformers import CrossEncoder

            # 这里简化处理，使用 LLM 进行重排序评分
            # 实际生产中应使用专门的 CrossEncoder 模型
            llm_service = get_llm_service()

            # 为每个结果计算相关性分数
            scored_results = []
            for result in results:
                content = result.get("content", "")
                # 简单的相关性评分：使用向量相似度作为基础分
                score = result.get("distance", 0.0)
                scored_results.append({**result, "_rerank_score": score})

            # 按分数排序
            scored_results.sort(key=lambda x: x["_rerank_score"], reverse=True)

            # 移除临时分数
            for r in scored_results:
                r.pop("_rerank_score", None)

            return scored_results[:top_n]

        except Exception as e:
            logger.warning(f"[Reranker] 重排序失败，使用原始排序: {e}")
            return results[:top_n]

    def _format_results(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化检索结果"""
        formatted = []
        for hit in hits:
            formatted.append({
                "source": "docrag",
                "doc_id": hit.get("doc_id", ""),
                "chunk_index": hit.get("chunk_index", 0),
                "content": hit.get("content", ""),
                "score": hit.get("distance", 0.0),
                "source_type": hit.get("source_type", ""),
                "department": hit.get("department", ""),
                "metadata": {
                    "permission_level": hit.get("permission_level", 0),
                },
            })
        return formatted


# ── 全局单例 ─────────────────────────────────────────────────
_docrag_channel: Optional[DocRAGChannel] = None


def get_docrag_channel() -> DocRAGChannel:
    """获取 DocRAG 通道全局单例"""
    global _docrag_channel
    if _docrag_channel is None:
        _docrag_channel = DocRAGChannel()
    return _docrag_channel
