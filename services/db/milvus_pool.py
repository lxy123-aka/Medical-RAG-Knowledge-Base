"""
Milvus 连接池管理（规则3/4: DataType 枚举、维度一致、健康检查 + 自动重连）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.milvus_schema import (
    CollectionNames,
    EMBEDDING_DIM,
    SCHEMA_REGISTRY,
    INDEX_PARAMS,
    SEARCH_PARAMS,
    validate_dimension_consistency,
)

logger = logging.getLogger(__name__)


class MilvusPool:
    """
    Milvus 连接池封装

    规则3: 字段类型必须使用 DataType 枚举，维度一致
    规则4: 健康检查探针 + 自动重连
    """

    def __init__(self):
        self._client = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """初始化连接（线程安全）"""
        async with self._lock:
            if self._initialized:
                return

            settings = get_settings()
            try:
                from pymilvus import MilvusClient

                self._client = MilvusClient(
                    uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
                    token=f"{settings.milvus_user}:{settings.milvus_password}",
                    db_name=settings.milvus_db_name,
                )
                # 验证维度一致性
                validate_dimension_consistency()
                self._initialized = True
                logger.info(
                    f"Milvus 连接初始化成功: {settings.milvus_host}:{settings.milvus_port}"
                )
            except Exception as e:
                logger.error(f"Milvus 连接初始化失败: {e}")
                raise

    async def close(self) -> None:
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._initialized = False
            logger.info("Milvus 连接已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查探针（规则4）"""
        start = time.time()
        try:
            if not self._initialized:
                await self.initialize()

            # 轻量查询验证连通性
            collections = self._client.list_collections()
            latency_ms = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "details": f"Milvus 连接正常，集合数: {len(collections)}",
            }
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"Milvus 健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "details": f"Milvus 健康检查失败: {str(e)}",
            }

    async def ensure_collections(self) -> None:
        """确保所有集合存在，不存在则创建"""
        if not self._initialized:
            await self.initialize()

        existing = set(self._client.list_collections())

        for name, schema_builder in SCHEMA_REGISTRY.items():
            if name not in existing:
                schema = schema_builder()
                self._client.create_collection(
                    collection_name=name,
                    schema=schema,
                )
                logger.info(f"Milvus 创建集合: {name}")

                # 创建向量索引（pymilvus 2.4.x 需用 prepare_index_params 构造）
                idx_params = self._client.prepare_index_params()
                idx_params.add_index(
                    field_name="embedding",
                    index_type=INDEX_PARAMS["index_type"],
                    metric_type=INDEX_PARAMS["metric_type"],
                    params=INDEX_PARAMS["params"],
                )
                self._client.create_index(
                    collection_name=name,
                    index_params=idx_params,
                )
                logger.info(f"Milvus 创建索引: {name}.embedding")
            else:
                logger.info(f"Milvus 集合已存在: {name}")

    async def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        向量检索（规则9: min_score 阈值按集合分别调优）

        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            top_k: 返回数量
            min_score: 最低相似度阈值（规则9）
            filter_expr: 过滤表达式（权限过滤等）
            output_fields: 输出字段列表

        Returns:
            每条查询对应的结果列表
        """
        if not self._initialized:
            await self.initialize()

        try:
            search_params = SEARCH_PARAMS.copy()
            results = self._client.search(
                collection_name=collection_name,
                data=query_vectors,
                limit=top_k,
                filter=filter_expr or "",
                output_fields=output_fields or [],
                search_params=search_params,
            )

            # 过滤低于 min_score 的结果
            filtered_results = []
            for hits in results:
                filtered_hits = []
                for hit in hits:
                    score = hit.get("distance", 0.0)
                    if score >= min_score:
                        filtered_hits.append(hit)
                filtered_results.append(filtered_hits)

            # 规则19: 空结果告警
            total_results = sum(len(r) for r in filtered_results)
            if total_results == 0:
                logger.warning(
                    f"[Milvus] 检索返回空结果 | 集合: {collection_name} | "
                    f"top_k: {top_k} | min_score: {min_score} | "
                    f"filter: {filter_expr}"
                )

            return filtered_results

        except Exception as e:
            logger.error(f"[Milvus] 检索失败: {e}")
            raise

    async def insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """插入数据"""
        if not self._initialized:
            await self.initialize()

        try:
            result = self._client.insert(
                collection_name=collection_name,
                data=data,
            )
            return result
        except Exception as e:
            logger.error(f"[Milvus] 插入失败: {e}")
            raise

    async def delete(
        self,
        collection_name: str,
        filter_expr: str,
    ) -> None:
        """删除数据"""
        if not self._initialized:
            await self.initialize()

        try:
            self._client.delete(
                collection_name=collection_name,
                filter=filter_expr,
            )
        except Exception as e:
            logger.error(f"[Milvus] 删除失败: {e}")
            raise

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self._initialized:
            await self.initialize()

        try:
            stats = self._client.get_collection_stats(collection_name)
            return stats
        except Exception as e:
            logger.error(f"[Milvus] 获取统计信息失败: {e}")
            return {}


# ── 全局单例 ─────────────────────────────────────────────────
_milvus_pool: Optional[MilvusPool] = None


def get_milvus_pool() -> MilvusPool:
    """获取 Milvus 连接池全局单例"""
    global _milvus_pool
    if _milvus_pool is None:
        _milvus_pool = MilvusPool()
    return _milvus_pool
