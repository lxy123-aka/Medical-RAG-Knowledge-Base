"""
Neo4j 异步连接池管理（规则2/4: 禁止 fetch(-1)，统一 data()；健康检查 + 自动重连）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.neo4j_schema import get_all_node_labels, get_all_rel_types, build_schema_summary

logger = logging.getLogger(__name__)


class Neo4jPool:
    """
    Neo4j 异步连接池封装

    规则2: 禁止 await result.fetch(n=-1)，统一使用 await result.data()
    规则4: 健康检查探针 + 连接失败自动重连
    """

    def __init__(self):
        self._driver = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._last_health_check: float = 0
        self._health_check_interval: float = 60.0  # 60秒健康检查间隔

    async def initialize(self) -> None:
        """初始化连接池（线程安全，规则18）"""
        async with self._lock:
            if self._initialized:
                return

            settings = get_settings()
            try:
                from neo4j import AsyncGraphDatabase

                self._driver = AsyncGraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60,
                )
                # 验证连接
                await self._driver.verify_connectivity()
                self._initialized = True
                logger.info(f"Neo4j 连接池初始化成功: {settings.neo4j_uri}")
            except Exception as e:
                logger.error(f"Neo4j 连接池初始化失败: {e}")
                raise

    async def close(self) -> None:
        """关闭连接池"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info("Neo4j 连接池已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查探针（规则4）
        返回 {"status": "healthy/unhealthy", "latency_ms": ..., "details": ...}
        """
        start = time.time()
        try:
            if not self._initialized:
                await self.initialize()

            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS health")
                records = await result.data()
                assert len(records) == 1 and records[0]["health"] == 1

            latency_ms = (time.time() - start) * 1000
            self._last_health_check = time.time()
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "details": "Neo4j 连接正常",
            }
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"Neo4j 健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "details": f"Neo4j 健康检查失败: {str(e)}",
            }

    async def execute_read(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行只读查询（规则2: 统一使用 result.data()）

        Args:
            cypher: Cypher 查询语句
            parameters: 参数
            database: 数据库名

        Returns:
            查询结果列表（每条记录为 dict）
        """
        return await self._execute(cypher, parameters, database, write=False)

    async def execute_write(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """执行写入查询"""
        return await self._execute(cypher, parameters, database, write=True)

    async def _execute(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]],
        database: Optional[str],
        write: bool,
    ) -> List[Dict[str, Any]]:
        """
        统一执行逻辑（规则2: 禁止 fetch(-1)，统一 data()）
        规则19: 静默失效比显式报错危险，查询结果必须非空校验
        """
        if not self._initialized:
            await self.initialize()

        settings = get_settings()
        db = database or settings.neo4j_database

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with self._driver.session(database=db) as session:
                    if write:
                        result = await session.run(cypher, parameters or {})
                    else:
                        result = await session.run(cypher, parameters or {})

                    # 规则2: 统一使用 result.data() 获取全部结果
                    records = await result.data()

                    # 规则19: 结果非空校验（对于写入操作可以为空）
                    if not write and not records:
                        logger.warning(
                            f"[Neo4j] 查询返回空结果 | Cypher: {cypher[:100]} | "
                            f"Params: {parameters}"
                        )

                    return records

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"[Neo4j] 查询失败 (尝试 {attempt + 1}/{max_retries + 1}): {error_msg}"
                )

                if attempt < max_retries:
                    # 尝试重连
                    try:
                        await self.close()
                        await self.initialize()
                        logger.info("[Neo4j] 重连成功，重试查询")
                    except Exception as reconnect_err:
                        logger.error(f"[Neo4j] 重连失败: {reconnect_err}")
                        raise

                if attempt == max_retries:
                    logger.error(
                        f"[Neo4j] 查询最终失败 | Cypher: {cypher[:200]} | Error: {error_msg}"
                    )
                    raise

        return []  # 不应到达此处

    async def validate_schema(self) -> List[str]:
        """
        启动时 Schema 校验（规则1: 自动比对属性名一致性）
        返回不一致项列表，空列表表示全部一致
        """
        issues = []
        try:
            # 检查数据库中实际存在的节点标签
            result = await self.execute_read(
                "CALL db.labels() YIELD label RETURN collect(label) AS labels"
            )
            if result:
                actual_labels = set(result[0].get("labels", []))
                expected_labels = set(get_all_node_labels())

                missing = expected_labels - actual_labels
                extra = actual_labels - expected_labels

                if missing:
                    issues.append(f"Schema 定义了但数据库中不存在的标签: {missing}")
                if extra:
                    logger.info(f"数据库中存在但 Schema 未定义的标签（可忽略）: {extra}")

            # 检查关系类型
            result = await self.execute_read(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN collect(relationshipType) AS types"
            )
            if result:
                actual_types = set(result[0].get("types", []))
                expected_types = set(get_all_rel_types())

                missing = expected_types - actual_types
                if missing:
                    issues.append(f"Schema 定义了但数据库中不存在的关系类型: {missing}")

        except Exception as e:
            issues.append(f"Schema 校验执行失败: {e}")

        return issues


# ── 全局单例 ─────────────────────────────────────────────────
_neo4j_pool: Optional[Neo4jPool] = None


def get_neo4j_pool() -> Neo4jPool:
    """获取 Neo4j 连接池全局单例"""
    global _neo4j_pool
    if _neo4j_pool is None:
        _neo4j_pool = Neo4jPool()
    return _neo4j_pool
