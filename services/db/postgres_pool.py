"""
PostgreSQL 异步连接池管理（规则4: 健康检查 + 自动重连）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class PostgresPool:
    """
    PostgreSQL 异步连接池

    规则4: 健康检查探针 + 连接失败自动重连
    规则18: 单例连接池用 asyncio.Lock 保护初始化
    """

    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """初始化连接池（线程安全）"""
        async with self._lock:
            if self._initialized:
                return

            settings = get_settings()
            try:
                from sqlalchemy.ext.asyncio import (
                    AsyncSession,
                    create_async_engine,
                    async_sessionmaker,
                )

                self._engine = create_async_engine(
                    settings.postgres_dsn,
                    pool_size=settings.postgres_pool_size,
                    max_overflow=20,
                    pool_pre_ping=True,  # 连接前自动 ping 验证
                    pool_recycle=3600,
                    echo=settings.debug,
                )

                self._session_factory = async_sessionmaker(
                    self._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )

                # 验证连接
                async with self._engine.begin() as conn:
                    await conn.execute(
                        __import__("sqlalchemy").text("SELECT 1")
                    )

                self._initialized = True
                logger.info(f"PostgreSQL 连接池初始化成功: {settings.postgres_host}:{settings.postgres_port}")

            except Exception as e:
                logger.error(f"PostgreSQL 连接池初始化失败: {e}")
                raise

    async def close(self) -> None:
        """关闭连接池"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("PostgreSQL 连接池已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查探针（规则4）"""
        start = time.time()
        try:
            if not self._initialized:
                await self.initialize()

            import sqlalchemy
            async with self._engine.begin() as conn:
                result = await conn.execute(sqlalchemy.text("SELECT 1 AS health"))
                row = result.fetchone()
                assert row is not None and row[0] == 1

            latency_ms = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "details": "PostgreSQL 连接正常",
            }
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"PostgreSQL 健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "details": f"PostgreSQL 健康检查失败: {str(e)}",
            }

    async def execute_query(
        self,
        sql: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行查询并返回结果

        Args:
            sql: SQL 查询语句
            parameters: 参数

        Returns:
            查询结果列表（每条记录为 dict）
        """
        if not self._initialized:
            await self.initialize()

        import sqlalchemy
        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                async with self._session_factory() as session:
                    result = await session.execute(
                        sqlalchemy.text(sql),
                        parameters or {},
                    )

                    # 如果有结果集
                    if result.returns_rows:
                        rows = result.fetchall()
                        columns = result.keys()
                        records = [dict(zip(columns, row)) for row in rows]

                        # 规则19: 空结果告警
                        if not records:
                            logger.warning(
                                f"[PostgreSQL] 查询返回空结果 | SQL: {sql[:100]}"
                            )
                        return records
                    else:
                        await session.commit()
                        return []

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"[PostgreSQL] 查询失败 (尝试 {attempt + 1}/{max_retries + 1}): {error_msg}"
                )

                if attempt < max_retries:
                    try:
                        await self.close()
                        await self.initialize()
                        logger.info("[PostgreSQL] 重连成功，重试查询")
                    except Exception as reconnect_err:
                        logger.error(f"[PostgreSQL] 重连失败: {reconnect_err}")
                        raise

                if attempt == max_retries:
                    logger.error(
                        f"[PostgreSQL] 查询最终失败 | SQL: {sql[:200]} | Error: {error_msg}"
                    )
                    raise

        return []

    async def execute_many(
        self,
        sql: str,
        data: List[Dict[str, Any]],
    ) -> int:
        """
        批量执行插入

        Returns:
            插入行数
        """
        if not self._initialized:
            await self.initialize()

        import sqlalchemy
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    sqlalchemy.text(sql),
                    data,
                )
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"[PostgreSQL] 批量插入失败: {e}")
            raise

    def get_session(self):
        """获取异步 Session 上下文管理器"""
        if not self._initialized:
            raise RuntimeError("PostgreSQL 连接池未初始化")
        return self._session_factory()


# ── 全局单例 ─────────────────────────────────────────────────
_postgres_pool: Optional[PostgresPool] = None


def get_postgres_pool() -> PostgresPool:
    """获取 PostgreSQL 连接池全局单例"""
    global _postgres_pool
    if _postgres_pool is None:
        _postgres_pool = PostgresPool()
    return _postgres_pool
