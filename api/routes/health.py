"""
API 路由 - 健康检查
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter

from config.settings import get_settings
from models.schemas import HealthStatus, SystemHealthResponse
from services.db.neo4j_pool import get_neo4j_pool
from services.db.milvus_pool import get_milvus_pool
from services.db.postgres_pool import get_postgres_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["健康检查"])


@router.get("/", response_model=SystemHealthResponse)
async def system_health() -> SystemHealthResponse:
    """
    系统整体健康检查

    检查所有数据库连接状态（规则4: 健康检查探针）
    """
    settings = get_settings()
    services = []

    # Neo4j 健康检查
    try:
        neo4j_pool = get_neo4j_pool()
        neo4j_health = await neo4j_pool.health_check()
        services.append(HealthStatus(
            service="neo4j",
            status=neo4j_health["status"],
            latency_ms=neo4j_health.get("latency_ms"),
            details=neo4j_health.get("details"),
        ))
    except Exception as e:
        services.append(HealthStatus(
            service="neo4j",
            status="unhealthy",
            details=f"检查失败: {str(e)}",
        ))

    # Milvus 健康检查
    try:
        milvus_pool = get_milvus_pool()
        milvus_health = await milvus_pool.health_check()
        services.append(HealthStatus(
            service="milvus",
            status=milvus_health["status"],
            latency_ms=milvus_health.get("latency_ms"),
            details=milvus_health.get("details"),
        ))
    except Exception as e:
        services.append(HealthStatus(
            service="milvus",
            status="unhealthy",
            details=f"检查失败: {str(e)}",
        ))

    # PostgreSQL 健康检查
    try:
        postgres_pool = get_postgres_pool()
        pg_health = await postgres_pool.health_check()
        services.append(HealthStatus(
            service="postgresql",
            status=pg_health["status"],
            latency_ms=pg_health.get("latency_ms"),
            details=pg_health.get("details"),
        ))
    except Exception as e:
        services.append(HealthStatus(
            service="postgresql",
            status="unhealthy",
            details=f"检查失败: {str(e)}",
        ))

    # 判断整体状态
    statuses = [s.status for s in services]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "healthy" for s in statuses):
        overall = "degraded"
    else:
        overall = "unhealthy"

    return SystemHealthResponse(
        overall_status=overall,
        services=services,
        version=settings.app_version,
    )


@router.get("/neo4j")
async def neo4j_health() -> Dict[str, Any]:
    """Neo4j 单独健康检查"""
    try:
        pool = get_neo4j_pool()
        return await pool.health_check()
    except Exception as e:
        return {"status": "unhealthy", "details": str(e)}


@router.get("/milvus")
async def milvus_health() -> Dict[str, Any]:
    """Milvus 单独健康检查"""
    try:
        pool = get_milvus_pool()
        return await pool.health_check()
    except Exception as e:
        return {"status": "unhealthy", "details": str(e)}


@router.get("/postgresql")
async def postgresql_health() -> Dict[str, Any]:
    """PostgreSQL 单独健康检查"""
    try:
        pool = get_postgres_pool()
        return await pool.health_check()
    except Exception as e:
        return {"status": "unhealthy", "details": str(e)}
