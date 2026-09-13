"""
医典 RAG 多维知识库系统 - FastAPI 启动入口

规则13: 生产用 Pydantic Settings 启动时一次性校验所有必填配置
规则18: 单例数据库连接池在 lifespan 中统一初始化
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings, validate_startup_settings

# ── 日志配置 ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan 管理（规则18: 在 lifespan 中统一初始化） ─────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时:
    1. 校验配置
    2. 初始化数据库连接池
    3. 初始化 Embedding 模型
    4. 初始化 LLM 服务
    5. 初始化评估系统
    6. Schema 校验

    关闭时:
    1. 关闭所有连接池
    """
    settings = get_settings()
    logger.info(f"{'='*60}")
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info(f"  启动模式: {'开发' if settings.debug else '生产'}")
    logger.info(f"{'='*60}")

    # ── 1. 配置校验 ──────────────────────────────────────────
    errors = validate_startup_settings(settings)
    if errors:
        for err in errors:
            logger.warning(f"[配置校验] {err}")
        # 非致命错误（如模型路径不存在）只告警，不阻止启动
        critical_errors = [e for e in errors if "未配置" in e]
        if critical_errors:
            logger.error(f"关键配置缺失，拒绝启动: {critical_errors}")
            sys.exit(1)

    # ── 2. 初始化数据库连接池 ────────────────────────────────
    logger.info("初始化数据库连接池...")

    # Neo4j
    try:
        from services.db.neo4j_pool import get_neo4j_pool
        neo4j_pool = get_neo4j_pool()
        await neo4j_pool.initialize()
        logger.info("  ✓ Neo4j 连接池就绪")
    except Exception as e:
        logger.error(f"  ✗ Neo4j 连接池初始化失败: {e}")

    # Milvus
    try:
        from services.db.milvus_pool import get_milvus_pool
        milvus_pool = get_milvus_pool()
        await milvus_pool.initialize()
        await milvus_pool.ensure_collections()
        logger.info("  ✓ Milvus 连接池就绪")
    except Exception as e:
        logger.error(f"  ✗ Milvus 连接池初始化失败: {e}")

    # PostgreSQL
    try:
        from services.db.postgres_pool import get_postgres_pool
        pg_pool = get_postgres_pool()
        await pg_pool.initialize()
        logger.info("  ✓ PostgreSQL 连接池就绪")
    except Exception as e:
        logger.error(f"  ✗ PostgreSQL 连接池初始化失败: {e}")

    # ── 3. 初始化 Embedding 模型 ─────────────────────────────
    try:
        from core.embedding import get_embedding_model
        embedding_model = get_embedding_model()
        await embedding_model.initialize()
        logger.info(f"  ✓ Embedding 模型就绪 (模式: {embedding_model.mode})")
    except Exception as e:
        logger.error(f"  ✗ Embedding 模型初始化失败: {e}")

    # ── 4. 初始化 LLM 服务 ──────────────────────────────────
    try:
        from services.llm.llm_service import get_llm_service
        llm_service = get_llm_service()
        await llm_service.initialize()
        logger.info("  ✓ LLM 服务就绪")
    except Exception as e:
        logger.error(f"  ✗ LLM 服务初始化失败: {e}")

    # ── 5. 初始化评估系统 ────────────────────────────────────
    try:
        from evaluation.trulens_eval import get_evaluator
        evaluator = get_evaluator()
        await evaluator.initialize()
        logger.info("  ✓ TruLens 评估系统就绪")
    except Exception as e:
        logger.warning(f"  ⚠ TruLens 初始化失败（降级运行）: {e}")

    # ── 6. Neo4j Schema 校验（规则1） ────────────────────────
    try:
        from services.db.neo4j_pool import get_neo4j_pool
        neo4j_pool = get_neo4j_pool()
        schema_issues = await neo4j_pool.validate_schema()
        if schema_issues:
            for issue in schema_issues:
                logger.warning(f"[Schema 校验] {issue}")
        else:
            logger.info("  ✓ Neo4j Schema 校验通过")
    except Exception as e:
        logger.warning(f"  ⚠ Neo4j Schema 校验跳过: {e}")

    # ── 7. OpenTelemetry ─────────────────────────────────────
    try:
        from observability.otel_tracing import setup_opentelemetry
        setup_opentelemetry()
    except Exception as e:
        logger.warning(f"  ⚠ OpenTelemetry 初始化跳过: {e}")

    # ── 8. MinerU 服务 ──────────────────────────────────────
    try:
        from document.mineru_parser import get_mineru_service
        mineru = get_mineru_service()
        await mineru.initialize()
        logger.info("  ✓ MinerU 服务就绪")
    except Exception as e:
        logger.warning(f"  ⚠ MinerU 服务初始化跳过: {e}")

    logger.info(f"{'='*60}")
    logger.info(f"  系统启动完成！")
    logger.info(f"  API 文档: http://localhost:8000/docs")
    logger.info(f"  启用通道: {settings.get_enabled_channels()}")
    logger.info(f"{'='*60}")

    yield

    # ── 关闭 ─────────────────────────────────────────────────
    logger.info("正在关闭系统...")

    try:
        from services.db.neo4j_pool import get_neo4j_pool
        await get_neo4j_pool().close()
    except Exception:
        pass

    try:
        from services.db.milvus_pool import get_milvus_pool
        await get_milvus_pool().close()
    except Exception:
        pass

    try:
        from services.db.postgres_pool import get_postgres_pool
        await get_postgres_pool().close()
    except Exception:
        pass

    try:
        from document.mineru_parser import get_mineru_service
        await get_mineru_service().close()
    except Exception:
        pass

    logger.info("系统已关闭")


# ── 创建 FastAPI 应用 ────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="医典 RAG 多维知识库系统 - DocRAG / GraphRAG / SQLRAG 三通道工程化检索方案",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ─────────────────────────────────────────────────
from api.routes.query import router as query_router
from api.routes.documents import router as documents_router
from api.routes.health import router as health_router
from api.routes.evaluation import router as evaluation_router

app.include_router(query_router)
app.include_router(documents_router)
app.include_router(health_router)
app.include_router(evaluation_router)

# OpenTelemetry 追踪
try:
    from observability.otel_tracing import instrument_fastapi
    instrument_fastapi(app)
except Exception:
    pass


# ── 根路由 ───────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "channels": settings.get_enabled_channels(),
    }


# ── 启动入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
