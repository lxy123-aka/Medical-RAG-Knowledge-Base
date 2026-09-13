"""
OpenTelemetry 可观测性配置
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


def setup_opentelemetry() -> None:
    """
    配置 OpenTelemetry 追踪

    追踪范围:
    - API 请求
    - 各通道检索
    - LLM 调用
    - 数据库操作
    """
    settings = get_settings()

    if not settings.otel_enabled:
        logger.info("[OTEL] OpenTelemetry 已禁用")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        # 配置 Resource
        resource = Resource.create({
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
        })

        # 配置 TracerProvider
        provider = TracerProvider(resource=resource)

        # 配置 OTLP Exporter
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=settings.otel_exporter_otlp_insecure,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # 设置全局 TracerProvider
        trace.set_tracer_provider(provider)

        logger.info(
            f"[OTEL] OpenTelemetry 初始化成功 | "
            f"服务: {settings.otel_service_name} | "
            f"端点: {settings.otel_exporter_otlp_endpoint}"
        )

    except ImportError as e:
        logger.warning(f"[OTEL] OpenTelemetry 依赖未安装: {e}")
    except Exception as e:
        logger.error(f"[OTEL] OpenTelemetry 初始化失败: {e}")


def instrument_fastapi(app) -> None:
    """为 FastAPI 应用添加 OpenTelemetry 追踪"""
    settings = get_settings()
    if not settings.otel_enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[OTEL] FastAPI 已添加追踪")
    except Exception as e:
        logger.warning(f"[OTEL] FastAPI 追踪添加失败: {e}")


def get_tracer(name: str):
    """获取 Tracer 实例"""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return None
