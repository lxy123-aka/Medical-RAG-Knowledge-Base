"""
Pydantic Settings - 应用启动时一次性校验所有必填配置
规则13: .env 只读一次，生产用 Pydantic Settings 启动时一次性校验
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，启动时一次性校验，配置不合法直接拒绝启动"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 应用基础 ──────────────────────────────────────────────
    app_name: str = Field(default="医典RAG多维知识库系统")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ── LLM ──────────────────────────────────────────────────
    llm_api_key: str = Field(default="")
    llm_api_base: str = Field(default="http://localhost:8000/v1")
    llm_model_name: str = Field(default="qwen-plus")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)

    # ── Embedding ────────────────────────────────────────────
    embedding_model_path: str = Field(default="./models/bge-large-zh-v1.5")
    embedding_api_key: str = Field(default="")
    embedding_api_base: str = Field(default="http://localhost:8001/v1")
    embedding_model_name: str = Field(default="bge-large-zh-v1.5")
    embedding_dimension: int = Field(default=1024)

    # ── Neo4j ────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    neo4j_database: str = Field(default="neo4j")

    # ── Milvus ───────────────────────────────────────────────
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_user: str = Field(default="root")
    milvus_password: str = Field(default="")
    milvus_db_name: str = Field(default="default")

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="")
    postgres_db: str = Field(default="medical_rag")
    postgres_pool_size: int = Field(default=10, gt=0)

    # ── Redis（可选） ────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")
    redis_db: int = Field(default=0)

    # ── MinerU ───────────────────────────────────────────────
    mineru_api_base: str = Field(default="http://localhost:8002")
    mineru_api_key: str = Field(default="")

    # ── 通道开关 ─────────────────────────────────────────────
    channel_doc_rag_enabled: bool = Field(default=True)
    channel_graph_rag_enabled: bool = Field(default=True)
    channel_sql_rag_enabled: bool = Field(default=True)

    # ── DocRAG ───────────────────────────────────────────────
    docrag_min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    docrag_top_k: int = Field(default=5, gt=0)
    docrag_chunk_size: int = Field(default=512, gt=0)
    docrag_chunk_overlap: int = Field(default=50, ge=0)

    # ── GraphRAG ─────────────────────────────────────────────
    graphrag_top_k: int = Field(default=5, gt=0)
    graphrag_nl2cypher_max_retries: int = Field(default=3, ge=1, le=5)

    # ── SQLRAG ───────────────────────────────────────────────
    sqlrag_top_k: int = Field(default=10, gt=0)

    # ── HyDE ─────────────────────────────────────────────────
    hyde_enabled: bool = Field(default=True)
    hyde_num_hypothetical_docs: int = Field(default=3, gt=0)

    # ── Query Rewrite ────────────────────────────────────────
    query_rewrite_enabled: bool = Field(default=True)

    # ── Reranker ─────────────────────────────────────────────
    reranker_enabled: bool = Field(default=True)
    reranker_top_n: int = Field(default=5, gt=0)

    # ── TruLens ──────────────────────────────────────────────
    trulens_enabled: bool = Field(default=True)
    trulens_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # ── OpenTelemetry ────────────────────────────────────────
    otel_enabled: bool = Field(default=True)
    otel_service_name: str = Field(default="medical-rag-knowledge-base")
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_exporter_otlp_insecure: bool = Field(default=True)

    # ── 状态管理（规则11/12） ────────────────────────────────
    min_inquiry_rounds: int = Field(default=1, ge=1)
    convergence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def milvus_connection_config(self) -> dict:
        return {
            "host": self.milvus_host,
            "port": self.milvus_port,
            "user": self.milvus_user,
            "password": self.milvus_password,
            "db_name": self.milvus_db_name,
        }

    def get_enabled_channels(self) -> List[str]:
        channels = []
        if self.channel_doc_rag_enabled:
            channels.append("docrag")
        if self.channel_graph_rag_enabled:
            channels.append("graphrag")
        if self.channel_sql_rag_enabled:
            channels.append("sqlrag")
        return channels


# ── 全局单例 ─────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例（规则13: .env 只读一次）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def validate_startup_settings(settings: Settings) -> List[str]:
    """
    启动时校验所有必填配置（规则13）
    返回错误信息列表，空列表表示全部通过
    """
    errors: List[str] = []

    # LLM API Key 非空
    if not settings.llm_api_key.strip():
        errors.append("LLM_API_KEY 未配置，请检查 .env 文件")

    # Neo4j 密码非空
    if not settings.neo4j_password.strip():
        errors.append("NEO4J_PASSWORD 未配置")

    # Milvus 密码非空
    if not settings.milvus_password.strip():
        errors.append("MILVUS_PASSWORD 未配置")

    # PostgreSQL 密码非空
    if not settings.postgres_password.strip():
        errors.append("POSTGRES_PASSWORD 未配置")

    # Embedding 模型路径存在（本地优先）
    if settings.embedding_model_path:
        path = Path(settings.embedding_model_path)
        if not path.exists():
            errors.append(
                f"本地 Embedding 模型路径不存在: {settings.embedding_model_path}，"
                f"将尝试 API fallback"
            )

    return errors
