"""
Embedding 模型加载器（规则10: 本地优先，失败降级 API，再失败随机向量占位）
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding 模型封装

    加载优先级:
    1. 本地模型（bge-large-zh-v1.5）
    2. API fallback
    3. 随机向量占位 + 告警日志
    """

    def __init__(self):
        self._model = None
        self._api_client = None
        self._mode: str = "uninitialized"  # "local", "api", "random"
        self._dimension: int = get_settings().embedding_dimension
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """初始化 Embedding 模型（线程安全，规则18）"""
        async with self._lock:
            if self._mode != "uninitialized":
                return

            settings = get_settings()

            # 1. 尝试加载本地模型
            try:
                from langchain_community.embeddings import HuggingFaceBgeEmbeddings

                self._model = HuggingFaceBgeEmbeddings(
                    model_name=settings.embedding_model_path,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                # 测试一下是否可用
                test_vec = self._model.embed_query("测试")
                assert len(test_vec) == self._dimension, (
                    f"本地模型输出维度 {len(test_vec)} != 期望 {self._dimension}"
                )
                self._mode = "local"
                logger.info(f"Embedding 模型加载成功: 本地模式 ({settings.embedding_model_path})")
                return
            except Exception as e:
                logger.warning(f"本地 Embedding 模型加载失败: {e}，尝试 API fallback")

            # 2. 尝试 API fallback
            try:
                from openai import AsyncOpenAI

                self._api_client = AsyncOpenAI(
                    api_key=settings.embedding_api_key,
                    base_url=settings.embedding_api_base,
                )
                # 测试 API
                test_resp = await self._api_client.embeddings.create(
                    input="测试",
                    model=settings.embedding_model_name,
                )
                test_vec = test_resp.data[0].embedding
                assert len(test_vec) == self._dimension, (
                    f"API 模型输出维度 {len(test_vec)} != 期望 {self._dimension}"
                )
                self._mode = "api"
                logger.info(f"Embedding 模型加载成功: API 模式 ({settings.embedding_api_base})")
                return
            except Exception as e:
                logger.warning(f"API Embedding 模型加载失败: {e}，降级为随机向量占位")

            # 3. 随机向量占位 + 告警
            self._mode = "random"
            logger.error(
                "⚠️ [Embedding] 本地模型和 API 均加载失败，使用随机向量占位！"
                "检索结果将无意义，请检查模型配置！"
            )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> List[float]:
        """
        对单条文本生成 Embedding 向量

        Args:
            text: 输入文本

        Returns:
            向量列表，维度为 embedding_dimension
        """
        if self._mode == "uninitialized":
            await self.initialize()

        if self._mode == "local":
            try:
                vec = self._model.embed_query(text)
                return vec
            except Exception as e:
                logger.error(f"[Embedding] 本地模型编码失败: {e}")
                return self._random_vector()

        elif self._mode == "api":
            try:
                settings = get_settings()
                resp = await self._api_client.embeddings.create(
                    input=text,
                    model=settings.embedding_model_name,
                )
                return resp.data[0].embedding
            except Exception as e:
                logger.error(f"[Embedding] API 编码失败: {e}")
                return self._random_vector()

        else:
            return self._random_vector()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对多条文本批量生成 Embedding 向量

        Args:
            texts: 文本列表

        Returns:
            向量列表的列表
        """
        if self._mode == "uninitialized":
            await self.initialize()

        if self._mode == "local":
            try:
                vecs = self._model.embed_documents(texts)
                return vecs
            except Exception as e:
                logger.error(f"[Embedding] 本地模型批量编码失败: {e}")
                return [self._random_vector() for _ in texts]

        elif self._mode == "api":
            try:
                settings = get_settings()
                # API 通常有批量限制，分批处理
                batch_size = 32
                all_vecs = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    resp = await self._api_client.embeddings.create(
                        input=batch,
                        model=settings.embedding_model_name,
                    )
                    batch_vecs = [d.embedding for d in resp.data]
                    all_vecs.extend(batch_vecs)
                return all_vecs
            except Exception as e:
                logger.error(f"[Embedding] API 批量编码失败: {e}")
                return [self._random_vector() for _ in texts]

        else:
            return [self._random_vector() for _ in texts]

    def _random_vector(self) -> List[float]:
        """生成随机向量占位（规则10: 随机向量占位 + 日志告警）"""
        vec = np.random.randn(self._dimension).tolist()
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (np.array(vec) / norm).tolist()
        return vec


# ── 全局单例 ─────────────────────────────────────────────────
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """获取 Embedding 模型全局单例"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
