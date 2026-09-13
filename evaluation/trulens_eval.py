"""
TruLens 评估体系（规则16: 从第一天就埋好观测点）

RAG 三元评估指标:
- Context Relevance: 检索上下文与问题的相关性
- Groundedness: 答案是否基于检索到的上下文
- Answer Relevance: 答案与问题的相关性
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class TruLensEvaluator:
    """
    TruLens 评估器

    按通道分别评估（不只评估最终答案）
    开发阶段采样率 100%，线上采样 0.01%
    """

    def __init__(self):
        self._enabled = False
        self._sample_rate = 1.0
        self._feedback_functions = {}

    async def initialize(self) -> None:
        """初始化 TruLens（只读配置；TruLens 库延迟到首次评估时才加载，避免拖慢启动）"""
        settings = get_settings()
        self._enabled = settings.trulens_enabled
        self._sample_rate = settings.trulens_sample_rate

        if not self._enabled:
            logger.info("[TruLens] 评估已禁用")
            return

        logger.info(f"[TruLens] 评估已启用（懒加载），采样率: {self._sample_rate}")

    def _ensure_feedback_functions(self) -> None:
        """懒加载 TruLens 反馈函数（首次评估时调用，import 较慢约 30s）"""
        if self._feedback_functions or not self._enabled:
            return
        try:
            # 仅在首次评估时加载 TruLens（import 耗时长，不阻塞服务启动）
            from trulens.core import Feedback
            from trulens.providers.openai import OpenAI as TruOpenAI

            settings = get_settings()
            provider = TruOpenAI(
                model_engine=settings.llm_model_name,
            )

            self._feedback_functions = {
                "context_relevance": Feedback(
                    provider.context_relevance,
                    name="Context Relevance",
                ),
                "groundedness": Feedback(
                    provider.groundedness,
                    name="Groundedness",
                ),
                "answer_relevance": Feedback(
                    provider.answer_relevance,
                    name="Answer Relevance",
                ),
            }

            logger.info(f"[TruLens] 反馈函数懒加载完成")
        except Exception as e:
            logger.warning(f"[TruLens] 反馈函数加载失败，评估功能降级: {e}")
            self._enabled = False

    def should_sample(self) -> bool:
        """是否应该采样（基于采样率配置）"""
        if not self._enabled:
            return False
        return random.random() < self._sample_rate

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估 RAG 三元指标

        Args:
            query: 用户查询
            answer: LLM 生成的答案
            contexts: 检索到的上下文列表
            channel: 通道名称（按通道分别评估）

        Returns:
            评估结果 dict
        """
        if not self._enabled or not self.should_sample():
            return {"sampled": False}

        # 懒加载反馈函数（首次评估时 import TruLens）
        self._ensure_feedback_functions()
        if not self._feedback_functions:
            return {"sampled": False, "error": "TruLens 反馈函数不可用"}

        start_time = time.time()
        results = {
            "query": query,
            "channel": channel,
            "sampled": True,
        }

        try:
            context_text = "\n\n".join(contexts) if contexts else ""

            # Context Relevance
            if "context_relevance" in self._feedback_functions:
                try:
                    cr_score = self._feedback_functions["context_relevance"](
                        question=query,
                        context=context_text,
                    )
                    results["context_relevance"] = float(cr_score)
                except Exception as e:
                    logger.warning(f"[TruLens] Context Relevance 评估失败: {e}")
                    results["context_relevance"] = None

            # Groundedness
            if "groundedness" in self._feedback_functions:
                try:
                    gr_score = self._feedback_functions["groundedness"](
                        answer=answer,
                        context=context_text,
                    )
                    results["groundedness"] = float(gr_score)
                except Exception as e:
                    logger.warning(f"[TruLens] Groundedness 评估失败: {e}")
                    results["groundedness"] = None

            # Answer Relevance
            if "answer_relevance" in self._feedback_functions:
                try:
                    ar_score = self._feedback_functions["answer_relevance"](
                        question=query,
                        answer=answer,
                    )
                    results["answer_relevance"] = float(ar_score)
                except Exception as e:
                    logger.warning(f"[TruLens] Answer Relevance 评估失败: {e}")
                    results["answer_relevance"] = None

            elapsed_ms = (time.time() - start_time) * 1000
            results["elapsed_ms"] = round(elapsed_ms, 2)

            logger.info(
                f"[TruLens] 评估完成 | 通道: {channel} | "
                f"CR: {results.get('context_relevance')} | "
                f"GR: {results.get('groundedness')} | "
                f"AR: {results.get('answer_relevance')}"
            )

        except Exception as e:
            logger.error(f"[TruLens] 评估流程异常: {e}")
            results["error"] = str(e)

        return results

    async def evaluate_channel_comparison(
        self,
        query: str,
        channel_results: Dict[str, Dict[str, Any]],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对比实验: 不同通道/策略的评估对比

        规则16: chunk 策略对比实验控制变量

        Args:
            query: 查询
            channel_results: {通道名: {"answer": ..., "contexts": [...]}}
            ground_truth: 标准答案（可选）

        Returns:
            各通道评估结果对比
        """
        comparison = {}

        for channel_name, result in channel_results.items():
            answer = result.get("answer", "")
            contexts = result.get("contexts", [])

            if answer and contexts:
                eval_result = await self.evaluate(
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    channel=channel_name,
                )
                comparison[channel_name] = eval_result

        return comparison


# ── 全局单例 ─────────────────────────────────────────────────
_evaluator: Optional[TruLensEvaluator] = None


def get_evaluator() -> TruLensEvaluator:
    """获取评估器全局单例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = TruLensEvaluator()
    return _evaluator
