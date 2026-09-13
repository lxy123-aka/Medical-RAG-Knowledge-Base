"""
API 路由 - 评估接口
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from evaluation.trulens_eval import get_evaluator
from models.schemas import EvaluationRequest, ChannelType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["评估"])


@router.post("/evaluate")
async def evaluate_rag(request: EvaluationRequest) -> Dict[str, Any]:
    """
    RAG 评估接口

    对指定查询进行 RAG 三元指标评估:
    - Context Relevance
    - Groundedness
    - Answer Relevance
    """
    try:
        evaluator = get_evaluator()

        # 如果提供了 contexts，直接评估
        if request.contexts:
            # 需要先生成答案
            from fusion.fusion_engine import get_fusion_engine
            from models.schemas import QueryRequest

            query_request = QueryRequest(
                query=request.query,
                channels=request.channels,
            )
            fusion_engine = get_fusion_engine()
            fused_result = await fusion_engine.process(query_request)

            answer = fused_result.llm_answer or ""
            contexts = [r.get("content", "") for r in fused_result.fused_results]

            eval_result = await evaluator.evaluate(
                query=request.query,
                answer=answer,
                contexts=contexts,
            )

            return {
                "status": "success",
                "query": request.query,
                "answer": answer,
                "evaluation": eval_result,
            }

        else:
            return {
                "status": "error",
                "message": "需要提供 contexts 参数",
            }

    except Exception as e:
        logger.error(f"[API] 评估失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@router.post("/compare")
async def compare_channels(
    query: str,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    通道对比评估

    分别执行各通道检索，对比评估结果
    """
    try:
        from fusion.fusion_engine import get_fusion_engine
        from models.schemas import QueryRequest

        evaluator = get_evaluator()
        channel_results = {}

        # 逐通道执行
        for channel_type in [ChannelType.DOCRAG, ChannelType.GRAPHRAG, ChannelType.SQLRAG]:
            query_request = QueryRequest(
                query=query,
                channels=[channel_type],
            )
            fusion_engine = get_fusion_engine()
            fused_result = await fusion_engine.process(query_request)

            # 找到对应通道结果
            for cr in fused_result.channel_results:
                if cr.channel == channel_type and cr.results:
                    contexts = [r.get("content", "") for r in cr.results]
                    channel_results[channel_type.value] = {
                        "answer": fused_result.llm_answer or "",
                        "contexts": contexts,
                        "result_count": len(cr.results),
                        "elapsed_ms": cr.elapsed_ms,
                    }

        # 对比评估
        comparison = await evaluator.evaluate_channel_comparison(
            query=query,
            channel_results=channel_results,
            ground_truth=ground_truth,
        )

        return {
            "status": "success",
            "query": query,
            "channel_results": channel_results,
            "comparison": comparison,
        }

    except Exception as e:
        logger.error(f"[API] 通道对比评估失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")
