"""
API 路由 - 检索查询
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from fusion.fusion_engine import get_fusion_engine
from models.schemas import QueryRequest, QueryResponse, ChannelType
from evaluation.trulens_eval import get_evaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["检索查询"])


@router.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest) -> QueryResponse:
    """
    多通道检索接口

    支持三通道并行检索（DocRAG/GraphRAG/SQLRAG），结果融合后由 LLM 生成统一回答。
    也支持单独指定单通道检索。

    参数:
    - query: 用户查询文本
    - channels: 指定检索通道（可选，默认使用配置中启用的所有通道）
    - top_k: 返回结果数量
    - permission_level: 权限等级过滤
    - enable_hyde: 是否启用 HyDE
    - enable_rewrite: 是否启用查询重写
    - enable_reranker: 是否启用 Reranker
    """
    trace_id = str(uuid.uuid4())

    try:
        fusion_engine = get_fusion_engine()
        fused_result = await fusion_engine.process(request)

        # TruLens 评估（规则16: 从第一天就埋好观测点）
        evaluator = get_evaluator()
        if fused_result.llm_answer and fused_result.fused_results:
            contexts = [r.get("content", "") for r in fused_result.fused_results]
            eval_result = await evaluator.evaluate(
                query=request.query,
                answer=fused_result.llm_answer,
                contexts=contexts,
            )
        else:
            eval_result = {}

        # 构建响应
        channel_details = []
        for cr in fused_result.channel_results:
            channel_details.append({
                "channel": cr.channel.value,
                "status": cr.status.value,
                "result_count": len(cr.results),
                "elapsed_ms": cr.elapsed_ms,
                "error": cr.error_message,
                "metadata": cr.metadata,
            })

        return QueryResponse(
            query=request.query,
            answer=fused_result.llm_answer or "未找到相关信息",
            channels_used=[cr.channel.value for cr in fused_result.channel_results],
            channel_details=channel_details,
            results=fused_result.fused_results,
            elapsed_ms=fused_result.total_elapsed_ms,
            trace_id=trace_id,
        )

    except Exception as e:
        logger.error(f"[API] 检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/search/single")
async def search_single_channel(
    request: QueryRequest,
    channel: ChannelType,
) -> Dict[str, Any]:
    """
    单通道检索接口

    只执行指定通道的检索，不融合。用于调试和单独评估。
    """
    try:
        request.channels = [channel]
        fusion_engine = get_fusion_engine()
        fused_result = await fusion_engine.process(request)

        # 找到对应通道的结果
        channel_result = None
        for cr in fused_result.channel_results:
            if cr.channel == channel:
                channel_result = cr
                break

        if not channel_result:
            return {
                "channel": channel.value,
                "status": "not_found",
                "results": [],
            }

        return {
            "channel": channel.value,
            "status": channel_result.status.value,
            "results": channel_result.results,
            "elapsed_ms": channel_result.elapsed_ms,
            "error": channel_result.error_message,
            "metadata": channel_result.metadata,
        }

    except Exception as e:
        logger.error(f"[API] 单通道检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")
