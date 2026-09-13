"""
融合层 - 三通道并行检索 + 降级 + 结果融合（规则8/9）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from models.schemas import (
    ChannelResult,
    ChannelStatus,
    ChannelType,
    FusedResult,
    QueryRequest,
    RewriteResult,
    IntentType,
)
from rag.docrag.channel import get_docrag_channel
from rag.graphrag.channel import get_graphrag_channel
from rag.sqlrag.channel import get_sqlrag_channel
from services.llm.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    融合引擎

    职责:
    1. Query Rewrite（术语标准化 + 子查询拆分 + 意图识别路由）
    2. 三通道 asyncio.gather 并行检索
    3. 空结果降级逻辑（规则8: 每通道检查异常，单通道失败不影响其他）
    4. 结果融合 + LLM 生成最终答案
    5. 每通道耗时监控日志
    """

    async def process(self, request: QueryRequest) -> FusedResult:
        """
        处理检索请求

        流程:
        1. Query Rewrite
        2. 并行检索三通道
        3. 融合结果
        4. LLM 生成答案
        """
        start_time = time.time()
        settings = get_settings()

        # ── 1. Query Rewrite ──────────────────────────────────
        rewrite = None
        if request.enable_rewrite and settings.query_rewrite_enabled:
            rewrite = await self._rewrite_query(request.query)

        # ── 2. 确定要执行的通道 ──────────────────────────────
        channels_to_run = self._determine_channels(request, rewrite)
        logger.info(f"[Fusion] 执行通道: {channels_to_run}")

        # ── 3. 并行检索（asyncio.gather + return_exceptions） ──
        channel_results = await self._parallel_search(
            query=request.query,
            rewrite=rewrite,
            channels=channels_to_run,
            top_k=request.top_k,
            permission_level=request.permission_level,
            department=request.department,
            enable_hyde=request.enable_hyde,
            enable_reranker=request.enable_reranker,
        )

        # ── 4. 融合结果 ──────────────────────────────────────
        fused_results = self._fuse_results(channel_results)

        # ── 5. LLM 生成最终答案 ──────────────────────────────
        llm_answer = None
        if fused_results:
            llm_service = get_llm_service()
            try:
                llm_answer = await llm_service.generate_answer(
                    query=request.query,
                    contexts=fused_results,
                )
            except Exception as e:
                logger.error(f"[Fusion] LLM 生成答案失败: {e}")
                llm_answer = "抱歉，生成回答时出现错误。以下是检索到的相关信息。"

        elapsed_ms = (time.time() - start_time) * 1000

        return FusedResult(
            query=request.query,
            rewrite=rewrite,
            channel_results=channel_results,
            fused_results=fused_results,
            llm_answer=llm_answer,
            total_elapsed_ms=round(elapsed_ms, 2),
        )

    async def _rewrite_query(self, query: str) -> RewriteResult:
        """查询重写"""
        llm_service = get_llm_service()
        try:
            result = await llm_service.rewrite_query(query)
            return RewriteResult(
                original_query=query,
                standardized_query=result.get("standardized_query", query),
                sub_queries=result.get("sub_queries", [query]),
                intent=IntentType(result.get("intent", "general_qa")),
                target_channels=[
                    ChannelType(ch)
                    for ch in result.get("target_channels", [])
                    if ch in [c.value for c in ChannelType]
                ],
            )
        except Exception as e:
            logger.warning(f"[Fusion] 查询重写失败: {e}")
            return RewriteResult(
                original_query=query,
                standardized_query=query,
                sub_queries=[query],
            )

    def _determine_channels(
        self,
        request: QueryRequest,
        rewrite: Optional[RewriteResult],
    ) -> List[ChannelType]:
        """
        确定要执行的通道

        优先级:
        1. 请求中明确指定的通道
        2. 查询重写推荐的通道
        3. 配置中启用的所有通道
        """
        settings = get_settings()

        # 1. 请求明确指定
        if request.channels:
            return request.channels

        # 2. 查询重写推荐
        if rewrite and rewrite.target_channels:
            return rewrite.target_channels

        # 3. 配置中启用的通道
        channel_map = {
            "docrag": ChannelType.DOCRAG,
            "graphrag": ChannelType.GRAPHRAG,
            "sqlrag": ChannelType.SQLRAG,
        }
        return [
            channel_map[ch]
            for ch in settings.get_enabled_channels()
            if ch in channel_map
        ]

    async def _parallel_search(
        self,
        query: str,
        rewrite: Optional[RewriteResult],
        channels: List[ChannelType],
        top_k: int,
        permission_level: int,
        department: Optional[str],
        enable_hyde: bool,
        enable_reranker: bool,
    ) -> List[ChannelResult]:
        """
        三通道并行检索（规则8: asyncio.gather + return_exceptions）

        每通道独立执行，单通道失败不影响其他通道
        每通道加耗时监控日志
        """
        tasks = []

        if ChannelType.DOCRAG in channels:
            tasks.append(self._search_docrag(
                query, rewrite, top_k, permission_level,
                department, enable_hyde, enable_reranker,
            ))

        if ChannelType.GRAPHRAG in channels:
            tasks.append(self._search_graphrag(
                query, rewrite, top_k, permission_level,
            ))

        if ChannelType.SQLRAG in channels:
            tasks.append(self._search_sqlrag(
                query, rewrite, top_k, permission_level,
            ))

        if not tasks:
            logger.warning("[Fusion] 没有可执行的通道")
            return []

        # 并行执行（return_exceptions=True 防止单通道异常导致整体失败）
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 逐通道检查异常（规则8）
        channel_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"[Fusion] 通道执行异常: {result}",
                    exc_info=result,
                )
                # 创建失败结果
                channel_type = channels[i] if i < len(channels) else ChannelType.DOCRAG
                channel_results.append(ChannelResult(
                    channel=channel_type,
                    status=ChannelStatus.FAILED,
                    results=[],
                    elapsed_ms=0.0,
                    error_message=str(result),
                ))
            elif isinstance(result, ChannelResult):
                channel_results.append(result)
            else:
                logger.error(f"[Fusion] 通道返回类型异常: {type(result)}")

        return channel_results

    async def _search_docrag(self, query, rewrite, top_k, permission_level,
                              department, enable_hyde, enable_reranker) -> ChannelResult:
        """DocRAG 通道检索"""
        channel = get_docrag_channel()
        return await channel.search(
            query=query,
            rewrite=rewrite,
            top_k=top_k,
            permission_level=permission_level,
            department=department,
            enable_hyde=enable_hyde,
            enable_reranker=enable_reranker,
        )

    async def _search_graphrag(self, query, rewrite, top_k, permission_level) -> ChannelResult:
        """GraphRAG 通道检索"""
        channel = get_graphrag_channel()
        return await channel.search(
            query=query,
            rewrite=rewrite,
            top_k=top_k,
            permission_level=permission_level,
        )

    async def _search_sqlrag(self, query, rewrite, top_k, permission_level) -> ChannelResult:
        """SQLRAG 通道检索"""
        channel = get_sqlrag_channel()
        return await channel.search(
            query=query,
            rewrite=rewrite,
            top_k=top_k,
            permission_level=permission_level,
        )

    def _fuse_results(
        self,
        channel_results: List[ChannelResult],
    ) -> List[Dict[str, Any]]:
        """
        融合多通道结果

        策略:
        1. 收集所有通道的成功结果
        2. 按来源和分数排序
        3. 去重（基于内容相似度）
        4. 规则8: 空结果降级（某通道为空/失败不影响其他）
        """
        all_results = []

        for cr in channel_results:
            # 规则8: 检查通道状态
            if cr.status == ChannelStatus.FAILED:
                logger.warning(
                    f"[Fusion] 通道 {cr.channel.value} 失败: {cr.error_message}"
                )
                continue

            if cr.status == ChannelStatus.EMPTY:
                logger.info(f"[Fusion] 通道 {cr.channel.value} 返回空结果")
                continue

            if cr.status == ChannelStatus.SUCCESS and cr.results:
                # 记录耗时
                logger.info(
                    f"[Fusion] 通道 {cr.channel.value} | "
                    f"结果数: {len(cr.results)} | 耗时: {cr.elapsed_ms:.1f}ms"
                )
                all_results.extend(cr.results)

        # 按分数降序排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 简单去重（基于内容前 100 字符）
        seen_contents = set()
        unique_results = []
        for result in all_results:
            content_key = result.get("content", "")[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_results.append(result)

        return unique_results


# ── 全局单例 ─────────────────────────────────────────────────
_fusion_engine: Optional[FusionEngine] = None


def get_fusion_engine() -> FusionEngine:
    """获取融合引擎全局单例"""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = FusionEngine()
    return _fusion_engine
