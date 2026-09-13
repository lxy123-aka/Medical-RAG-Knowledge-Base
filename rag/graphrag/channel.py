"""
GraphRAG 通道 - Neo4j 知识图谱检索（含 NL2Cypher + 错误反馈重试）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.neo4j_schema import build_schema_summary
from core.safe_json import extract_cypher_from_text
from models.schemas import (
    ChannelResult,
    ChannelStatus,
    ChannelType,
    NL2CypherResult,
    RewriteResult,
)
from services.db.neo4j_pool import get_neo4j_pool
from services.llm.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class GraphRAGChannel:
    """
    GraphRAG 知识图谱检索通道

    流程:
    1. NL2Cypher: 自然语言 → Cypher 查询
    2. 执行 Cypher 查询
    3. 失败时错误反馈重试（规则15: 最多 2-3 次）
    4. 结果格式化
    """

    def __init__(self):
        self._channel_type = ChannelType.GRAPHRAG

    async def search(
        self,
        query: str,
        rewrite: Optional[RewriteResult] = None,
        top_k: Optional[int] = None,
        permission_level: int = 0,
        **kwargs,
    ) -> ChannelResult:
        """
        执行 GraphRAG 检索

        Args:
            query: 查询文本
            rewrite: 查询重写结果
            top_k: 返回数量
            permission_level: 权限等级

        Returns:
            ChannelResult
        """
        start_time = time.time()
        settings = get_settings()
        top_k = top_k or settings.graphrag_top_k

        try:
            # 1. NL2Cypher 转换（含重试）
            nl2cypher_result = await self._nl2cypher_with_retry(query)

            if not nl2cypher_result.success:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.warning(
                    f"[GraphRAG] NL2Cypher 最终失败 | "
                    f"重试次数: {nl2cypher_result.retries}"
                )
                return ChannelResult(
                    channel=self._channel_type,
                    status=ChannelStatus.FAILED,
                    results=[],
                    elapsed_ms=round(elapsed_ms, 2),
                    error_message=f"NL2Cypher 失败: {'; '.join(nl2cypher_result.error_messages)}",
                )

            # 2. 格式化结果
            formatted_results = self._format_results(
                nl2cypher_result.execution_result or [],
                nl2cypher_result.generated_cypher,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            status = ChannelStatus.SUCCESS if formatted_results else ChannelStatus.EMPTY

            logger.info(
                f"[GraphRAG] 检索完成 | 结果数: {len(formatted_results)} | "
                f"重试次数: {nl2cypher_result.retries} | 耗时: {elapsed_ms:.1f}ms"
            )

            return ChannelResult(
                channel=self._channel_type,
                status=status,
                results=formatted_results[:top_k],
                elapsed_ms=round(elapsed_ms, 2),
                metadata={
                    "cypher": nl2cypher_result.generated_cypher,
                    "retries": nl2cypher_result.retries,
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"[GraphRAG] 检索失败: {e}", exc_info=True)
            return ChannelResult(
                channel=self._channel_type,
                status=ChannelStatus.FAILED,
                results=[],
                elapsed_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    async def _nl2cypher_with_retry(
        self,
        query: str,
    ) -> NL2CypherResult:
        """
        NL2Cypher 带错误反馈重试（规则15）

        重试策略:
        1. 生成 Cypher
        2. 执行 Cypher
        3. 失败则将错误信息精简反馈给 LLM
        4. 重新生成 Cypher（最多重试 max_retries 次）
        """
        settings = get_settings()
        max_retries = settings.graphrag_nl2cypher_max_retries
        llm_service = get_llm_service()
        neo4j_pool = get_neo4j_pool()
        schema_summary = build_schema_summary()

        error_feedback = None
        error_messages = []

        for attempt in range(max_retries + 1):
            try:
                # 1. 生成 Cypher（如有错误反馈则传入）
                raw_cypher = await llm_service.nl2cypher(
                    query=query,
                    schema_summary=schema_summary,
                    error_feedback=error_feedback,
                )

                # 2. 提取 Cypher 语句
                cypher = extract_cypher_from_text(raw_cypher)
                if not cypher:
                    error_msg = "无法从 LLM 输出中提取 Cypher 语句"
                    logger.warning(f"[NL2Cypher] {error_msg} | 尝试 {attempt + 1}")
                    error_feedback = error_msg
                    error_messages.append(error_msg)
                    continue

                logger.info(f"[NL2Cypher] 生成 Cypher (尝试 {attempt + 1}): {cypher[:100]}")

                # 3. 执行 Cypher
                try:
                    results = await neo4j_pool.execute_read(cypher)

                    # 4. 执行成功
                    return NL2CypherResult(
                        original_query=query,
                        generated_cypher=cypher,
                        execution_result=results,
                        retries=attempt,
                        success=True,
                        error_messages=error_messages,
                    )

                except Exception as exec_err:
                    # 5. 执行失败，提取关键错误信息（规则15: 精简反馈）
                    error_msg = self._extract_cypher_error(str(exec_err))
                    error_feedback = error_msg
                    error_messages.append(f"尝试 {attempt + 1}: {error_msg}")
                    logger.warning(
                        f"[NL2Cypher] Cypher 执行失败 (尝试 {attempt + 1}): {error_msg}"
                    )

            except Exception as e:
                error_msg = f"NL2Cypher 流程异常: {str(e)}"
                error_messages.append(error_msg)
                logger.error(f"[NL2Cypher] {error_msg}", exc_info=True)

        # 所有重试均失败
        return NL2CypherResult(
            original_query=query,
            success=False,
            retries=max_retries,
            error_messages=error_messages,
        )

    def _extract_cypher_error(self, error_msg: str) -> str:
        """
        提取 Cypher 语法错误关键信息（规则15: 禁止塞整个 stack trace）

        只保留关键错误信息:
        - Invalid input
        - Unknown relationship type
        - Undefined property
        - Type mismatch
        等
        """
        # 截取前 300 字符，避免过长
        error_lines = error_msg.split('\n')
        key_errors = []

        keywords = [
            "Invalid input", "Unknown", "Undefined", "Type mismatch",
            "syntax error", "expected", "unexpected", "not found",
            "does not exist", "permission", "authorization",
        ]

        for line in error_lines:
            line = line.strip()
            if any(kw.lower() in line.lower() for kw in keywords):
                key_errors.append(line)

        if key_errors:
            return "\n".join(key_errors[:3])[:300]

        # 如果没有匹配到关键词，截取前 200 字符
        return error_msg[:200]

    def _format_results(
        self,
        records: List[Dict[str, Any]],
        cypher: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """格式化图谱查询结果"""
        formatted = []
        for record in records:
            # 尝试提取有用的信息
            content_parts = []
            entities = []

            for key, value in record.items():
                if isinstance(value, dict):
                    # 节点或关系对象
                    if "name" in value:
                        entities.append(value["name"])
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    content_parts.append(f"{key}: {value}")
                else:
                    content_parts.append(f"{key}: {value}")

            content = " | ".join(content_parts) if content_parts else str(record)

            formatted.append({
                "source": "graphrag",
                "content": content,
                "entities": entities,
                "score": 1.0,  # 图谱查询结果给固定高分
                "metadata": {
                    "cypher": cypher,
                    "record_keys": list(record.keys()),
                },
            })

        return formatted


# ── 全局单例 ─────────────────────────────────────────────────
_graphrag_channel: Optional[GraphRAGChannel] = None


def get_graphrag_channel() -> GraphRAGChannel:
    """获取 GraphRAG 通道全局单例"""
    global _graphrag_channel
    if _graphrag_channel is None:
        _graphrag_channel = GraphRAGChannel()
    return _graphrag_channel
