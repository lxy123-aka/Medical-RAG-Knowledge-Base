"""
LLM 统一服务层 - Qwen 系列（OpenAI 兼容接口）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from config.settings import get_settings
from core.safe_json import safe_json_parse, extract_json_from_markdown

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM 统一封装

    使用 OpenAI 兼容接口调用 Qwen 系列模型
    提供 chat、structured_output 等统一接口
    """

    def __init__(self):
        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化 LLM 客户端"""
        if self._initialized:
            return

        settings = get_settings()
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_base,
            )
            self._initialized = True
            logger.info(f"LLM 服务初始化成功: {settings.llm_model_name} @ {settings.llm_api_base}")
        except Exception as e:
            logger.error(f"LLM 服务初始化失败: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        通用对话接口

        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            model: 模型名称

        Returns:
            模型回复文本
        """
        if not self._initialized:
            await self.initialize()

        settings = get_settings()
        try:
            response = await self._client.chat.completions.create(
                model=model or settings.llm_model_name,
                messages=messages,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning("[LLM] 模型返回空内容")
                return ""
            return content.strip()
        except Exception as e:
            logger.error(f"[LLM] 对话请求失败: {e}")
            raise

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Optional[Union[Dict, List]]:
        """
        对话并解析 JSON 输出（规则6/7: 统一 JSON 解析）

        Args:
            messages: 消息列表
            temperature: 温度

        Returns:
            解析后的 dict/list，或 None
        """
        raw = await self.chat(messages, temperature=temperature)
        if not raw:
            return None
        return extract_json_from_markdown(raw)

    async def generate_hypothetical_docs(
        self,
        query: str,
        num_docs: int = 3,
    ) -> List[str]:
        """
        HyDE: 生成假设文档（规则3: 动态假设文档增强检索）

        Args:
            query: 用户查询
            num_docs: 生成假设文档数量

        Returns:
            假设文档列表
        """
        prompt = f"""你是一个医学知识专家。基于以下用户查询，生成 {num_docs} 个可能的医学文档片段作为假设性答案。
这些文档片段应该模拟真实医学教材、药典或病历中可能出现的内容。

用户查询: {query}

请生成 {num_docs} 个不同角度的假设文档片段，每个片段 100-200 字。
只输出 JSON 数组格式，不要任何解释:

["文档片段1", "文档片段2", "文档片段3"]"""

        messages = [
            {"role": "system", "content": "你是医学知识专家，只输出 JSON 数组格式。"},
            {"role": "user", "content": prompt},
        ]

        result = await self.chat_json(messages, temperature=0.8)
        if isinstance(result, list):
            docs = [str(d) for d in result if d]
            if docs:
                logger.info(f"[HyDE] 生成 {len(docs)} 个假设文档")
                return docs

        logger.warning("[HyDE] 假设文档生成失败，返回空列表")
        return []

    async def rewrite_query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        查询重写（规则3: 口语→医学术语标准化 + 子查询拆分 + 意图识别）

        基于 Prompt + All-Shot 方式，让 LLM 一次性完成:
        1. 口语→医学术语标准化
        2. 复杂问题子查询拆分
        3. 意图识别路由

        Returns:
            {"standardized_query": "...", "sub_queries": [...], "intent": "...", "target_channels": [...]}
        """
        prompt = f"""你是一个医学查询分析专家。对用户查询进行以下处理:

1. **术语标准化**: 将口语化表达转换为标准医学术语
2. **子查询拆分**: 将复杂问题拆分为可独立检索的子查询
3. **意图识别**: 识别查询意图类型
4. **通道路由**: 判断应该使用哪些检索通道

用户查询: "{query}"

意图类型选项:
- disease_query: 疾病查询
- symptom_analysis: 症状分析
- drug_recommendation: 药物推荐
- exam_interpretation: 检查解读
- medical_record: 病历查询
- general_qa: 通用问答
- multi_intent: 多意图

通道选项: docrag, graphrag, sqlrag

只输出 JSON 对象，不要任何解释:
{{"standardized_query": "标准化后的查询", "sub_queries": ["子查询1", "子查询2"], "intent": "意图类型", "target_channels": ["docrag", "graphrag"]}}"""

        messages = [
            {"role": "system", "content": "你是医学查询分析专家，只输出 JSON 对象。"},
            {"role": "user", "content": prompt},
        ]

        result = await self.chat_json(messages, temperature=0.3)
        if isinstance(result, dict):
            logger.info(f"[QueryRewrite] 意图: {result.get('intent')}, 通道: {result.get('target_channels')}")
            return result

        # 降级：返回原始查询
        logger.warning("[QueryRewrite] 解析失败，返回原始查询")
        return {
            "standardized_query": query,
            "sub_queries": [query],
            "intent": "general_qa",
            "target_channels": ["docrag"],
        }

    async def generate_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
    ) -> str:
        """
        基于检索上下文生成最终答案

        Args:
            query: 用户查询
            contexts: 检索到的上下文列表

        Returns:
            LLM 生成的答案
        """
        # 构建上下文文本
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("source", "未知")
            content = ctx.get("content", "")
            score = ctx.get("score", 0)
            context_parts.append(
                f"[{i}] 来源: {source} | 相关度: {score:.2f}\n{content}"
            )

        context_text = "\n\n".join(context_parts)

        prompt = f"""你是医学知识问答专家。基于以下检索到的参考资料，回答用户的问题。

要求:
1. 回答必须基于提供的参考资料，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 回答要专业、准确、条理清晰
4. 使用中文回答

用户问题: {query}

参考资料:
{context_text}

请给出专业回答:"""

        messages = [
            {"role": "system", "content": "你是医学知识问答专家，基于参考资料给出专业回答。"},
            {"role": "user", "content": prompt},
        ]

        answer = await self.chat(messages, temperature=0.5)
        return answer

    async def nl2cypher(
        self,
        query: str,
        schema_summary: str,
        error_feedback: Optional[str] = None,
    ) -> str:
        """
        NL2Cypher: 自然语言转 Cypher 查询（规则15）

        Args:
            query: 自然语言查询
            schema_summary: Neo4j Schema 摘要
            error_feedback: 上一次执行的错误反馈（用于重试）

        Returns:
            生成的 Cypher 语句
        """
        error_section = ""
        if error_feedback:
            error_section = f"""
上一次生成的 Cypher 执行失败，错误信息:
{error_feedback}

请根据错误信息修正 Cypher 语句。注意:
- 属性名必须与 Schema 完全一致
- 关系类型必须使用 Schema 中定义的类型
- 只修正错误部分，保持正确的查询逻辑
"""

        prompt = f"""你是一个 Neo4j Cypher 查询专家。根据用户的自然语言问题和图谱 Schema，生成正确的 Cypher 查询语句。

{schema_summary}
{error_section}

用户问题: {query}

要求:
1. 只输出 Cypher 语句，不要任何解释
2. 属性名必须与上述 Schema 完全一致
3. 关系类型必须使用上述 Schema 中定义的类型
4. 查询结果要有意义（使用 RETURN 返回关键信息）
5. 使用 MATCH 模式匹配，避免笛卡尔积

Cypher:"""

        messages = [
            {"role": "system", "content": "你是 Cypher 查询专家，只输出 Cypher 语句。"},
            {"role": "user", "content": prompt},
        ]

        cypher = await self.chat(messages, temperature=0.1)
        return cypher

    async def nl2sql(
        self,
        query: str,
        table_schemas: str,
    ) -> str:
        """
        NL2SQL: 自然语言转 SQL 查询

        Args:
            query: 自然语言查询
            table_schemas: 表结构描述

        Returns:
            生成的 SQL 语句
        """
        prompt = f"""你是一个 PostgreSQL SQL 查询专家。根据用户的自然语言问题和数据库表结构，生成正确的 SQL 查询语句。

数据库表结构:
{table_schemas}

用户问题: {query}

要求:
1. 只输出 SQL 语句，不要任何解释
2. 使用标准 PostgreSQL 语法
3. 查询要高效，避免全表扫描
4. 使用 LIMIT 限制返回行数（默认 LIMIT 20）

SQL:"""

        messages = [
            {"role": "system", "content": "你是 SQL 查询专家，只输出 SQL 语句。"},
            {"role": "user", "content": prompt},
        ]

        sql = await self.chat(messages, temperature=0.1)
        return sql


# ── 全局单例 ─────────────────────────────────────────────────
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取 LLM 服务全局单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
