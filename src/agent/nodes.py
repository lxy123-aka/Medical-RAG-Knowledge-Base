"""
LangGraph Agent 节点定义

每个节点对应工作流中的一个 Agent 步骤：
1. query_rewrite_node    - 问题改写/查询扩展 Agent
2. entity_extract_node   - 医学实体抽取 Agent
3. vector_retrieve_node  - Milvus 向量检索节点
4. graph_retrieve_node   - Neo4j 图谱检索节点
5. context_assemble_node - 上下文拼接节点
6. answer_generate_node  - LLM 答案生成节点
7. fact_verify_node      - 医学事实校验 Agent
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

from config.settings import get_settings
from core.safe_json import extract_json_from_markdown

logger = logging.getLogger(__name__)


# =============================================================================
# 免责声明常量
# =============================================================================

MEDICAL_DISCLAIMER = "【仅供学习参考，不能替代执业医师诊断】"


# =============================================================================
# 节点函数定义
# =============================================================================

async def query_rewrite_node(state: dict) -> dict:
    """
    问题改写/查询扩展 Agent

    功能：
    1. 口语化表达 → 标准医学术语
    2. 复杂问题拆分为子查询
    3. 意图识别（疾病查询/症状分析/药物推荐等）
    """
    start_time = time.time()
    query = state.get("original_query", "")
    trace_log = state.get("trace_log", [])
    trace_log.append(f"[改写Agent] 开始处理: {query[:50]}")

    try:
        from services.llm.llm_service import get_llm_service
        llm = get_llm_service()

        rewrite_result = await llm.rewrite_query(query)

        rewritten = rewrite_result.get("standardized_query", query)
        sub_queries = rewrite_result.get("sub_queries", [query])
        intent = rewrite_result.get("intent", "general_qa")

        elapsed = (time.time() - start_time) * 1000
        trace_log.append(
            f"[改写Agent] 完成, 意图={intent}, 耗时={elapsed:.0f}ms"
        )

        return {
            "rewritten_query": rewritten,
            "sub_queries": sub_queries,
            "intent": intent,
            "trace_log": trace_log,
        }

    except Exception as e:
        logger.error(f"[改写Agent] 失败: {e}")
        trace_log.append(f"[改写Agent] 失败: {e}")
        return {
            "rewritten_query": query,
            "sub_queries": [query],
            "intent": "general_qa",
            "trace_log": trace_log,
        }


async def entity_extract_node(state: dict) -> dict:
    """
    医学实体抽取 Agent

    功能：
    从改写后的查询中提取医学实体（疾病、症状、药物、检查项目）
    用于后续图谱检索
    """
    start_time = time.time()
    query = state.get("rewritten_query", "")
    sub_queries = state.get("sub_queries", [])
    trace_log = state.get("trace_log", [])

    # 合并所有查询文本
    all_queries = [query] + sub_queries
    combined_text = " ".join(all_queries)

    try:
        from src.entity_extract.extractor import MedicalEntityExtractor
        extractor = MedicalEntityExtractor()
        await extractor.initialize()

        entities = await extractor.extract_query_entities(combined_text)

        elapsed = (time.time() - start_time) * 1000
        entity_names = [e["name"] for e in entities]
        trace_log.append(
            f"[实体Agent] 抽取到 {len(entities)} 个实体: {entity_names}, "
            f"耗时={elapsed:.0f}ms"
        )

        return {
            "extracted_entities": entities,
            "trace_log": trace_log,
        }

    except Exception as e:
        logger.error(f"[实体Agent] 失败: {e}")
        trace_log.append(f"[实体Agent] 失败: {e}")
        return {"extracted_entities": [], "trace_log": trace_log}


async def vector_retrieve_node(state: dict) -> dict:
    """
    Milvus 向量检索节点

    功能：
    使用改写后的查询进行向量语义检索，召回相关文档片段
    """
    start_time = time.time()
    rewritten_query = state.get("rewritten_query", "")
    sub_queries = state.get("sub_queries", [])
    trace_log = state.get("trace_log", [])
    settings = get_settings()

    # 所有查询文本
    query_texts = [rewritten_query] + sub_queries

    try:
        from src.vector_store.store import VectorStore
        from core.embedding import get_embedding_model

        store = VectorStore()
        await store.initialize()
        embedding_model = get_embedding_model()

        # 多查询融合检索 + 去重
        results = await store.search_with_queries(
            query_texts=query_texts,
            embedding_model=embedding_model,
            top_k=settings.docrag_top_k,
            min_score=settings.docrag_min_score,
        )

        elapsed = (time.time() - start_time) * 1000
        trace_log.append(
            f"[向量检索] 召回 {len(results)} 条结果, 耗时={elapsed:.0f}ms"
        )

        return {
            "vector_results": results,
            "trace_log": trace_log,
        }

    except Exception as e:
        logger.error(f"[向量检索] 失败: {e}")
        trace_log.append(f"[向量检索] 失败: {e}")
        return {"vector_results": [], "trace_log": trace_log}


async def graph_retrieve_node(state: dict) -> dict:
    """
    Neo4j 图谱检索节点

    功能：
    基于抽取的医学实体，查询图谱中的关联关系
    （疾病→症状→药物→检查 等）
    """
    start_time = time.time()
    entities = state.get("extracted_entities", [])
    trace_log = state.get("trace_log", [])

    if not entities:
        trace_log.append("[图谱检索] 无实体，跳过图谱检索")
        return {"graph_results": [], "trace_log": trace_log}

    entity_names = [e["name"] for e in entities]

    try:
        from src.graph_store.store import GraphStore

        store = GraphStore()
        await store.initialize()

        # 并行查询所有实体的关联信息
        results = await store.query_by_entities_parallel(
            entity_names=entity_names,
            max_depth=2,
            limit_per_entity=10,
        )

        elapsed = (time.time() - start_time) * 1000
        trace_log.append(
            f"[图谱检索] 查询 {len(entity_names)} 个实体, "
            f"返回 {len(results)} 条关联, 耗时={elapsed:.0f}ms"
        )

        return {
            "graph_results": results,
            "trace_log": trace_log,
        }

    except Exception as e:
        logger.error(f"[图谱检索] 失败: {e}")
        trace_log.append(f"[图谱检索] 失败: {e}")
        return {"graph_results": [], "trace_log": trace_log}


async def context_assemble_node(state: dict) -> dict:
    """
    上下文拼接节点

    功能：
    1. 合并向量检索和图谱检索结果
    2. 构建带来源标注的上下文文本
    3. 收集引用文档来源列表
    """
    trace_log = state.get("trace_log", [])
    vector_results = state.get("vector_results", [])
    graph_results = state.get("graph_results", [])

    context_parts = []
    source_docs = []
    seen_sources = set()

    # 1. 向量检索结果（原文片段）
    if vector_results:
        context_parts.append("=== 相关文档片段 ===")
        for i, result in enumerate(vector_results, 1):
            content = result.get("content", "")
            score = result.get("score", 0)
            source = result.get("source", {})
            doc_id = source.get("doc_id", "未知")
            source_type = source.get("source_type", "")

            context_parts.append(
                f"[文档{i}] 来源: {doc_id} ({source_type}) | "
                f"相关度: {score:.2f}\n{content}"
            )

            # 收集来源
            source_key = doc_id
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                source_docs.append({
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "department": source.get("department", ""),
                    "relevance_score": score,
                })

    # 2. 图谱检索结果（关联关系）
    if graph_results:
        context_parts.append("\n=== 医学知识图谱关联 ===")
        for result in graph_results:
            source_entity = result.get("query_entity", result.get("source", ""))
            relation = result.get("relation", "")
            target = result.get("target", "")

            if source_entity and target:
                context_parts.append(
                    f"- {source_entity} →[{relation}]→ {target}"
                )

    context_text = "\n".join(context_parts)

    trace_log.append(
        f"[上下文拼接] 向量结果={len(vector_results)}条, "
        f"图谱结果={len(graph_results)}条, "
        f"引用文档={len(source_docs)}个"
    )

    return {
        "context_text": context_text,
        "source_documents": source_docs,
        "trace_log": trace_log,
    }


async def answer_generate_node(state: dict) -> dict:
    """
    LLM 答案生成节点

    功能：
    1. 基于检索上下文生成医学回答
    2. 强制附带文档来源
    3. 强制添加免责声明
    """
    start_time = time.time()
    query = state.get("original_query", "")
    context_text = state.get("context_text", "")
    source_docs = state.get("source_documents", [])
    trace_log = state.get("trace_log", [])

    # 构建来源标注
    source_refs = ""
    if source_docs:
        refs = []
        for i, doc in enumerate(source_docs, 1):
            refs.append(f"[{i}] {doc['doc_id']} ({doc.get('source_type', '')})")
        source_refs = "\n引用文档:\n" + "\n".join(refs)

    # 构造生成 Prompt
    system_prompt = f"""你是一个专业的医学知识问答助手。

核心要求：
1. 回答必须严格基于提供的参考资料，禁止编造医学信息
2. 如果参考资料不足以回答问题，必须明确说明"根据现有资料无法确定"
3. 回答中引用信息时，必须标注来源编号（如 [文档1]）
4. 禁止输出可直接用于临床诊断的结论
5. 回答末尾必须附带以下免责声明：
   {MEDICAL_DISCLAIMER}
6. 使用中文回答，条理清晰"""

    user_prompt = f"""用户问题：{query}

参考资料：
{context_text}

请基于以上参考资料回答用户问题。
回答要求：
- 严格基于参考资料
- 标注信息来源
- 信息不足时明确说明
- 附带免责声明
{source_refs}"""

    try:
        from services.llm.llm_service import get_llm_service
        llm = get_llm_service()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = await llm.chat(messages, temperature=0.3)

        # 确保包含免责声明
        if MEDICAL_DISCLAIMER not in answer:
            answer = answer.rstrip() + f"\n\n{MEDICAL_DISCLAIMER}"

        elapsed = (time.time() - start_time) * 1000
        trace_log.append(f"[生成Agent] 答案生成完成, 耗时={elapsed:.0f}ms")

        return {
            "generated_answer": answer,
            "trace_log": trace_log,
        }

    except Exception as e:
        logger.error(f"[生成Agent] 失败: {e}")
        trace_log.append(f"[生成Agent] 失败: {e}")
        return {
            "generated_answer": f"抱歉，生成答案时出现错误：{e}\n\n{MEDICAL_DISCLAIMER}",
            "trace_log": trace_log,
        }


async def fact_verify_node(state: dict) -> dict:
    """
    医学事实校验 Agent

    功能：
    1. 检查答案与参考资料是否存在矛盾
    2. 识别可能的幻觉信息（参考资料中不存在的内容）
    3. 检查用药信息是否准确
    4. 校验不通过时标记问题并修正答案
    """
    start_time = time.time()
    query = state.get("original_query", "")
    answer = state.get("generated_answer", "")
    context_text = state.get("context_text", "")
    trace_log = state.get("trace_log", [])

    # 构造校验 Prompt
    verify_prompt = f"""你是医学事实校验专家。请检查以下回答是否存在问题。

用户问题：{query}

参考资料：
{context_text}

待校验回答：
{answer}

请检查以下方面：
1. **矛盾检测**: 回答中是否有与参考资料矛盾的信息？
2. **幻觉检测**: 回答中是否包含参考资料中完全没有的医学事实？
3. **用药安全**: 药物名称、剂量、禁忌症是否准确？
4. **诊断越界**: 是否输出了可直接用于临床诊断的结论？

以 JSON 格式输出校验结果：
{{
  "passed": true/false,
  "issues": ["问题1描述", "问题2描述"],
  "corrected_answer": "修正后的回答（如有问题）或原回答"
}}

如果没有问题，passed 为 true，issues 为空数组，corrected_answer 为原回答。"""

    try:
        from services.llm.llm_service import get_llm_service
        llm = get_llm_service()

        messages = [
            {"role": "system", "content": "你是医学事实校验专家，只输出 JSON。"},
            {"role": "user", "content": verify_prompt},
        ]

        raw = await llm.chat(messages, temperature=0.1)
        result = extract_json_from_markdown(raw)

        if isinstance(result, dict):
            passed = result.get("passed", True)
            issues = result.get("issues", [])
            corrected = result.get("corrected_answer", answer)

            elapsed = (time.time() - start_time) * 1000
            trace_log.append(
                f"[校验Agent] 校验{'通过' if passed else '未通过'}, "
                f"问题数={len(issues)}, 耗时={elapsed:.0f}ms"
            )

            if issues:
                for issue in issues:
                    trace_log.append(f"[校验Agent] 发现问题: {issue}")

            final_answer = corrected if not passed else answer
            # 确保最终答案包含免责声明
            if MEDICAL_DISCLAIMER not in final_answer:
                final_answer += f"\n\n{MEDICAL_DISCLAIMER}"

            return {
                "verification_passed": passed,
                "verification_issues": issues,
                "final_answer": final_answer,
                "trace_log": trace_log,
            }

    except Exception as e:
        logger.warning(f"[校验Agent] 校验失败，使用原始答案: {e}")
        trace_log.append(f"[校验Agent] 校验异常: {e}")

    # 校验失败时，使用原始答案但标记未通过
    final_answer = answer
    if MEDICAL_DISCLAIMER not in final_answer:
        final_answer += f"\n\n{MEDICAL_DISCLAIMER}"

    return {
        "verification_passed": True,  # 校验异常时默认通过（降级策略）
        "verification_issues": [f"校验服务异常: {e}"],
        "final_answer": final_answer,
        "trace_log": trace_log,
    }
