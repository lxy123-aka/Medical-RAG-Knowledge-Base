"""
LangGraph 图构建 - 医疗RAG多Agent工作流编排

设计说明：
1. 使用 LangGraph StateGraph 编排多个 Agent 节点
2. 工作流：
   用户提问 → 问题改写Agent → 医学实体抽取Agent
   → 并行执行(Milvus向量召回 + Neo4j图谱查询)
   → 上下文拼接 → LLM生成答案 → 医学事实校验Agent → 输出
3. 向量检索和图谱检索并行执行，提升效率
4. 错误降级：单节点失败不阻塞整体流程

使用方法：
    from src.agent import build_medical_rag_graph
    graph = build_medical_rag_graph()
    result = await graph.ainvoke({"original_query": "高血压有什么症状"})
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .nodes import (
    query_rewrite_node,
    entity_extract_node,
    vector_retrieve_node,
    graph_retrieve_node,
    context_assemble_node,
    answer_generate_node,
    fact_verify_node,
)

logger = logging.getLogger(__name__)


def build_medical_rag_graph():
    """
    构建医疗RAG LangGraph 工作流

    流程图：
    ┌──────────────┐
    │  用户提问     │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  问题改写     │  query_rewrite_node
    │  Agent        │  (口语→术语, 子查询拆分)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  实体抽取     │  entity_extract_node
    │  Agent        │  (疾病/症状/药物/检查)
    └──────┬───────┘
           │
    ┌──────▼───────────────────────┐
    │         并行检索              │
    │  ┌─────────┐  ┌──────────┐  │
    │  │向量检索  │  │图谱检索   │  │
    │  │(Milvus) │  │(Neo4j)   │  │
    │  └────┬────┘  └────┬─────┘  │
    └───────┼────────────┼────────┘
            │            │
    ┌───────▼────────────▼────────┐
    │       上下文拼接             │  context_assemble_node
    └───────────┬─────────────────┘
                │
    ┌───────────▼─────────────────┐
    │       答案生成              │  answer_generate_node
    │       (LLM + 溯源)          │
    └───────────┬─────────────────┘
                │
    ┌───────────▼─────────────────┐
    │       事实校验              │  fact_verify_node
    │       (幻觉/矛盾检测)       │
    └───────────┬─────────────────┘
                │
    ┌───────────▼─────────────────┐
    │       输出答案              │
    │  + 引用来源 + 免责声明      │
    └─────────────────────────────┘

    Returns:
        编译后的 LangGraph CompiledGraph
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError(
            "请安装 langgraph: pip install langgraph"
        )

    from .state import AgentState

    # 构建状态图
    workflow = StateGraph(dict)

    # ── 添加节点 ──────────────────────────────────────────────
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("entity_extract", entity_extract_node)
    workflow.add_node("vector_retrieve", vector_retrieve_node)
    workflow.add_node("graph_retrieve", graph_retrieve_node)
    workflow.add_node("context_assemble", context_assemble_node)
    workflow.add_node("answer_generate", answer_generate_node)
    workflow.add_node("fact_verify", fact_verify_node)

    # ── 设置入口 ──────────────────────────────────────────────
    workflow.set_entry_point("query_rewrite")

    # ── 定义边（流转关系） ────────────────────────────────────

    # 问题改写 → 实体抽取
    workflow.add_edge("query_rewrite", "entity_extract")

    # 实体抽取 → 并行检索（向量 + 图谱）
    # LangGraph 支持从一个节点扇出到多个节点并行执行
    workflow.add_edge("entity_extract", "vector_retrieve")
    workflow.add_edge("entity_extract", "graph_retrieve")

    # 向量检索 + 图谱检索 → 上下文拼接（汇聚）
    workflow.add_edge("vector_retrieve", "context_assemble")
    workflow.add_edge("graph_retrieve", "context_assemble")

    # 上下文拼接 → 答案生成
    workflow.add_edge("context_assemble", "answer_generate")

    # 答案生成 → 事实校验
    workflow.add_edge("answer_generate", "fact_verify")

    # 事实校验 → 结束
    workflow.add_edge("fact_verify", END)

    # ── 编译 ──────────────────────────────────────────────────
    graph = workflow.compile()
    logger.info("LangGraph 医疗RAG工作流构建完成")
    return graph


async def run_medical_rag(query: str) -> Dict[str, Any]:
    """
    运行完整的医疗RAG问答流程

    Args:
        query: 用户问题

    Returns:
        完整的问答结果字典
    """
    start_time = time.time()

    logger.info(f"{'='*60}")
    logger.info(f"  医疗RAG问答开始: {query[:80]}")
    logger.info(f"{'='*60}")

    try:
        graph = build_medical_rag_graph()

        # 初始状态
        initial_state = {
            "original_query": query,
            "rewritten_query": "",
            "sub_queries": [],
            "intent": "",
            "extracted_entities": [],
            "vector_results": [],
            "graph_results": [],
            "context_text": "",
            "source_documents": [],
            "generated_answer": "",
            "verification_passed": False,
            "verification_issues": [],
            "final_answer": "",
            "elapsed_ms": 0.0,
            "error": None,
            "trace_log": [],
        }

        # 执行图
        result = await graph.ainvoke(initial_state)

        elapsed = (time.time() - start_time) * 1000
        result["elapsed_ms"] = elapsed

        logger.info(f"{'='*60}")
        logger.info(f"  问答完成, 总耗时: {elapsed:.0f}ms")
        logger.info(f"  校验通过: {result.get('verification_passed', False)}")
        logger.info(f"  引用文档: {len(result.get('source_documents', []))} 个")
        for log_entry in result.get("trace_log", []):
            logger.info(f"  {log_entry}")
        logger.info(f"{'='*60}")

        return result

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"医疗RAG问答失败: {e}")
        return {
            "original_query": query,
            "final_answer": f"系统处理出错，请稍后重试。错误: {e}",
            "elapsed_ms": elapsed,
            "error": str(e),
            "trace_log": [f"[错误] {e}"],
        }
