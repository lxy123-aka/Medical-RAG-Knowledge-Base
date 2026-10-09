"""
LangGraph Agent 状态定义

定义 LangGraph 多Agent流程中流转的状态数据结构。
状态在各节点间传递，每个节点读取并修改状态中的特定字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """
    LangGraph Agent 全局状态

    工作流：
    用户提问 → 问题改写 → 实体抽取 → 并行检索 → 上下文拼接 → 答案生成 → 事实校验 → 输出
    """

    # ── 输入 ──────────────────────────────────────────────────
    original_query: str = ""                  # 用户原始问题

    # ── 问题改写阶段 ──────────────────────────────────────────
    rewritten_query: str = ""                 # 改写后的标准化查询
    sub_queries: List[str] = field(default_factory=list)  # 子查询列表
    intent: str = "general_qa"                # 意图识别结果

    # ── 实体抽取阶段 ──────────────────────────────────────────
    extracted_entities: List[Dict[str, str]] = field(default_factory=list)
    # [{"name": "高血压", "type": "disease"}, ...]

    # ── 检索阶段 ──────────────────────────────────────────────
    vector_results: List[Dict[str, Any]] = field(default_factory=list)
    # 向量检索结果 [{"content": "...", "score": 0.85, "source": {...}}, ...]

    graph_results: List[Dict[str, Any]] = field(default_factory=list)
    # 图谱查询结果 [{"entity": "...", "relations": [...]}, ...]

    # ── 上下文拼接 ────────────────────────────────────────────
    context_text: str = ""                    # 拼接后的上下文文本
    source_documents: List[Dict[str, Any]] = field(default_factory=list)
    # 引用文档来源列表

    # ── 答案生成 ──────────────────────────────────────────────
    generated_answer: str = ""                # LLM 生成的原始答案

    # ── 事实校验 ──────────────────────────────────────────────
    verification_passed: bool = False         # 是否通过事实校验
    verification_issues: List[str] = field(default_factory=list)
    # 发现的问题列表（矛盾、幻觉、错误用药等）
    final_answer: str = ""                    # 最终输出答案

    # ── 元信息 ────────────────────────────────────────────────
    elapsed_ms: float = 0.0                   # 总耗时
    error: Optional[str] = None               # 错误信息
    trace_log: List[str] = field(default_factory=list)
    # 执行轨迹日志
