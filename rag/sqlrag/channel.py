"""
SQLRAG 通道 - PostgreSQL 关系数据库检索（含 NL2SQL）
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
    NL2SQLResult,
    RewriteResult,
)
from services.db.postgres_pool import get_postgres_pool
from services.llm.llm_service import get_llm_service

logger = logging.getLogger(__name__)


# =============================================================================
# PostgreSQL 表结构描述（用于 NL2SQL Prompt）
# =============================================================================

TABLE_SCHEMAS = """
-- 文档元数据表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(128) UNIQUE NOT NULL,       -- 文档唯一标识
    title VARCHAR(512),                         -- 文档标题
    source_type VARCHAR(64) NOT NULL,           -- 来源类型: textbook/record/pharmacopoeia
    department VARCHAR(128),                    -- 所属科室
    permission_level INTEGER DEFAULT 0,         -- 权限等级
    page_count INTEGER,                         -- 页数
    file_path VARCHAR(1024),                    -- 文件路径
    created_at TIMESTAMP DEFAULT NOW(),         -- 创建时间
    updated_at TIMESTAMP DEFAULT NOW()          -- 更新时间
);

-- 文档分片表
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(128) REFERENCES documents(doc_id),
    chunk_index INTEGER NOT NULL,               -- 分片序号
    content TEXT NOT NULL,                      -- 分片内容
    token_count INTEGER,                        -- token 数量
    created_at TIMESTAMP DEFAULT NOW()
);

-- 病历信息表
CREATE TABLE medical_records (
    id SERIAL PRIMARY KEY,
    record_id VARCHAR(128) UNIQUE NOT NULL,     -- 病历编号
    patient_id VARCHAR(128) NOT NULL,           -- 患者编号
    doctor_name VARCHAR(128),                   -- 主治医生
    department VARCHAR(128),                    -- 科室
    admission_date DATE,                        -- 入院日期
    discharge_date DATE,                        -- 出院日期
    chief_complaint TEXT,                       -- 主诉
    diagnosis TEXT,                             -- 诊断
    treatment_plan TEXT,                        -- 治疗方案
    permission_level INTEGER DEFAULT 0,         -- 权限等级
    created_at TIMESTAMP DEFAULT NOW()
);

-- 药物信息表
CREATE TABLE drugs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,                 -- 药物名称
    generic_name VARCHAR(256),                  -- 通用名
    dosage_form VARCHAR(128),                   -- 剂型
    manufacturer VARCHAR(256),                  -- 生产厂家
    approval_number VARCHAR(128),               -- 批准文号
    contraindications TEXT,                     -- 禁忌症
    side_effects TEXT,                          -- 副作用
    dosage_instructions TEXT,                   -- 用法用量
    created_at TIMESTAMP DEFAULT NOW()
);

-- 疾病信息表
CREATE TABLE diseases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,                 -- 疾病名称
    icd_code VARCHAR(32),                       -- ICD 编码
    category VARCHAR(128),                      -- 分类
    description TEXT,                           -- 描述
    prevalence VARCHAR(128),                    -- 患病率
    common_symptoms TEXT,                       -- 常见症状
    created_at TIMESTAMP DEFAULT NOW()
);
"""


class SQLRAGChannel:
    """
    SQLRAG 关系数据库检索通道

    流程:
    1. NL2SQL: 自然语言 → SQL 查询
    2. 执行 SQL 查询
    3. 结果格式化
    """

    def __init__(self):
        self._channel_type = ChannelType.SQLRAG

    async def search(
        self,
        query: str,
        rewrite: Optional[RewriteResult] = None,
        top_k: Optional[int] = None,
        permission_level: int = 0,
        **kwargs,
    ) -> ChannelResult:
        """
        执行 SQLRAG 检索

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
        top_k = top_k or settings.sqlrag_top_k

        try:
            # 1. NL2SQL 转换
            nl2sql_result = await self._nl2sql(query, permission_level)

            if not nl2sql_result.success:
                elapsed_ms = (time.time() - start_time) * 1000
                return ChannelResult(
                    channel=self._channel_type,
                    status=ChannelStatus.FAILED,
                    results=[],
                    elapsed_ms=round(elapsed_ms, 2),
                    error_message=f"NL2SQL 失败: {nl2sql_result.error_message}",
                )

            # 2. 格式化结果
            formatted_results = self._format_results(
                nl2sql_result.execution_result or [],
                nl2sql_result.generated_sql,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            status = ChannelStatus.SUCCESS if formatted_results else ChannelStatus.EMPTY

            logger.info(
                f"[SQLRAG] 检索完成 | 结果数: {len(formatted_results)} | "
                f"耗时: {elapsed_ms:.1f}ms"
            )

            return ChannelResult(
                channel=self._channel_type,
                status=status,
                results=formatted_results[:top_k],
                elapsed_ms=round(elapsed_ms, 2),
                metadata={
                    "sql": nl2sql_result.generated_sql,
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"[SQLRAG] 检索失败: {e}", exc_info=True)
            return ChannelResult(
                channel=self._channel_type,
                status=ChannelStatus.FAILED,
                results=[],
                elapsed_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )

    async def _nl2sql(
        self,
        query: str,
        permission_level: int = 0,
    ) -> NL2SQLResult:
        """
        NL2SQL: 自然语言转 SQL 查询

        自动添加权限过滤条件
        """
        llm_service = get_llm_service()
        postgres_pool = get_postgres_pool()

        try:
            # 1. 生成 SQL
            raw_sql = await llm_service.nl2sql(query, TABLE_SCHEMAS)

            # 2. 清理 SQL（移除可能的 markdown 标记）
            sql = self._clean_sql(raw_sql)
            if not sql:
                return NL2SQLResult(
                    original_query=query,
                    success=False,
                    error_message="无法从 LLM 输出中提取 SQL 语句",
                )

            # 3. 添加权限过滤
            sql = self._add_permission_filter(sql, permission_level)

            logger.info(f"[NL2SQL] 生成 SQL: {sql[:100]}")

            # 4. 执行 SQL
            try:
                results = await postgres_pool.execute_query(sql)
                return NL2SQLResult(
                    original_query=query,
                    generated_sql=sql,
                    execution_result=results,
                    success=True,
                )
            except Exception as exec_err:
                error_msg = f"SQL 执行失败: {str(exec_err)}"
                logger.error(f"[NL2SQL] {error_msg}")
                return NL2SQLResult(
                    original_query=query,
                    generated_sql=sql,
                    success=False,
                    error_message=error_msg,
                )

        except Exception as e:
            logger.error(f"[NL2SQL] 流程异常: {e}", exc_info=True)
            return NL2SQLResult(
                original_query=query,
                success=False,
                error_message=str(e),
            )

    def _clean_sql(self, raw_sql: str) -> str:
        """清理 SQL 语句（移除 markdown 标记等）"""
        import re

        # 移除 ```sql ... ``` 标记
        pattern = r'```(?:sql)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, raw_sql, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 移除前后空白
        sql = raw_sql.strip()

        # 移除末尾分号（某些执行环境不需要）
        if sql.endswith(';'):
            sql = sql[:-1]

        return sql

    def _add_permission_filter(self, sql: str, permission_level: int) -> str:
        """
        添加权限过滤条件

        对包含 permission_level 字段的表自动添加过滤
        """
        sql_upper = sql.upper()

        # 如果已经有 WHERE 子句，追加条件
        if "WHERE" in sql_upper:
            # 简单追加（实际应更智能地解析 SQL）
            if "PERMISSION_LEVEL" not in sql_upper:
                # 在 ORDER BY / LIMIT / GROUP BY 之前插入
                for keyword in ["ORDER BY", "LIMIT", "GROUP BY", "HAVING"]:
                    if keyword in sql_upper:
                        idx = sql_upper.index(keyword)
                        sql = (
                            sql[:idx]
                            + f"AND permission_level <= {permission_level} "
                            + sql[idx:]
                        )
                        return sql

                # 没有这些关键字，直接追加
                sql += f" AND permission_level <= {permission_level}"
        else:
            # 没有 WHERE，在 FROM 子句后添加
            # 简单处理：在表名后添加 WHERE
            if "FROM" in sql_upper and "PERMISSION_LEVEL" not in sql_upper:
                # 找到 FROM 后的表名
                from_idx = sql_upper.index("FROM") + 4
                rest = sql[from_idx:]
                stripped = rest.lstrip()
                leading = len(rest) - len(stripped)  # 保留表名前的原始空格

                # 找到表名结束位置（空格、逗号、换行等）
                end_idx = 0
                for i, ch in enumerate(stripped):
                    if ch in (' ', '\n', '\t', ','):
                        end_idx = i
                        break

                if end_idx > 0:
                    table_name = stripped[:end_idx]
                    sql = (
                        sql[:from_idx]
                        + " " * max(leading, 1)  # 至少保留一个空格，避免 FROMdocuments
                        + table_name
                        + f" WHERE permission_level <= {permission_level}"
                        + stripped[end_idx:]
                    )
                else:
                    # 表名是 SQL 最后一个 token（无后续空格/逗号）
                    sql = (
                        sql[:from_idx]
                        + " " * max(leading, 1)
                        + stripped
                        + f" WHERE permission_level <= {permission_level}"
                    )

        return sql

    def _format_results(
        self,
        records: List[Dict[str, Any]],
        sql: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """格式化 SQL 查询结果"""
        formatted = []
        for record in records:
            # 将记录转换为可读文本
            content_parts = []
            for key, value in record.items():
                if value is not None:
                    content_parts.append(f"{key}: {value}")

            content = " | ".join(content_parts) if content_parts else str(record)

            formatted.append({
                "source": "sqlrag",
                "content": content,
                "score": 1.0,  # SQL 查询结果给固定高分
                "metadata": {
                    "sql": sql,
                    "record": record,
                },
            })

        return formatted


# ── 全局单例 ─────────────────────────────────────────────────
_sqlrag_channel: Optional[SQLRAGChannel] = None


def get_sqlrag_channel() -> SQLRAGChannel:
    """获取 SQLRAG 通道全局单例"""
    global _sqlrag_channel
    if _sqlrag_channel is None:
        _sqlrag_channel = SQLRAGChannel()
    return _sqlrag_channel
