"""
验证清单脚本（规则17）

检查项目:
1. 语法检查（python -m py_compile）
2. 属性名一致性检查（查询代码 vs 初始化 Schema）
3. 环境变量生效检查
4. 数据库连通性检查
5. JSON 解析容错测试（纯 JSON / markdown 包裹 / 带额外文字三种输入）
"""

import asyncio
import json
import os
import py_compile
import sys
import traceback
from pathlib import Path

# 将项目根目录加入 sys.path
_project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, _project_root)
os.chdir(_project_root)

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    msg = f"  {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    results.append((name, passed))
    return passed


def check_syntax():
    """1. 语法检查"""
    print("\n" + "=" * 50)
    print("  1. 语法检查 (py_compile)")
    print("=" * 50)

    root = Path(_project_root)
    # 排除虚拟环境等非项目代码目录
    skip_dirs = {".venv", "venv", ".git", ".idea", "__pycache__", ".pytest_cache", "node_modules", "logs", "data"}
    py_files = [p for p in root.rglob("*.py") if not any(part in skip_dirs for part in p.parts)]
    errors = []

    for py_file in py_files:
        # 跳过 __pycache__
        if "__pycache__" in str(py_file):
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append((str(py_file), str(e)))

    total = len(py_files)
    passed = total - len(errors)
    check(f"Python 文件语法检查 ({passed}/{total})", len(errors) == 0,
          f"{len(errors)} 个文件有语法错误" if errors else "")

    for f, e in errors:
        print(f"    ✗ {f}: {e}")


def check_schema_consistency():
    """2. 属性名一致性检查"""
    print("\n" + "=" * 50)
    print("  2. Schema 属性名一致性检查")
    print("=" * 50)

    try:
        from core.neo4j_schema import (
            get_all_node_labels, get_all_rel_types,
            NODE_PROPERTY_SCHEMAS, RELATIONSHIP_PROPERTY_SCHEMAS,
            get_all_property_names,
        )

        # 所有节点标签都有 Schema
        all_labels = get_all_node_labels()
        labels_with_schema = list(NODE_PROPERTY_SCHEMAS.keys())
        check("节点标签 Schema 覆盖",
              set(all_labels) <= set(labels_with_schema),
              f"节点: {len(all_labels)}, Schema: {len(labels_with_schema)}")

        # 所有关系类型都有 Schema
        all_rels = get_all_rel_types()
        rels_with_schema = list(RELATIONSHIP_PROPERTY_SCHEMAS.keys())
        check("关系类型 Schema 覆盖",
              set(all_rels) <= set(rels_with_schema),
              f"关系: {len(all_rels)}, Schema: {len(rels_with_schema)}")

        # 属性名集合
        all_props = get_all_property_names()
        check("属性名集合非空", len(all_props) > 0, f"共 {len(all_props)} 个属性名")

    except Exception as e:
        check("Schema 一致性检查", False, str(e))


def check_milvus_schema():
    """Milvus Schema 检查"""
    print("\n" + "=" * 50)
    print("  3. Milvus Schema 维度一致性检查")
    print("=" * 50)

    try:
        from core.milvus_schema import validate_dimension_consistency, EMBEDDING_DIM
        result = validate_dimension_consistency()
        check("Embedding 维度一致性", result, f"维度: {EMBEDDING_DIM}")
    except Exception as e:
        check("Milvus Schema 检查", False, str(e))


def check_settings():
    """4. 环境变量生效检查"""
    print("\n" + "=" * 50)
    print("  4. 环境变量/配置检查")
    print("=" * 50)

    try:
        from config.settings import get_settings, validate_startup_settings
        settings = get_settings()

        check("配置可加载", True, f"app_name={settings.app_name}")
        check("Embedding 维度配置", settings.embedding_dimension == 1024)
        check("通道开关可获取", len(settings.get_enabled_channels()) > 0,
              f"启用通道: {settings.get_enabled_channels()}")
        check("PostgreSQL DSN 格式正确",
              settings.postgres_dsn.startswith("postgresql+asyncpg://"))

        # 检查必填项
        errors = validate_startup_settings(settings)
        warnings = [e for e in errors if "将尝试" in e]
        critical = [e for e in errors if "未配置" in e]

        if warnings:
            for w in warnings:
                print(f"    {WARN} {w}")
        if critical:
            check("关键配置完整", False, f"缺失: {critical}")
        else:
            check("关键配置完整", True)

    except Exception as e:
        check("配置检查", False, str(e))


def check_json_parse():
    """5. JSON 解析容错测试"""
    print("\n" + "=" * 50)
    print("  5. JSON 解析容错测试")
    print("=" * 50)

    from core.safe_json import safe_json_parse, extract_json_from_markdown

    # 纯 JSON
    check("纯 JSON dict",
          safe_json_parse('{"key": "value"}') == {"key": "value"})

    check("纯 JSON list",
          safe_json_parse('[1, 2, 3]') == [1, 2, 3])

    # dict 透传
    d = {"a": 1}
    check("dict 直接返回",
          safe_json_parse(d) is d)

    # markdown 包裹
    md_text = '```json\n{"name": "test"}\n```'
    check("Markdown 包裹 JSON",
          extract_json_from_markdown(md_text) == {"name": "test"})

    # 带额外文字
    extra_text = '以下是结果：{"name": "高血压"} 希望有帮助'
    result = safe_json_parse(extra_text)
    check("带额外文字的 JSON",
          result is not None and result.get("name") == "高血压")

    # 单引号
    single_quote = "{'key': 'value'}"
    result = safe_json_parse(single_quote)
    check("单引号 JSON 容错",
          result is not None and result.get("key") == "value")

    # 尾部逗号
    trailing = '{"key": "value",}'
    result = safe_json_parse(trailing)
    check("尾部逗号 JSON 容错",
          result is not None and result.get("key") == "value")

    # 无效 JSON
    check("无效 JSON 返回 None",
          safe_json_parse("这不是JSON") is None)

    check("空字符串返回 None",
          safe_json_parse("") is None)


async def check_db_connectivity():
    """6. 数据库连通性检查（可选，数据库未启动时跳过）"""
    print("\n" + "=" * 50)
    print("  6. 数据库连通性检查（可选）")
    print("=" * 50)

    # Neo4j
    try:
        from services.db.neo4j_pool import get_neo4j_pool
        pool = get_neo4j_pool()
        await pool.initialize()
        health = await pool.health_check()
        check("Neo4j 连通", health["status"] == "healthy",
              f"延迟: {health.get('latency_ms')}ms")
        await pool.close()
    except Exception as e:
        check("Neo4j 连通", False, f"连接失败（可能未启动）: {str(e)[:80]}")

    # Milvus
    try:
        from services.db.milvus_pool import get_milvus_pool
        pool = get_milvus_pool()
        await pool.initialize()
        health = await pool.health_check()
        check("Milvus 连通", health["status"] == "healthy",
              f"延迟: {health.get('latency_ms')}ms")
        await pool.close()
    except Exception as e:
        check("Milvus 连通", False, f"连接失败（可能未启动）: {str(e)[:80]}")

    # PostgreSQL
    try:
        from services.db.postgres_pool import get_postgres_pool
        pool = get_postgres_pool()
        await pool.initialize()
        health = await pool.health_check()
        check("PostgreSQL 连通", health["status"] == "healthy",
              f"延迟: {health.get('latency_ms')}ms")
        await pool.close()
    except Exception as e:
        check("PostgreSQL 连通", False, f"连接失败（可能未启动）: {str(e)[:80]}")


def main():
    print("=" * 50)
    print("  医典 RAG 多维知识库系统 - 验证清单")
    print("=" * 50)

    # 1-5: 离线检查
    check_syntax()
    check_schema_consistency()
    check_milvus_schema()
    check_settings()
    check_json_parse()

    # 6: 数据库连通性（可选）
    asyncio.run(check_db_connectivity())

    # 汇总
    print("\n" + "=" * 50)
    print("  验证结果汇总")
    print("=" * 50)

    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)
    total = len(results)

    print(f"  总计: {total} | 通过: {passed} | 失败: {failed}")

    if failed > 0:
        print(f"\n  失败项:")
        for name, p in results:
            if not p:
                print(f"    ✗ {name}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
