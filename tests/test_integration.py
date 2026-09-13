"""
测试用例 - 完整三通道检索 + 融合 + NL2Cypher 重试流程

运行方式:
    python -m pytest tests/test_integration.py -v
"""

import asyncio
import json
import pytest
import sys
import os

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# 测试 safe_json_parse（规则6/17: JSON 解析容错测试）
# =============================================================================

class TestSafeJsonParse:
    """规则17: JSON 解析容错测试（纯 JSON / markdown 包裹 / 带额外文字三种输入）"""

    def test_pure_json_dict(self):
        """纯 JSON dict"""
        from core.safe_json import safe_json_parse
        result = safe_json_parse('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_pure_json_list(self):
        """纯 JSON list"""
        from core.safe_json import safe_json_parse
        result = safe_json_parse('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_dict_passthrough(self):
        """dict 直接返回"""
        from core.safe_json import safe_json_parse
        data = {"key": "value"}
        result = safe_json_parse(data)
        assert result is data

    def test_markdown_wrapped_json(self):
        """markdown 包裹的 JSON"""
        from core.safe_json import extract_json_from_markdown
        text = '''
这是回答。
```json
{"standardized_query": "高血压", "intent": "disease_query"}
```
以上是结果。
'''
        result = extract_json_from_markdown(text)
        assert result is not None
        assert result["intent"] == "disease_query"

    def test_json_with_extra_text(self):
        """带额外文字的 JSON"""
        from core.safe_json import safe_json_parse
        text = '好的，以下是结果：{"name": "高血压", "code": "I10"} 希望对你有帮助'
        result = safe_json_parse(text)
        assert result is not None
        assert result["name"] == "高血压"

    def test_single_quotes_json(self):
        """单引号 JSON（容错修复）"""
        from core.safe_json import safe_json_parse
        text = "{'key': 'value', 'num': 42}"
        result = safe_json_parse(text)
        assert result is not None
        assert result["key"] == "value"

    def test_trailing_comma_json(self):
        """尾部逗号 JSON（容错修复）"""
        from core.safe_json import safe_json_parse
        text = '{"key": "value",}'
        result = safe_json_parse(text)
        assert result is not None
        assert result["key"] == "value"

    def test_invalid_json_returns_none(self):
        """无效 JSON 返回 None"""
        from core.safe_json import safe_json_parse
        result = safe_json_parse("这不是 JSON")
        assert result is None

    def test_empty_string_returns_none(self):
        """空字符串返回 None"""
        from core.safe_json import safe_json_parse
        result = safe_json_parse("")
        assert result is None


# =============================================================================
# 测试 Cypher 提取
# =============================================================================

class TestCypherExtraction:
    """测试 Cypher 语句提取"""

    def test_markdown_wrapped_cypher(self):
        """markdown 包裹的 Cypher"""
        from core.safe_json import extract_cypher_from_text
        text = '''
```cypher
MATCH (d:Disease {name: '高血压'})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name, s.name
```
'''
        result = extract_cypher_from_text(text)
        assert result is not None
        assert "MATCH" in result
        assert "HAS_SYMPTOM" in result

    def test_plain_cypher(self):
        """纯 Cypher 语句"""
        from core.safe_json import extract_cypher_from_text
        text = "MATCH (d:Disease) RETURN d.name LIMIT 10"
        result = extract_cypher_from_text(text)
        assert result is not None
        assert "MATCH" in result


# =============================================================================
# 测试 Neo4j Schema 完整性
# =============================================================================

class TestNeo4jSchema:
    """测试 Neo4j Schema 完整性"""

    def test_all_node_labels_defined(self):
        """所有节点标签都有属性 Schema"""
        from core.neo4j_schema import get_all_node_labels, NODE_PROPERTY_SCHEMAS
        for label in get_all_node_labels():
            assert label in NODE_PROPERTY_SCHEMAS, f"节点 {label} 缺少属性 Schema"

    def test_all_rel_types_defined(self):
        """所有关系类型都有属性 Schema"""
        from core.neo4j_schema import get_all_rel_types, RELATIONSHIP_PROPERTY_SCHEMAS
        for rel_type in get_all_rel_types():
            assert rel_type in RELATIONSHIP_PROPERTY_SCHEMAS, f"关系 {rel_type} 缺少属性 Schema"

    def test_schema_summary_not_empty(self):
        """Schema 摘要不为空"""
        from core.neo4j_schema import build_schema_summary
        summary = build_schema_summary()
        assert len(summary) > 100
        assert "节点类型" in summary
        assert "关系类型" in summary

    def test_all_property_names_unique(self):
        """属性名集合不为空"""
        from core.neo4j_schema import get_all_property_names
        names = get_all_property_names()
        assert len(names) > 0
        assert "name" in names
        assert "weight" in names


# =============================================================================
# 测试 Milvus Schema 一致性
# =============================================================================

class TestMilvusSchema:
    """测试 Milvus Schema 一致性"""

    def test_dimension_consistency(self):
        """Embedding 维度一致"""
        from core.milvus_schema import validate_dimension_consistency
        assert validate_dimension_consistency() is True

    def test_all_collections_have_embedding(self):
        """所有集合都有 embedding 字段"""
        from core.milvus_schema import SCHEMA_REGISTRY
        from pymilvus import DataType
        for name, builder in SCHEMA_REGISTRY.items():
            schema = builder()
            embedding_fields = [
                f for f in schema.fields if f.dtype == DataType.FLOAT_VECTOR
            ]
            assert len(embedding_fields) >= 1, f"集合 {name} 缺少 embedding 字段"

    def test_collection_names_defined(self):
        """集合名称已定义"""
        from core.milvus_schema import get_all_collection_names
        names = get_all_collection_names()
        assert len(names) >= 3


# =============================================================================
# 测试配置
# =============================================================================

class TestSettings:
    """测试配置系统"""

    def test_settings_load(self):
        """配置可加载"""
        from config.settings import get_settings
        settings = get_settings()
        assert settings.app_name != ""
        assert settings.embedding_dimension == 1024

    def test_enabled_channels(self):
        """通道开关可获取"""
        from config.settings import get_settings
        settings = get_settings()
        channels = settings.get_enabled_channels()
        assert isinstance(channels, list)

    def test_postgres_dsn(self):
        """PostgreSQL DSN 格式正确"""
        from config.settings import get_settings
        settings = get_settings()
        dsn = settings.postgres_dsn
        assert dsn.startswith("postgresql+asyncpg://")


# =============================================================================
# 测试数据模型
# =============================================================================

class TestModels:
    """测试 Pydantic 数据模型"""

    def test_query_request(self):
        """QueryRequest 模型"""
        from models.schemas import QueryRequest
        req = QueryRequest(query="高血压的症状")
        assert req.query == "高血压的症状"
        assert req.top_k == 5
        assert req.enable_hyde is True

    def test_channel_result(self):
        """ChannelResult 模型"""
        from models.schemas import ChannelResult, ChannelType, ChannelStatus
        result = ChannelResult(
            channel=ChannelType.DOCRAG,
            status=ChannelStatus.SUCCESS,
            results=[{"content": "test", "score": 0.9}],
            elapsed_ms=100.0,
        )
        assert result.channel == ChannelType.DOCRAG
        assert len(result.results) == 1

    def test_rewrite_result(self):
        """RewriteResult 模型"""
        from models.schemas import RewriteResult, IntentType
        result = RewriteResult(
            original_query="头疼",
            standardized_query="头痛",
            sub_queries=["头痛的常见原因", "头痛的治疗方法"],
            intent=IntentType.SYMPTOM_ANALYSIS,
        )
        assert result.standardized_query == "头痛"
        assert len(result.sub_queries) == 2


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
