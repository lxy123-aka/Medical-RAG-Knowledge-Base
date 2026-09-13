"""
Milvus Schema 集中管理（规则3: 字段类型必须使用 DataType 枚举，维度一致）
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pymilvus import CollectionSchema, DataType, FieldSchema


# =============================================================================
# 全局 Embedding 维度常量
# =============================================================================
EMBEDDING_DIM = 1024  # bge-large-zh-v1.5 输出维度


# =============================================================================
# 集合名称常量
# =============================================================================
class CollectionNames:
    DOC_RAG = "doc_rag_chunks"          # DocRAG 文档分片
    GRAPH_RAG = "graph_rag_embeddings"  # GraphRAG 实体/关系嵌入
    GENERAL = "general_knowledge"       # 通用知识


# =============================================================================
# Schema 定义函数
# =============================================================================

def build_doc_rag_schema() -> CollectionSchema:
    """
    DocRAG 文档分片集合 Schema
    字段: id, doc_id, chunk_index, content, embedding, metadata
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128,
                    description="文档唯一标识"),
        FieldSchema(name="chunk_index", dtype=DataType.INT64,
                    description="分片序号"),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192,
                    description="分片文本内容"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                    description="文本向量嵌入"),
        # ── 权限过滤 meta 字段（规则: 基于 meta 字段实现权限过滤切片） ──
        FieldSchema(name="permission_level", dtype=DataType.INT64,
                    description="权限等级，用于过滤"),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64,
                    description="来源类型: textbook/record/pharmacopoeia"),
        FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=128,
                    description="所属科室"),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32,
                    description="创建时间"),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="DocRAG 文档分片向量集合",
    )
    return schema


def build_graph_rag_schema() -> CollectionSchema:
    """
    GraphRAG 实体/关系嵌入集合 Schema
    用于图谱节点的语义检索
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=128,
                    description="实体唯一标识（对应 Neo4j 节点 ID）"),
        FieldSchema(name="entity_label", dtype=DataType.VARCHAR, max_length=64,
                    description="实体标签/类型"),
        FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512,
                    description="实体名称"),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096,
                    description="实体描述文本"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                    description="实体描述向量嵌入"),
        FieldSchema(name="permission_level", dtype=DataType.INT64,
                    description="权限等级"),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="GraphRAG 实体嵌入向量集合",
    )
    return schema


def build_general_schema() -> CollectionSchema:
    """通用知识集合 Schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192,
                    description="文本内容"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM,
                    description="向量嵌入"),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256,
                    description="来源"),
        FieldSchema(name="permission_level", dtype=DataType.INT64,
                    description="权限等级"),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="通用知识向量集合",
    )
    return schema


# =============================================================================
# Schema 注册表
# =============================================================================

SCHEMA_REGISTRY: Dict[str, callable] = {
    CollectionNames.DOC_RAG: build_doc_rag_schema,
    CollectionNames.GRAPH_RAG: build_graph_rag_schema,
    CollectionNames.GENERAL: build_general_schema,
}


def get_all_collection_names() -> List[str]:
    """获取所有集合名称"""
    return list(SCHEMA_REGISTRY.keys())


def validate_dimension_consistency() -> bool:
    """
    规则3: 所有集合的 Embedding 维度保持一致
    返回 True 表示一致，False 表示不一致
    """
    dims = set()
    for name, builder in SCHEMA_REGISTRY.items():
        schema = builder()
        for field in schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                dims.add(field.params.get("dim", 0))
    if len(dims) != 1:
        raise ValueError(
            f"Embedding 维度不一致! 发现以下维度: {dims}，"
            f"所有集合必须使用统一的 {EMBEDDING_DIM} 维"
        )
    return True


# =============================================================================
# 索引配置
# =============================================================================

INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16},
}
