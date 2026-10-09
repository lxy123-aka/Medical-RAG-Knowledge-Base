"""
向量召回测试

测试内容：
1. 向量入库与检索基本功能
2. 多查询融合检索
3. 分数阈值过滤
4. 来源溯源信息完整性
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 向量检索测试
# =============================================================================

class TestVectorRetrieval:
    """向量检索功能测试"""

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """测试向量检索能返回结果"""
        from src.vector_store.store import VectorStore
        from core.embedding import get_embedding_model

        store = VectorStore()
        await store.initialize()

        embedding_model = get_embedding_model()
        if embedding_model.mode == "uninitialized":
            await embedding_model.initialize()

        # 生成查询向量
        query_vec = await embedding_model.embed_query("高血压的症状")

        # 执行检索
        results = await store.search(
            query_vector=query_vec,
            top_k=5,
            min_score=0.0,  # 低阈值确保有结果
        )

        # 验证结果格式
        assert isinstance(results, list)
        for r in results:
            assert "content" in r, "结果应包含 content 字段"
            assert "score" in r, "结果应包含 score 字段"
            assert "source" in r, "结果应包含 source 字段"

    @pytest.mark.asyncio
    async def test_search_with_queries(self):
        """测试多查询融合检索"""
        from src.vector_store.store import VectorStore
        from core.embedding import get_embedding_model

        store = VectorStore()
        await store.initialize()

        embedding_model = get_embedding_model()
        if embedding_model.mode == "uninitialized":
            await embedding_model.initialize()

        results = await store.search_with_queries(
            query_texts=["高血压症状", "高血压临床表现"],
            embedding_model=embedding_model,
            top_k=3,
            min_score=0.0,
        )

        assert isinstance(results, list)
        # 结果应已去重
        contents = [r["content"][:50] for r in results]
        assert len(contents) == len(set(contents)), "结果应已去重"

    @pytest.mark.asyncio
    async def test_min_score_filter(self):
        """测试最低分数过滤"""
        from src.vector_store.store import VectorStore
        from core.embedding import get_embedding_model

        store = VectorStore()
        await store.initialize()

        embedding_model = get_embedding_model()
        if embedding_model.mode == "uninitialized":
            await embedding_model.initialize()

        query_vec = await embedding_model.embed_query("高血压")

        # 高阈值应返回更少结果
        results_high = await store.search(
            query_vector=query_vec,
            top_k=10,
            min_score=0.9,
        )

        results_low = await store.search(
            query_vector=query_vec,
            top_k=10,
            min_score=0.0,
        )

        assert len(results_high) <= len(results_low), (
            "高阈值结果数应 <= 低阈值结果数"
        )

    @pytest.mark.asyncio
    async def test_source_traceability(self):
        """测试来源溯源信息完整性"""
        from src.vector_store.store import VectorStore
        from core.embedding import get_embedding_model

        store = VectorStore()
        await store.initialize()

        embedding_model = get_embedding_model()
        if embedding_model.mode == "uninitialized":
            await embedding_model.initialize()

        query_vec = await embedding_model.embed_query("糖尿病诊断")

        results = await store.search(
            query_vector=query_vec,
            top_k=3,
            min_score=0.0,
        )

        for r in results:
            source = r.get("source", {})
            assert "doc_id" in source, "来源应包含 doc_id"
            assert "source_type" in source, "来源应包含 source_type"
            assert r.get("retrieval_method") == "vector", "应标注检索方法"


# =============================================================================
# 图谱检索测试
# =============================================================================

class TestGraphRetrieval:
    """图谱检索功能测试"""

    @pytest.mark.asyncio
    async def test_query_by_entity(self):
        """测试实体关联查询"""
        from src.graph_store.store import GraphStore

        store = GraphStore()
        await store.initialize()

        results = await store.query_by_entity("高血压", max_depth=2)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_disease_info(self):
        """测试疾病信息查询"""
        from src.graph_store.store import GraphStore

        store = GraphStore()
        await store.initialize()

        info = await store.query_disease_info("高血压")

        assert "name" in info
        assert "symptoms" in info
        assert "drugs" in info
        assert "exam_items" in info
        assert "treatments" in info

    @pytest.mark.asyncio
    async def test_parallel_entity_query(self):
        """测试多实体并行查询"""
        from src.graph_store.store import GraphStore

        store = GraphStore()
        await store.initialize()

        results = await store.query_by_entities_parallel(
            entity_names=["高血压", "糖尿病"],
            max_depth=1,
        )

        assert isinstance(results, list)
