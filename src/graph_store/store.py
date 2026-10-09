"""
图谱存储模块 - Neo4j 图谱构建与查询封装

设计说明：
1. 封装 Neo4j 操作，提供面向业务的高层接口
2. 医学实体/关系批量写入图谱（MERGE 防重复）
3. 基于实体名称的关联查询（疾病→症状→药物→检查）
4. 支持自然语言转 Cypher 查询（NL2Cypher）
5. 查询结果附带来源溯源信息

使用方法：
    store = GraphStore()
    await store.initialize()
    # 写入实体
    await store.upsert_entities(entities, relations)
    # 按实体名查询关联
    results = await store.query_by_entity("高血压", max_depth=2)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


class GraphStore:
    """
    图谱存储封装

    基于现有 Neo4jPool 提供面向业务的高层接口：
    - 医学实体/关系批量写入
    - 基于实体的关联查询
    - Cypher 查询执行
    """

    def __init__(self):
        self._neo4j_pool = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化，获取 Neo4j 连接"""
        if self._initialized:
            return
        from services.db.neo4j_pool import get_neo4j_pool
        self._neo4j_pool = get_neo4j_pool()
        if not self._neo4j_pool._initialized:
            await self._neo4j_pool.initialize()
        self._initialized = True
        logger.info("GraphStore 初始化完成")

    # ── 实体/关系写入 ─────────────────────────────────────────

    async def upsert_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量写入/更新实体节点（MERGE 防重复）

        Args:
            entities: 实体列表，每个实体包含:
                - label: 节点标签 (Disease/Symptom/Drug/...)
                - name: 实体名称（唯一标识）
                - properties: 属性字典

        Returns:
            写入统计
        """
        if not self._initialized:
            await self.initialize()

        if not entities:
            return {"upserted": 0}

        total = 0
        for entity in entities:
            label = entity.get("label", "")
            name = entity.get("name", "")
            properties = entity.get("properties", {})

            if not label or not name:
                continue

            # 构造 MERGE 语句
            props_set = ", ".join(
                f"n.{k} = ${k}" for k in properties.keys()
            )
            if props_set:
                cypher = f"""
                MERGE (n:{label} {{name: $name}})
                SET {props_set}
                RETURN n.name AS name
                """
            else:
                cypher = f"""
                MERGE (n:{label} {{name: $name}})
                RETURN n.name AS name
                """

            params = {"name": name, **properties}

            try:
                await self._neo4j_pool.execute_write(cypher, params)
                total += 1
            except Exception as e:
                logger.warning(f"写入实体失败: {label}:{name}, 错误: {e}")

        logger.info(f"实体写入完成: {total}/{len(entities)} 个")
        return {"upserted": total, "total": len(entities)}

    async def upsert_relations(
        self,
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量写入/更新关系（MERGE 防重复）

        Args:
            relations: 关系列表，每个关系包含:
                - start_label: 起始节点标签
                - start_name: 起始节点名称
                - rel_type: 关系类型
                - end_label: 目标节点标签
                - end_name: 目标节点名称
                - properties: 关系属性字典（可选）

        Returns:
            写入统计
        """
        if not self._initialized:
            await self.initialize()

        if not relations:
            return {"upserted": 0}

        total = 0
        for rel in relations:
            start_label = rel.get("start_label", "")
            start_name = rel.get("start_name", "")
            rel_type = rel.get("rel_type", "")
            end_label = rel.get("end_label", "")
            end_name = rel.get("end_name", "")
            properties = rel.get("properties", {})

            if not all([start_label, start_name, rel_type, end_label, end_name]):
                continue

            # 构造关系属性 SET
            props_set = ""
            if properties:
                props_items = ", ".join(
                    f"r.{k} = ${k}" for k in properties.keys()
                )
                props_set = f" SET {props_items}"

            cypher = f"""
            MERGE (a:{start_label} {{name: $start_name}})
            MERGE (b:{end_label} {{name: $end_name}})
            MERGE (a)-[r:{rel_type}]->(b)
            {props_set}
            RETURN a.name AS from, b.name AS to
            """

            params = {
                "start_name": start_name,
                "end_name": end_name,
                **properties,
            }

            try:
                await self._neo4j_pool.execute_write(cypher, params)
                total += 1
            except Exception as e:
                logger.warning(
                    f"写入关系失败: {start_name}-[{rel_type}]->{end_name}, 错误: {e}"
                )

        logger.info(f"关系写入完成: {total}/{len(relations)} 条")
        return {"upserted": total, "total": len(relations)}

    # ── 图谱查询 ──────────────────────────────────────────────

    async def query_by_entity(
        self,
        entity_name: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        基于实体名称的关联查询

        查询指定实体的所有关联节点（N跳邻居）

        Args:
            entity_name: 实体名称
            max_depth: 最大查询深度
            limit: 返回数量限制

        Returns:
            关联信息列表
        """
        if not self._initialized:
            await self.initialize()

        cypher = f"""
        MATCH (n)-[r*1..{max_depth}]-(m)
        WHERE n.name = $entity_name
        RETURN n.name AS source, type(last(r)) AS relation,
               last(nodes(r)).name AS target,
               last(nodes(r)) AS target_node
        LIMIT $limit
        """

        try:
            results = await self._neo4j_pool.execute_read(
                cypher, {"entity_name": entity_name, "limit": limit}
            )
            logger.info(
                f"实体关联查询: '{entity_name}' 返回 {len(results)} 条关联"
            )
            return results
        except Exception as e:
            logger.error(f"实体关联查询失败: {e}")
            return []

    async def query_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """
        查询疾病完整信息（症状、药物、检查、治疗方案）

        Args:
            disease_name: 疾病名称

        Returns:
            疾病信息字典
        """
        if not self._initialized:
            await self.initialize()

        info = {
            "name": disease_name,
            "symptoms": [],
            "drugs": [],
            "exam_items": [],
            "treatments": [],
            "causes": [],
        }

        # 查询症状
        try:
            results = await self._neo4j_pool.execute_read(
                """
                MATCH (d:Disease {name: $name})-[:HAS_SYMPTOM]->(s:Symptom)
                RETURN s.name AS name, s.description AS description
                """,
                {"name": disease_name},
            )
            info["symptoms"] = results
        except Exception as e:
            logger.warning(f"查询症状失败: {e}")

        # 查询药物
        try:
            results = await self._neo4j_pool.execute_read(
                """
                MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: $name})
                RETURN d.name AS name, d.dosage_form AS form,
                       d.contraindications AS contraindications
                """,
                {"name": disease_name},
            )
            info["drugs"] = results
        except Exception as e:
            logger.warning(f"查询药物失败: {e}")

        # 查询检查项目
        try:
            results = await self._neo4j_pool.execute_read(
                """
                MATCH (d:Disease {name: $name})-[:REQUIRES_EXAM]->(e:ExamItem)
                RETURN e.name AS name, e.type AS type, e.description AS description
                """,
                {"name": disease_name},
            )
            info["exam_items"] = results
        except Exception as e:
            logger.warning(f"查询检查项目失败: {e}")

        # 查询治疗方案
        try:
            results = await self._neo4j_pool.execute_read(
                """
                MATCH (d:Disease {name: $name})-[:RECOMMENDS_TREATMENT]->(t:Treatment)
                RETURN t.name AS name, t.type AS type, t.description AS description
                """,
                {"name": disease_name},
            )
            info["treatments"] = results
        except Exception as e:
            logger.warning(f"查询治疗方案失败: {e}")

        return info

    async def query_by_entities_parallel(
        self,
        entity_names: List[str],
        max_depth: int = 2,
        limit_per_entity: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        批量实体并行关联查询（用于 Agent 流程中多实体同时查询）

        Args:
            entity_names: 实体名称列表
            max_depth: 查询深度
            limit_per_entity: 每个实体的返回限制

        Returns:
            所有实体的关联信息合并列表
        """
        import asyncio

        tasks = [
            self.query_by_entity(name, max_depth, limit_per_entity)
            for name in entity_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = []
        for name, result in zip(entity_names, results):
            if isinstance(result, Exception):
                logger.warning(f"实体 '{name}' 查询失败: {result}")
                continue
            for item in result:
                item["query_entity"] = name
                merged.append(item)

        return merged

    async def execute_cypher(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行自定义 Cypher 查询

        Args:
            cypher: Cypher 语句
            parameters: 参数

        Returns:
            查询结果
        """
        if not self._initialized:
            await self.initialize()

        try:
            results = await self._neo4j_pool.execute_read(cypher, parameters)
            return results
        except Exception as e:
            logger.error(f"Cypher 查询失败: {e}")
            raise

    async def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not self._initialized:
            await self.initialize()

        stats = {"node_counts": {}, "relationship_counts": {}}

        try:
            # 节点统计
            result = await self._neo4j_pool.execute_read(
                "CALL db.labels() YIELD label "
                "WITH label "
                "CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS count', {}) YIELD value "
                "RETURN label, value.count AS count"
            )
            for row in result:
                stats["node_counts"][row["label"]] = row["count"]
        except Exception:
            # APOC 可能不可用，降级为简单统计
            try:
                for label in ["Disease", "Symptom", "Drug", "ExamItem", "Treatment"]:
                    result = await self._neo4j_pool.execute_read(
                        f"MATCH (n:{label}) RETURN count(n) AS count"
                    )
                    if result:
                        stats["node_counts"][label] = result[0]["count"]
            except Exception as e:
                logger.warning(f"降级统计也失败: {e}")

        return stats
