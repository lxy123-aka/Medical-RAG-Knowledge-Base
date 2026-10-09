"""
医学实体抽取器 - LLM驱动的实体与关系抽取

设计说明：
1. 使用 LLM (Qwen) 从医学文本中抽取实体和关系
2. 抽取的实体类型：疾病、症状、药物、检查项目、治疗方案
3. 抽取的关系类型：HAS_SYMPTOM、TREATS、REQUIRES_EXAM 等
4. 抽取结果写入 Neo4j 图谱
5. 支持批量处理文档分块

使用方法：
    extractor = MedicalEntityExtractor()
    await extractor.initialize()
    result = await extractor.extract_from_text("高血压常见症状包括头痛、头晕...")
    # result 包含 entities 和 relations
    await extractor.write_to_graph(result)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.safe_json import extract_json_from_markdown

logger = logging.getLogger(__name__)


# =============================================================================
# 实体/关系类型常量
# =============================================================================

ENTITY_TYPES = {
    "disease": "Disease",        # 疾病
    "symptom": "Symptom",        # 症状
    "drug": "Drug",              # 药物
    "exam_item": "ExamItem",     # 检查项目
    "treatment": "Treatment",    # 治疗方案
    "body_part": "BodyPart",     # 身体部位
}

RELATION_TYPES = {
    "has_symptom": "HAS_SYMPTOM",
    "treats": "TREATS",
    "causes": "CAUSES",
    "requires_exam": "REQUIRES_EXAM",
    "recommends_treatment": "RECOMMENDS_TREATMENT",
    "located_in": "LOCATED_IN",
}


# =============================================================================
# 抽取结果数据结构
# =============================================================================

class ExtractionResult:
    """实体抽取结果"""

    def __init__(self):
        self.entities: List[Dict[str, Any]] = []
        self.relations: List[Dict[str, Any]] = []
        self.source_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "relations": self.relations,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
        }


# =============================================================================
# 医学实体抽取器
# =============================================================================

class MedicalEntityExtractor:
    """
    医学实体抽取器

    使用 LLM 从医学文本中抽取实体和关系：
    - 疾病名称、症状描述、药物名称、检查项目
    - 疾病-症状、药物-疾病 等关联关系
    """

    def __init__(self):
        self._llm_service = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化 LLM 服务"""
        if self._initialized:
            return
        from services.llm.llm_service import get_llm_service
        self._llm_service = get_llm_service()
        if not self._llm_service._initialized:
            await self._llm_service.initialize()
        self._initialized = True
        logger.info("MedicalEntityExtractor 初始化完成")

    async def extract_from_text(
        self,
        text: str,
        source_doc_id: str = "",
    ) -> ExtractionResult:
        """
        从文本中抽取医学实体和关系

        Args:
            text: 输入文本
            source_doc_id: 来源文档ID

        Returns:
            ExtractionResult 包含 entities 和 relations
        """
        if not self._initialized:
            await self.initialize()

        result = ExtractionResult()
        result.source_text = text

        if not text.strip():
            return result

        # 构造抽取 Prompt
        prompt = self._build_extraction_prompt(text)

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            # 调用 LLM 进行结构化抽取
            raw_response = await self._llm_service.chat(messages, temperature=0.1)

            # 解析 JSON 结果
            parsed = extract_json_from_markdown(raw_response)
            if parsed and isinstance(parsed, dict):
                result.entities = self._normalize_entities(
                    parsed.get("entities", []), source_doc_id
                )
                result.relations = self._normalize_relations(
                    parsed.get("relations", [])
                )

            logger.info(
                f"实体抽取完成: {len(result.entities)} 个实体, "
                f"{len(result.relations)} 条关系"
            )

        except Exception as e:
            logger.error(f"实体抽取失败: {e}")

        return result

    async def extract_from_chunks(
        self,
        chunks: list,
        batch_size: int = 5,
    ) -> ExtractionResult:
        """
        从多个文档分块中批量抽取实体和关系

        Args:
            chunks: Chunk 对象列表
            batch_size: 每批处理的分块数

        Returns:
            合并后的抽取结果
        """
        if not self._initialized:
            await self.initialize()

        merged_result = ExtractionResult()
        seen_entities = set()
        seen_relations = set()

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]

            for chunk in batch:
                # 跳过表格类型分块（表格结构化程度高，不适合LLM抽取）
                if getattr(chunk, "chunk_type", "") == "table":
                    continue

                text = chunk.content
                if len(text) < 50:
                    continue

                result = await self.extract_from_text(
                    text, source_doc_id=chunk.doc_id
                )

                # 去重合并
                for entity in result.entities:
                    key = (entity["label"], entity["name"])
                    if key not in seen_entities:
                        seen_entities.add(key)
                        merged_result.entities.append(entity)

                for rel in result.relations:
                    key = (
                        rel["start_label"], rel["start_name"],
                        rel["rel_type"],
                        rel["end_label"], rel["end_name"],
                    )
                    if key not in seen_relations:
                        seen_relations.add(key)
                        merged_result.relations.append(rel)

        logger.info(
            f"批量抽取完成: {len(merged_result.entities)} 个实体, "
            f"{len(merged_result.relations)} 条关系"
        )
        return merged_result

    async def write_to_graph(
        self,
        extraction_result: ExtractionResult,
    ) -> Dict[str, Any]:
        """
        将抽取结果写入 Neo4j 图谱

        Args:
            extraction_result: 抽取结果

        Returns:
            写入统计
        """
        from src.graph_store.store import GraphStore

        store = GraphStore()
        await store.initialize()

        # 写入实体
        entity_stats = await store.upsert_entities(extraction_result.entities)

        # 写入关系
        relation_stats = await store.upsert_relations(extraction_result.relations)

        stats = {
            "entities_upserted": entity_stats.get("upserted", 0),
            "relations_upserted": relation_stats.get("upserted", 0),
        }
        logger.info(f"图谱写入完成: {stats}")
        return stats

    # ── Prompt 构造 ───────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        return """你是一个医学实体识别与关系抽取专家。
从输入的医学文本中识别以下类型的实体和关系。

实体类型：
- disease: 疾病名称（如高血压、糖尿病、冠心病）
- symptom: 症状（如头痛、头晕、恶心、胸闷）
- drug: 药物名称（如阿司匹林、二甲双胍、硝苯地平）
- exam_item: 检查项目（如血常规、CT、心电图、血压测量）
- treatment: 治疗方案（如手术治疗、物理治疗、饮食控制）
- body_part: 身体部位（如心脏、肝脏、肺部）

关系类型：
- has_symptom: 疾病有症状 (disease -> symptom)
- treats: 药物治疗疾病 (drug -> disease)
- causes: 病因导致疾病 (symptom/cause -> disease)
- requires_exam: 疾病需要检查 (disease -> exam_item)
- recommends_treatment: 疾病推荐治疗 (disease -> treatment)
- located_in: 症状位于部位 (symptom -> body_part)

请严格以 JSON 格式输出，不要任何解释。"""

    def _build_extraction_prompt(self, text: str) -> str:
        # 截断过长文本
        max_text_len = 2000
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."

        return f"""请从以下医学文本中抽取实体和关系：

文本：
{text}

请以 JSON 格式输出：
{{
  "entities": [
    {{"type": "disease", "name": "高血压", "properties": {{"description": "..."}}}},
    {{"type": "symptom", "name": "头痛", "properties": {{"description": "..."}}}}
  ],
  "relations": [
    {{"type": "has_symptom", "from": "高血压", "from_type": "disease", "to": "头痛", "to_type": "symptom"}},
    {{"type": "treats", "from": "硝苯地平", "from_type": "drug", "to": "高血压", "to_type": "disease"}}
  ]
}}

注意：
1. 只抽取文本中明确提到的实体和关系，不要推断
2. 实体名称使用标准医学术语
3. 如果没有找到实体或关系，返回空数组"""

    # ── 结果规范化 ────────────────────────────────────────────

    def _normalize_entities(
        self, raw_entities: List[Dict], source_doc_id: str
    ) -> List[Dict[str, Any]]:
        """规范化实体格式"""
        normalized = []
        for entity in raw_entities:
            entity_type = entity.get("type", "")
            label = ENTITY_TYPES.get(entity_type, "")
            if not label:
                continue

            name = entity.get("name", "").strip()
            if not name:
                continue

            properties = entity.get("properties", {})
            if source_doc_id:
                properties["source_doc"] = source_doc_id

            normalized.append({
                "label": label,
                "name": name,
                "properties": properties,
            })

        return normalized

    def _normalize_relations(
        self, raw_relations: List[Dict]
    ) -> List[Dict[str, Any]]:
        """规范化关系格式"""
        normalized = []
        for rel in raw_relations:
            rel_type = rel.get("type", "")
            rel_label = RELATION_TYPES.get(rel_type, "")
            if not rel_label:
                continue

            from_name = rel.get("from", "").strip()
            from_type = rel.get("from_type", "")
            to_name = rel.get("to", "").strip()
            to_type = rel.get("to_type", "")

            from_label = ENTITY_TYPES.get(from_type, "")
            to_label = ENTITY_TYPES.get(to_type, "")

            if not all([from_name, to_name, from_label, to_label]):
                continue

            normalized.append({
                "start_label": from_label,
                "start_name": from_name,
                "rel_type": rel_label,
                "end_label": to_label,
                "end_name": to_name,
                "properties": {"weight": 1.0},
            })

        return normalized

    # ── 查询时实体抽取（Agent 流程用） ────────────────────────

    async def extract_query_entities(self, query: str) -> List[Dict[str, str]]:
        """
        从用户查询中抽取医学实体（用于图谱检索）

        Args:
            query: 用户查询文本

        Returns:
            实体列表 [{"name": "高血压", "type": "disease"}, ...]
        """
        if not self._initialized:
            await self.initialize()

        prompt = f"""从以下用户问题中提取医学实体：

问题：{query}

以 JSON 数组格式输出：
[{{"name": "实体名", "type": "disease/symptom/drug/exam_item"}}]

如果没有医学实体，返回空数组 []。"""

        messages = [
            {"role": "system", "content": "你是医学实体识别专家，只输出 JSON 数组。"},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._llm_service.chat(messages, temperature=0.1)
            parsed = extract_json_from_markdown(raw)
            if isinstance(parsed, list):
                entities = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        entities.append({
                            "name": item["name"],
                            "type": item.get("type", "disease"),
                        })
                return entities
        except Exception as e:
            logger.warning(f"查询实体抽取失败: {e}")

        return []
