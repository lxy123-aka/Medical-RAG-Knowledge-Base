"""
Neo4j 医学知识图谱初始化脚本

功能:
1. 创建约束和索引
2. 导入示例医学知识图谱数据
3. 断言校验属性完整性（规则5）
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from core.neo4j_schema import (
    NodeLabels, RelTypes,
    NODE_PROPERTY_SCHEMAS, RELATIONSHIP_PROPERTY_SCHEMAS,
    VALID_TRIPLETS, get_all_node_labels, get_all_rel_types,
)
from services.db.neo4j_pool import get_neo4j_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_constraints():
    """创建唯一性约束和索引"""
    pool = get_neo4j_pool()
    await pool.initialize()

    logger.info("创建 Neo4j 约束和索引...")

    # 为每个节点类型的 name 属性创建唯一性约束
    constraints = [
        f"CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (n:{NodeLabels.DISEASE}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:{NodeLabels.SYMPTOM}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (n:{NodeLabels.DRUG}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT bodypart_name IF NOT EXISTS FOR (n:{NodeLabels.BODY_PART}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT examitem_name IF NOT EXISTS FOR (n:{NodeLabels.EXAM_ITEM}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT dept_name IF NOT EXISTS FOR (n:{NodeLabels.DEPARTMENT}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT doctor_name IF NOT EXISTS FOR (n:{NodeLabels.DOCTOR}) REQUIRE n.name IS UNIQUE",
        f"CREATE CONSTRAINT icd_code IF NOT EXISTS FOR (n:{NodeLabels.ICD_CODE}) REQUIRE n.code IS UNIQUE",
        f"CREATE CONSTRAINT record_id IF NOT EXISTS FOR (n:{NodeLabels.MEDICAL_RECORD}) REQUIRE n.record_id IS UNIQUE",
    ]

    for cypher in constraints:
        try:
            await pool.execute_write(cypher)
            logger.info(f"  ✓ 约束创建成功")
        except Exception as e:
            logger.warning(f"  ⚠ 约束创建跳过: {e}")

    logger.info("约束和索引创建完成")


async def import_sample_data():
    """导入示例医学知识图谱数据"""
    pool = get_neo4j_pool()

    logger.info("导入示例医学数据...")

    # ── 疾病节点 ─────────────────────────────────────────────
    diseases = [
        {"name": "高血压", "icd_code": "I10", "description": "以动脉血压持续升高为特征的慢性病", "category": "心血管"},
        {"name": "2型糖尿病", "icd_code": "E11", "description": "以慢性高血糖为特征的代谢性疾病", "category": "内分泌"},
        {"name": "冠心病", "icd_code": "I25.1", "description": "冠状动脉粥样硬化导致的心肌缺血", "category": "心血管"},
        {"name": "肺炎", "icd_code": "J18", "description": "肺部感染性炎症", "category": "呼吸"},
        {"name": "胃炎", "icd_code": "K29", "description": "胃黏膜炎症", "category": "消化"},
    ]

    for d in diseases:
        await pool.execute_write(
            f"MERGE (n:{NodeLabels.DISEASE} {{name: $name}}) SET n += $props",
            {"name": d["name"], "props": d},
        )
    logger.info(f"  ✓ 导入 {len(diseases)} 个疾病节点")

    # ── 症状节点 ─────────────────────────────────────────────
    symptoms = [
        {"name": "头痛", "description": "头部疼痛感"},
        {"name": "头晕", "description": "头部昏沉感"},
        {"name": "多饮", "description": "饮水量明显增多"},
        {"name": "多尿", "description": "排尿次数和量增多"},
        {"name": "胸闷", "description": "胸部压迫感"},
        {"name": "心悸", "description": "心跳不规则或加速感"},
        {"name": "咳嗽", "description": "反射性呼气动作"},
        {"name": "发热", "description": "体温升高"},
        {"name": "腹痛", "description": "腹部疼痛"},
        {"name": "恶心", "description": "胃部不适感"},
    ]

    for s in symptoms:
        await pool.execute_write(
            f"MERGE (n:{NodeLabels.SYMPTOM} {{name: $name}}) SET n += $props",
            {"name": s["name"], "props": s},
        )
    logger.info(f"  ✓ 导入 {len(symptoms)} 个症状节点")

    # ── 药物节点 ─────────────────────────────────────────────
    drugs = [
        {"name": "硝苯地平", "generic_name": "Nifedipine", "dosage_form": "缓释片", "side_effects": "面部潮红、头痛"},
        {"name": "二甲双胍", "generic_name": "Metformin", "dosage_form": "片剂", "side_effects": "恶心、腹泻"},
        {"name": "阿司匹林", "generic_name": "Aspirin", "dosage_form": "肠溶片", "side_effects": "胃肠道出血"},
        {"name": "阿莫西林", "generic_name": "Amoxicillin", "dosage_form": "胶囊", "side_effects": "过敏反应"},
        {"name": "奥美拉唑", "generic_name": "Omeprazole", "dosage_form": "肠溶胶囊", "side_effects": "头痛、腹泻"},
    ]

    for d in drugs:
        await pool.execute_write(
            f"MERGE (n:{NodeLabels.DRUG} {{name: $name}}) SET n += $props",
            {"name": d["name"], "props": d},
        )
    logger.info(f"  ✓ 导入 {len(drugs)} 个药物节点")

    # ── 身体部位节点 ─────────────────────────────────────────
    body_parts = [
        {"name": "头部", "system": "神经系统"},
        {"name": "胸部", "system": "呼吸系统"},
        {"name": "腹部", "system": "消化系统"},
        {"name": "心脏", "system": "循环系统"},
    ]

    for bp in body_parts:
        await pool.execute_write(
            f"MERGE (n:{NodeLabels.BODY_PART} {{name: $name}}) SET n += $props",
            {"name": bp["name"], "props": bp},
        )
    logger.info(f"  ✓ 导入 {len(body_parts)} 个身体部位节点")

    # ── 关系 ─────────────────────────────────────────────────
    relationships = [
        # 疾病-症状
        ("高血压", "HAS_SYMPTOM", "头痛", {"weight": 0.8, "frequency": 0.6}),
        ("高血压", "HAS_SYMPTOM", "头晕", {"weight": 0.9, "frequency": 0.7}),
        ("2型糖尿病", "HAS_SYMPTOM", "多饮", {"weight": 0.9, "frequency": 0.8}),
        ("2型糖尿病", "HAS_SYMPTOM", "多尿", {"weight": 0.9, "frequency": 0.8}),
        ("冠心病", "HAS_SYMPTOM", "胸闷", {"weight": 0.9, "frequency": 0.8}),
        ("冠心病", "HAS_SYMPTOM", "心悸", {"weight": 0.7, "frequency": 0.5}),
        ("肺炎", "HAS_SYMPTOM", "咳嗽", {"weight": 0.9, "frequency": 0.9}),
        ("肺炎", "HAS_SYMPTOM", "发热", {"weight": 0.8, "frequency": 0.7}),
        ("胃炎", "HAS_SYMPTOM", "腹痛", {"weight": 0.9, "frequency": 0.8}),
        ("胃炎", "HAS_SYMPTOM", "恶心", {"weight": 0.7, "frequency": 0.6}),
        # 药物治疗
        ("硝苯地平", "TREATS", "高血压", {"weight": 0.9, "effectiveness": 0.85}),
        ("二甲双胍", "TREATS", "2型糖尿病", {"weight": 0.9, "effectiveness": 0.80}),
        ("阿司匹林", "TREATS", "冠心病", {"weight": 0.8, "effectiveness": 0.75}),
        ("阿莫西林", "TREATS", "肺炎", {"weight": 0.8, "effectiveness": 0.80}),
        ("奥美拉唑", "TREATS", "胃炎", {"weight": 0.9, "effectiveness": 0.85}),
        # 症状-部位
        ("头痛", "LOCATED_IN", "头部", {"weight": 1.0}),
        ("头晕", "LOCATED_IN", "头部", {"weight": 1.0}),
        ("胸闷", "LOCATED_IN", "胸部", {"weight": 1.0}),
        ("心悸", "LOCATED_IN", "心脏", {"weight": 1.0}),
        ("腹痛", "LOCATED_IN", "腹部", {"weight": 1.0}),
    ]

    for start_name, rel_type, end_name, props in relationships:
        # 确定节点标签
        if rel_type == "TREATS":
            start_label = NodeLabels.DRUG
            end_label = NodeLabels.DISEASE
        elif rel_type == "HAS_SYMPTOM":
            start_label = NodeLabels.DISEASE
            end_label = NodeLabels.SYMPTOM
        elif rel_type == "LOCATED_IN":
            start_label = NodeLabels.SYMPTOM
            end_label = NodeLabels.BODY_PART
        else:
            continue

        await pool.execute_write(
            f"MATCH (a:{start_label} {{name: $start}}), (b:{end_label} {{name: $end}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
            {"start": start_name, "end": end_name, "props": props},
        )

    logger.info(f"  ✓ 导入 {len(relationships)} 条关系")


async def validate_properties():
    """
    断言校验属性完整性（规则5: 防止属性遗漏导致读取全为 None）
    """
    pool = get_neo4j_pool()

    logger.info("执行属性完整性断言校验...")

    # 检查关键关系的 weight 属性不为空
    rel_types_with_weight = [
        RelTypes.HAS_SYMPTOM, RelTypes.TREATS, RelTypes.CAUSES,
        RelTypes.LOCATED_IN, RelTypes.REQUIRES_EXAM,
    ]

    for rel_type in rel_types_with_weight:
        result = await pool.execute_read(
            f"MATCH ()-[r:{rel_type}]->() "
            f"WHERE r.weight IS NULL "
            f"RETURN count(r) AS null_count"
        )
        if result and result[0].get("null_count", 0) > 0:
            logger.error(
                f"  ✗ 关系 {rel_type} 有 {result[0]['null_count']} 条 weight 为 NULL！"
            )
        else:
            logger.info(f"  ✓ 关系 {rel_type} weight 属性完整")


async def main():
    """主入口"""
    logger.info("=" * 50)
    logger.info("  Neo4j 医学知识图谱初始化")
    logger.info("=" * 50)

    try:
        await create_constraints()
        await import_sample_data()
        await validate_properties()

        logger.info("=" * 50)
        logger.info("  初始化完成！")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"初始化失败: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pool = get_neo4j_pool()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
