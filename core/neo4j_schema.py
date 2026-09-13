"""
Neo4j Schema 共享常量文件（规则1: 属性名全局强一致）

所有节点类型、关系类型、属性名在此统一定义。
初始化脚本和所有查询代码从同一处引用，启动时自动校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


# =============================================================================
# 节点类型定义
# =============================================================================

class NodeLabels:
    """Neo4j 节点标签（Label）常量"""
    DISEASE = "Disease"              # 疾病
    SYMPTOM = "Symptom"              # 症状
    DRUG = "Drug"                    # 药物
    BODY_PART = "BodyPart"           # 身体部位/器官
    EXAM_ITEM = "ExamItem"           # 检查项目
    EXAM_RESULT = "ExamResult"       # 检查结果
    TREATMENT = "Treatment"          # 治疗方案
    MEDICAL_RECORD = "MedicalRecord" # 病历
    DEPARTMENT = "Department"        # 科室
    DOCTOR = "Doctor"                # 医生
    ICD_CODE = "ICDCode"             # ICD 编码


# =============================================================================
# 关系类型定义
# =============================================================================

class RelTypes:
    """Neo4j 关系类型常量"""
    HAS_SYMPTOM = "HAS_SYMPTOM"              # 疾病-症状
    TREATS = "TREATS"                        # 药物治疗疾病
    CAUSES = "CAUSES"                        # 病因导致疾病
    LOCATED_IN = "LOCATED_IN"                # 症状位于身体部位
    REQUIRES_EXAM = "REQUIRES_EXAM"          # 疾病需要检查
    SHOWS_RESULT = "SHOWS_RESULT"            # 检查显示结果
    RECOMMENDS_TREATMENT = "RECOMMENDS_TREATMENT"  # 推荐治疗
    RECORDED_IN = "RECORDED_IN"              # 病历记录
    BELONGS_TO_DEPT = "BELONGS_TO_DEPT"      # 医生属于科室
    DIAGNOSED_BY = "DIAGNOSED_BY"            # 医生诊断
    CODED_AS = "CODED_AS"                    # 疾病编码为 ICD


# =============================================================================
# 属性 Schema 定义（规则5: 关系类型定义完整属性 Schema）
# =============================================================================

@dataclass
class PropertySchema:
    """属性 Schema 定义"""
    name: str
    type: str  # "str", "int", "float", "bool", "list"
    required: bool = True
    description: str = ""


# ── 节点属性 Schema ──────────────────────────────────────────

NODE_PROPERTY_SCHEMAS: Dict[str, List[PropertySchema]] = {
    NodeLabels.DISEASE: [
        PropertySchema("name", "str", True, "疾病名称"),
        PropertySchema("icd_code", "str", False, "ICD 编码"),
        PropertySchema("description", "str", False, "疾病描述"),
        PropertySchema("prevalence", "str", False, "患病率"),
        PropertySchema("category", "str", False, "疾病分类"),
    ],
    NodeLabels.SYMPTOM: [
        PropertySchema("name", "str", True, "症状名称"),
        PropertySchema("description", "str", False, "症状描述"),
        PropertySchema("severity_levels", "str", False, "严重程度分级"),
    ],
    NodeLabels.DRUG: [
        PropertySchema("name", "str", True, "药物名称"),
        PropertySchema("generic_name", "str", False, "通用名"),
        PropertySchema("dosage_form", "str", False, "剂型"),
        PropertySchema("contraindications", "str", False, "禁忌症"),
        PropertySchema("side_effects", "str", False, "副作用"),
    ],
    NodeLabels.BODY_PART: [
        PropertySchema("name", "str", True, "部位名称"),
        PropertySchema("system", "str", False, "所属系统"),
    ],
    NodeLabels.EXAM_ITEM: [
        PropertySchema("name", "str", True, "检查项目名称"),
        PropertySchema("type", "str", False, "检查类型（影像/化验/...）"),
        PropertySchema("description", "str", False, "检查描述"),
    ],
    NodeLabels.EXAM_RESULT: [
        PropertySchema("value", "str", True, "检查结果值"),
        PropertySchema("unit", "str", False, "单位"),
        PropertySchema("reference_range", "str", False, "参考范围"),
        PropertySchema("is_abnormal", "bool", False, "是否异常"),
    ],
    NodeLabels.TREATMENT: [
        PropertySchema("name", "str", True, "治疗方案名称"),
        PropertySchema("type", "str", False, "治疗类型"),
        PropertySchema("description", "str", False, "治疗描述"),
    ],
    NodeLabels.MEDICAL_RECORD: [
        PropertySchema("record_id", "str", True, "病历编号"),
        PropertySchema("patient_id", "str", True, "患者编号"),
        PropertySchema("admission_date", "str", False, "入院日期"),
        PropertySchema("discharge_date", "str", False, "出院日期"),
        PropertySchema("chief_complaint", "str", False, "主诉"),
        PropertySchema("permission_level", "int", False, "权限等级"),
    ],
    NodeLabels.DEPARTMENT: [
        PropertySchema("name", "str", True, "科室名称"),
        PropertySchema("floor", "str", False, "楼层"),
    ],
    NodeLabels.DOCTOR: [
        PropertySchema("name", "str", True, "医生姓名"),
        PropertySchema("title", "str", False, "职称"),
        PropertySchema("specialty", "str", False, "专长"),
    ],
    NodeLabels.ICD_CODE: [
        PropertySchema("code", "str", True, "ICD 编码"),
        PropertySchema("name", "str", True, "编码对应名称"),
        PropertySchema("chapter", "str", False, "章节"),
    ],
}


# ── 关系属性 Schema ──────────────────────────────────────────

RELATIONSHIP_PROPERTY_SCHEMAS: Dict[str, List[PropertySchema]] = {
    RelTypes.HAS_SYMPTOM: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("frequency", "float", False, "发生频率"),
        PropertySchema("description", "str", False, "描述"),
    ],
    RelTypes.TREATS: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("effectiveness", "float", False, "有效率"),
        PropertySchema("dosage", "str", False, "推荐剂量"),
    ],
    RelTypes.CAUSES: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("probability", "float", False, "致病概率"),
    ],
    RelTypes.LOCATED_IN: [
        PropertySchema("weight", "float", True, "关联权重"),
    ],
    RelTypes.REQUIRES_EXAM: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("priority", "int", False, "优先级"),
    ],
    RelTypes.SHOWS_RESULT: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("indicator", "str", False, "指示意义"),
    ],
    RelTypes.RECOMMENDS_TREATMENT: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("evidence_level", "str", False, "证据等级"),
    ],
    RelTypes.RECORDED_IN: [
        PropertySchema("weight", "float", True, "关联权重"),
    ],
    RelTypes.BELONGS_TO_DEPT: [
        PropertySchema("weight", "float", True, "关联权重"),
    ],
    RelTypes.DIAGNOSED_BY: [
        PropertySchema("weight", "float", True, "关联权重"),
        PropertySchema("diagnosis_date", "str", False, "诊断日期"),
    ],
    RelTypes.CODED_AS: [
        PropertySchema("weight", "float", True, "关联权重"),
    ],
}


# =============================================================================
# 节点-关系-节点 三元组约束（用于 Schema 校验）
# =============================================================================

@dataclass
class TripletConstraint:
    """节点-关系-节点 三元组约束"""
    start_label: str
    rel_type: str
    end_label: str


VALID_TRIPLETS: List[TripletConstraint] = [
    TripletConstraint(NodeLabels.DISEASE, RelTypes.HAS_SYMPTOM, NodeLabels.SYMPTOM),
    TripletConstraint(NodeLabels.DRUG, RelTypes.TREATS, NodeLabels.DISEASE),
    TripletConstraint(NodeLabels.SYMPTOM, RelTypes.CAUSES, NodeLabels.DISEASE),
    TripletConstraint(NodeLabels.SYMPTOM, RelTypes.LOCATED_IN, NodeLabels.BODY_PART),
    TripletConstraint(NodeLabels.DISEASE, RelTypes.REQUIRES_EXAM, NodeLabels.EXAM_ITEM),
    TripletConstraint(NodeLabels.EXAM_ITEM, RelTypes.SHOWS_RESULT, NodeLabels.EXAM_RESULT),
    TripletConstraint(NodeLabels.DISEASE, RelTypes.RECOMMENDS_TREATMENT, NodeLabels.TREATMENT),
    TripletConstraint(NodeLabels.MEDICAL_RECORD, RelTypes.RECORDED_IN, NodeLabels.DISEASE),
    TripletConstraint(NodeLabels.DOCTOR, RelTypes.BELONGS_TO_DEPT, NodeLabels.DEPARTMENT),
    TripletConstraint(NodeLabels.DOCTOR, RelTypes.DIAGNOSED_BY, NodeLabels.DISEASE),
    TripletConstraint(NodeLabels.DISEASE, RelTypes.CODED_AS, NodeLabels.ICD_CODE),
]


# =============================================================================
# Schema 工具函数
# =============================================================================

def get_all_node_labels() -> List[str]:
    """获取所有节点标签"""
    return [
        NodeLabels.DISEASE, NodeLabels.SYMPTOM, NodeLabels.DRUG,
        NodeLabels.BODY_PART, NodeLabels.EXAM_ITEM, NodeLabels.EXAM_RESULT,
        NodeLabels.TREATMENT, NodeLabels.MEDICAL_RECORD, NodeLabels.DEPARTMENT,
        NodeLabels.DOCTOR, NodeLabels.ICD_CODE,
    ]


def get_all_rel_types() -> List[str]:
    """获取所有关系类型"""
    return [
        RelTypes.HAS_SYMPTOM, RelTypes.TREATS, RelTypes.CAUSES,
        RelTypes.LOCATED_IN, RelTypes.REQUIRES_EXAM, RelTypes.SHOWS_RESULT,
        RelTypes.RECOMMENDS_TREATMENT, RelTypes.RECORDED_IN,
        RelTypes.BELONGS_TO_DEPT, RelTypes.DIAGNOSED_BY, RelTypes.CODED_AS,
    ]


def get_all_property_names() -> Set[str]:
    """获取所有属性名（用于 Cypher 语句校验）"""
    names: Set[str] = set()
    for props in NODE_PROPERTY_SCHEMAS.values():
        for p in props:
            names.add(p.name)
    for props in RELATIONSHIP_PROPERTY_SCHEMAS.values():
        for p in props:
            names.add(p.name)
    return names


def get_node_properties(label: str) -> List[PropertySchema]:
    """获取指定节点类型的属性列表"""
    return NODE_PROPERTY_SCHEMAS.get(label, [])


def get_rel_properties(rel_type: str) -> List[PropertySchema]:
    """获取指定关系类型的属性列表"""
    return RELATIONSHIP_PROPERTY_SCHEMAS.get(rel_type, [])


def build_schema_summary() -> str:
    """
    构建 Schema 摘要文本，用于注入 NL2Cypher Prompt（规则15）
    """
    lines = ["=== Neo4j 图谱 Schema ===", ""]

    lines.append("【节点类型及属性】")
    for label in get_all_node_labels():
        props = get_node_properties(label)
        prop_strs = [f"{p.name}({p.type})" for p in props]
        required_props = [p.name for p in props if p.required]
        lines.append(f"  ({label}) -> 属性: {', '.join(prop_strs)}")
        lines.append(f"    必填: {', '.join(required_props)}")

    lines.append("")
    lines.append("【关系类型及属性】")
    for rel_type in get_all_rel_types():
        props = get_rel_properties(rel_type)
        prop_strs = [f"{p.name}({p.type})" for p in props]
        lines.append(f"  [:{rel_type}] -> 属性: {', '.join(prop_strs)}")

    lines.append("")
    lines.append("【合法三元组模式】")
    for t in VALID_TRIPLETS:
        lines.append(f"  ({t.start_label})-[:{t.rel_type}]->({t.end_label})")

    return "\n".join(lines)
