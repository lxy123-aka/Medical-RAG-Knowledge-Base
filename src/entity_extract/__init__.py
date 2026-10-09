"""
entity_extract - 医学实体抽取模块
功能：医学实体识别、关系抽取、写入Neo4j图谱
"""

from .extractor import MedicalEntityExtractor

__all__ = ["MedicalEntityExtractor"]
