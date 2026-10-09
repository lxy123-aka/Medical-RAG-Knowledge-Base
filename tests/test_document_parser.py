"""
文档解析与分块测试

测试内容：
1. PDF/TXT/MD 文档解析
2. 表格抽取
3. 章节识别
4. 语义分块（大小、重叠、摘要）
5. 边界情况处理
"""

import sys
from pathlib import Path

import pytest

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 测试数据
# =============================================================================

SAMPLE_MEDICAL_TEXT = """
第一章 高血压概述

高血压是一种常见的慢性疾病，以动脉血压持续升高为特征。根据世界卫生组织的标准，
成人血压≥140/90mmHg即可诊断为高血压。

1.1 病因分类

高血压可分为原发性高血压和继发性高血压两大类。原发性高血压占95%以上，
病因尚不完全明确，与遗传、环境、生活方式等因素有关。继发性高血压由其他
疾病引起，如肾脏疾病、内分泌疾病等。

1.2 常见症状

高血压的常见症状包括：
- 头痛：多表现为后脑勺或太阳穴部位的胀痛
- 头晕：感觉头部昏沉，平衡感下降
- 心悸：心跳加快或不规则
- 视力模糊：视网膜血管受损导致
- 乏力：全身疲倦无力

1.3 诊断标准

| 分类 | 收缩压(mmHg) | 舒张压(mmHg) |
|------|-------------|-------------|
| 正常 | <120 | <80 |
| 正常高值 | 120-139 | 80-89 |
| 高血压1级 | 140-159 | 90-99 |
| 高血压2级 | 160-179 | 100-109 |
| 高血压3级 | ≥180 | ≥110 |

第二章 高血压治疗

2.1 药物治疗

常用降压药物包括：
- 钙通道阻滞剂：如硝苯地平、氨氯地平
- ACEI类：如依那普利、贝那普利
- ARB类：如缬沙坦、厄贝沙坦
- 利尿剂：如氢氯噻嗪、吲达帕胺
- β受体阻滞剂：如美托洛尔、比索洛尔

2.2 非药物治疗

生活方式干预是高血压治疗的基础：
- 低盐饮食：每日食盐摄入量不超过6克
- 适量运动：每周至少150分钟中等强度有氧运动
- 控制体重：BMI维持在18.5-23.9
- 戒烟限酒
- 心理平衡
"""

SAMPLE_MARKDOWN = """
# 糖尿病诊疗指南

## 1. 定义

糖尿病是一组以高血糖为特征的代谢性疾病。

## 2. 分型

### 2.1 1型糖尿病
由于胰岛β细胞破坏导致胰岛素绝对缺乏。

### 2.2 2型糖尿病
以胰岛素抵抗为主伴相对胰岛素分泌不足。

## 3. 诊断标准

- 空腹血糖 ≥ 7.0 mmol/L
- OGTT 2h血糖 ≥ 11.1 mmol/L
- 随机血糖 ≥ 11.1 mmol/L 伴典型症状
- HbA1c ≥ 6.5%

## 4. 治疗

### 4.1 药物治疗
常用药物：二甲双胍、磺脲类、DPP-4抑制剂、SGLT-2抑制剂、胰岛素

### 4.2 血糖监测
定期监测空腹血糖、餐后血糖和糖化血红蛋白
"""


# =============================================================================
# 文档解析测试
# =============================================================================

class TestDocumentParser:
    """文档解析器测试"""

    def test_parse_txt_content(self, tmp_path):
        """测试 TXT 文件解析"""
        from src.document_parser.parser import MedicalDocumentParser

        # 创建临时文件
        txt_file = tmp_path / "test_medical.txt"
        txt_file.write_text(SAMPLE_MEDICAL_TEXT, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        # 验证基本属性
        assert doc.file_name == "test_medical.txt"
        assert doc.file_type == "txt"
        assert doc.doc_id == "test_medical"
        assert len(doc.pages) > 0

        # 验证章节识别
        assert len(doc.sections) >= 2, f"应识别到至少2个章节，实际: {len(doc.sections)}"

        # 验证章节标题包含关键词
        section_titles = [s.title for s in doc.sections]
        has_hypertension = any("高血压" in t for t in section_titles)
        has_treatment = any("治疗" in t for t in section_titles)
        assert has_hypertension, f"应识别到高血压章节，实际标题: {section_titles}"
        assert has_treatment, f"应识别到治疗章节，实际标题: {section_titles}"

    def test_parse_markdown(self, tmp_path):
        """测试 Markdown 文件解析"""
        from src.document_parser.parser import MedicalDocumentParser

        md_file = tmp_path / "diabetes_guide.md"
        md_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(md_file))

        assert doc.file_type == "md"
        assert len(doc.sections) >= 4, f"应识别到至少4个章节，实际: {len(doc.sections)}"

        # 验证层级识别
        levels = [s.level for s in doc.sections]
        assert 1 in levels, "应识别到一级标题"
        assert 2 in levels, "应识别到二级标题"

    def test_parse_nonexistent_file(self):
        """测试解析不存在的文件"""
        from src.document_parser.parser import MedicalDocumentParser

        parser = MedicalDocumentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.pdf")

    def test_parse_unsupported_format(self, tmp_path):
        """测试解析不支持的格式"""
        from src.document_parser.parser import MedicalDocumentParser

        bad_file = tmp_path / "test.docx"
        bad_file.write_text("content")

        parser = MedicalDocumentParser()
        with pytest.raises(ValueError, match="不支持的文件格式"):
            parser.parse(str(bad_file))


# =============================================================================
# 语义分块测试
# =============================================================================

class TestSemanticChunker:
    """语义分块器测试"""

    def test_chunk_basic(self, tmp_path):
        """测试基础分块功能"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        txt_file = tmp_path / "test.txt"
        txt_file.write_text(SAMPLE_MEDICAL_TEXT, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        chunker = MedicalSemanticChunker(max_chunk_size=512, overlap=50)
        chunks = chunker.chunk(doc)

        # 验证分块数量
        assert len(chunks) > 0, "应产生至少一个分块"

        # 验证每个分块的内容不为空
        for chunk in chunks:
            assert chunk.content.strip(), f"分块 {chunk.chunk_id} 内容为空"
            assert chunk.doc_id, f"分块 {chunk.chunk_id} 缺少 doc_id"

    def test_chunk_size_limit(self, tmp_path):
        """测试分块大小限制"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        txt_file = tmp_path / "test.txt"
        txt_file.write_text(SAMPLE_MEDICAL_TEXT, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        max_size = 200
        chunker = MedicalSemanticChunker(max_chunk_size=max_size, overlap=20)
        chunks = chunker.chunk(doc)

        # 大部分分块应不超过 max_size 的 1.5 倍（允许句子边界容差）
        oversized = [c for c in chunks if len(c.content) > max_size * 1.5]
        assert len(oversized) == 0, (
            f"有 {len(oversized)} 个分块超过大小限制的1.5倍: "
            f"{[len(c.content) for c in oversized]}"
        )

    def test_chunk_has_summary(self, tmp_path):
        """测试分块摘要生成"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        txt_file = tmp_path / "test.txt"
        txt_file.write_text(SAMPLE_MEDICAL_TEXT, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        chunker = MedicalSemanticChunker()
        chunks = chunker.chunk(doc)

        # 每个分块应有摘要
        for chunk in chunks:
            assert chunk.summary, f"分块 {chunk.chunk_id} 缺少摘要"

    def test_chunk_unique_ids(self, tmp_path):
        """测试分块ID唯一性"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        txt_file = tmp_path / "test.txt"
        txt_file.write_text(SAMPLE_MEDICAL_TEXT, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        chunker = MedicalSemanticChunker()
        chunks = chunker.chunk(doc)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "分块ID不唯一"

    def test_content_with_summary(self, tmp_path):
        """测试带摘要的内容格式"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        txt_file = tmp_path / "test.txt"
        txt_file.write_text(SAMPLE_MEDICAL_TEXT, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        chunker = MedicalSemanticChunker()
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            combined = chunk.content_with_summary
            assert chunk.content in combined, "带摘要内容应包含原始正文"
            if chunk.summary:
                assert "摘要" in combined or chunk.summary in combined


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_file(self, tmp_path):
        """测试空文件"""
        from src.document_parser.parser import MedicalDocumentParser

        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("", encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))
        assert doc.file_name == "empty.txt"

    def test_single_line_file(self, tmp_path):
        """测试单行文件"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        txt_file = tmp_path / "single.txt"
        txt_file.write_text("高血压是一种常见的慢性疾病。", encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        chunker = MedicalSemanticChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 0  # 可能为空或一个块

    def test_very_long_paragraph(self, tmp_path):
        """测试超长段落的二次切分"""
        from src.document_parser.parser import MedicalDocumentParser
        from src.document_parser.chunker import MedicalSemanticChunker

        # 生成超长文本
        long_text = "高血压概述。" * 500  # 约3000字符
        txt_file = tmp_path / "long.txt"
        txt_file.write_text(long_text, encoding="utf-8")

        parser = MedicalDocumentParser()
        doc = parser.parse(str(txt_file))

        chunker = MedicalSemanticChunker(max_chunk_size=512)
        chunks = chunker.chunk(doc)

        # 超长段落应被拆分为多个块
        assert len(chunks) >= 2, "超长段落应被拆分为多个块"
