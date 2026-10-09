"""
医学文档解析器 - PDF解析、表格抽取

设计说明：
1. 使用 PyMuPDF (fitz) 解析 PDF，支持文本和表格抽取
2. 按章节/段落结构化输出，保留文档层级信息
3. 表格单独抽取为结构化数据，避免混入正文分块
4. 支持 .pdf / .md / .txt 三种格式

使用方法：
    parser = MedicalDocumentParser()
    document = parser.parse("path/to/file.pdf")
    # document 包含 pages、sections、tables 等结构化信息
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class TableData:
    """表格数据"""
    page_num: int                          # 所在页码
    headers: List[str] = field(default_factory=list)   # 表头
    rows: List[List[str]] = field(default_factory=list) # 行数据
    caption: str = ""                      # 表格标题
    raw_text: str = ""                     # 原始文本表示

    def to_markdown(self) -> str:
        """转换为 Markdown 表格格式"""
        if not self.headers and not self.rows:
            return ""
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**")
        # 表头
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        # 数据行
        for row in self.rows:
            # 补齐列数
            while len(row) < len(self.headers):
                row.append("")
            lines.append("| " + " | ".join(row[:len(self.headers)]) + " |")
        return "\n".join(lines)


@dataclass
class Section:
    """文档章节"""
    title: str                             # 章节标题
    level: int = 1                         # 层级 (1=一级标题, 2=二级标题...)
    content: str = ""                      # 章节正文内容
    page_start: int = 0                    # 起始页码
    page_end: int = 0                      # 结束页码
    tables: List[TableData] = field(default_factory=list)  # 章节内表格


@dataclass
class Document:
    """解析后的文档结构"""
    doc_id: str                            # 文档唯一标识
    file_path: str                         # 原始文件路径
    file_name: str                         # 文件名
    file_type: str                         # 文件类型 (pdf/md/txt)
    title: str = ""                        # 文档标题
    pages: List[str] = field(default_factory=list)       # 各页原始文本
    sections: List[Section] = field(default_factory=list) # 章节结构
    tables: List[TableData] = field(default_factory=list) # 全部表格
    metadata: Dict[str, Any] = field(default_factory=dict) # 元数据

    @property
    def full_text(self) -> str:
        """获取全文文本"""
        if self.sections:
            return "\n\n".join(
                f"{'#' * s.level} {s.title}\n{s.content}" for s in self.sections
            )
        return "\n\n".join(self.pages)


# =============================================================================
# 文档解析器
# =============================================================================

class MedicalDocumentParser:
    """
    医学文档解析器

    支持格式: PDF / Markdown / TXT
    功能:
    - PDF 文本和表格抽取
    - 章节结构化识别
    - 表格独立抽取为结构化数据
    """

    def __init__(self):
        self._fitz = None  # PyMuPDF 延迟导入

    def parse(self, file_path: str, doc_id: Optional[str] = None) -> Document:
        """
        解析文档

        Args:
            file_path: 文件路径
            doc_id: 文档ID，不指定则从文件名生成

        Returns:
            Document 对象
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_type = path.suffix.lower().lstrip(".")
        file_name = path.name

        if doc_id is None:
            doc_id = path.stem

        logger.info(f"开始解析文档: {file_name} (类型: {file_type})")

        if file_type == "pdf":
            return self._parse_pdf(file_path, doc_id, file_name, file_type)
        elif file_type in ("md", "markdown"):
            return self._parse_markdown(file_path, doc_id, file_name, file_type)
        elif file_type == "txt":
            return self._parse_txt(file_path, doc_id, file_name, file_type)
        else:
            raise ValueError(f"不支持的文件格式: {file_type}，仅支持 pdf/md/txt")

    # ── PDF 解析 ──────────────────────────────────────────────

    def _parse_pdf(
        self, file_path: str, doc_id: str, file_name: str, file_type: str
    ) -> Document:
        """解析 PDF 文档"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "请安装 PyMuPDF: pip install pymupdf"
            )

        doc = fitz.open(file_path)
        pages_text = []
        all_tables: List[TableData] = []
        all_sections: List[Section] = []

        try:
            # 1. 逐页提取文本
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                pages_text.append(text)

            # 2. 抽取表格（逐页扫描）
            for page_num in range(len(doc)):
                page = doc[page_num]
                tables = self._extract_tables_from_page(page, page_num)
                all_tables.extend(tables)

            # 3. 识别章节结构
            full_text = "\n".join(pages_text)
            all_sections = self._identify_sections(full_text, pages_text)

            logger.info(
                f"PDF 解析完成: {len(pages_text)} 页, "
                f"{len(all_sections)} 章节, {len(all_tables)} 表格"
            )

        finally:
            doc.close()

        return Document(
            doc_id=doc_id,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            title=self._extract_title(pages_text),
            pages=pages_text,
            sections=all_sections,
            tables=all_tables,
            metadata={"total_pages": len(pages_text)},
        )

    def _extract_tables_from_page(
        self, page: Any, page_num: int
    ) -> List[TableData]:
        """
        从单页 PDF 中抽取表格

        使用 PyMuPDF 内置的表格检测功能（pymupdf4llm 或 find_tables）
        如果版本不支持，降级为基于文本模式的表格识别
        """
        tables: List[TableData] = []

        # 尝试使用新版 pymupdf 的 find_tables 功能
        try:
            tab_finder = page.find_tables()
            if tab_finder and tab_finder.tables:
                for tab in tab_finder.tables:
                    # 提取表头和数据行
                    data = tab.extract()
                    if data and len(data) > 0:
                        headers = [str(cell) if cell else "" for cell in data[0]]
                        rows = [
                            [str(cell) if cell else "" for cell in row]
                            for row in data[1:]
                        ]
                        tables.append(TableData(
                            page_num=page_num + 1,
                            headers=headers,
                            rows=rows,
                            raw_text=tab.to_markdown_string() if hasattr(tab, "to_markdown_string") else "",
                        ))
                return tables
        except (AttributeError, Exception):
            pass

        # 降级方案：基于文本模式识别简单表格
        text = page.get_text("text")
        table_blocks = self._detect_table_patterns(text)
        for block in table_blocks:
            tables.append(TableData(
                page_num=page_num + 1,
                raw_text=block,
            ))

        return tables

    def _detect_table_patterns(self, text: str) -> List[str]:
        """
        基于文本模式检测表格区域（降级方案）

        识别特征：
        - 包含制表符分隔的多行文本
        - 包含 "|" 分隔符的文本
        """
        tables = []
        lines = text.split("\n")
        current_table_lines: List[str] = []

        for line in lines:
            # 检测是否为表格行（包含多个制表符或 | 分隔符）
            is_table_line = (
                line.count("\t") >= 2 or
                line.count("|") >= 2 or
                (len(line.split()) >= 4 and re.match(r"^\s*\S+\s+\S+\s+\S+", line))
            )
            if is_table_line:
                current_table_lines.append(line)
            else:
                if len(current_table_lines) >= 2:
                    tables.append("\n".join(current_table_lines))
                current_table_lines = []

        if len(current_table_lines) >= 2:
            tables.append("\n".join(current_table_lines))

        return tables

    # ── 章节识别 ──────────────────────────────────────────────

    def _identify_sections(
        self, full_text: str, pages_text: List[str]
    ) -> List[Section]:
        """
        识别文档章节结构

        策略：
        1. 匹配常见医学文档标题模式（第X章、第X节、X.X.X 编号标题等）
        2. 按标题切分章节
        3. 将表格关联到对应章节
        """
        sections: List[Section] = []

        # 医学文档常见标题模式
        title_patterns = [
            # 第X章 / 第X节
            r"^(第[一二三四五六七八九十百千\d]+[章节篇部])\s*(.*)",
            # 编号标题: 1. / 1.1 / 1.1.1
            r"^(\d+(?:\.\d+){0,3})\s+(.+)",
            # 大写标题: 一、/ 二、
            r"^([一二三四五六七八九十]+)[、.]\s*(.+)",
        ]

        lines = full_text.split("\n")
        current_section: Optional[Section] = None
        content_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                content_lines.append("")
                continue

            # 检查是否为标题行
            is_title = False
            title_text = ""
            title_level = 1

            for pattern in title_patterns:
                match = re.match(pattern, stripped)
                if match:
                    is_title = True
                    # 判断层级
                    if re.match(r"^\d+\.\d+\.\d+", stripped):
                        title_level = 3
                    elif re.match(r"^\d+\.\d+", stripped):
                        title_level = 2
                    elif re.match(r"^\d+[.、]", stripped):
                        title_level = 1
                    elif re.match(r"^第.+[章节]", stripped):
                        title_level = 1
                    elif re.match(r"^[一二三四五六七八九十]+[、.]", stripped):
                        title_level = 1

                    title_text = stripped
                    break

            # 短行且全为大写/中文数字也可能是标题
            if (
                not is_title
                and len(stripped) < 40
                and not stripped.endswith(("。", "；", "，", "、"))
                and re.match(r"^[\u4e00-\u9fa5A-Za-z\s\d]+$", stripped)
            ):
                # 可能是简短标题，但需要前后有空行佐证
                line_idx = lines.index(line) if line in lines else -1
                if line_idx > 0 and lines[line_idx - 1].strip() == "":
                    is_title = True
                    title_text = stripped
                    title_level = 2

            if is_title:
                # 保存前一章节
                if current_section is not None:
                    current_section.content = "\n".join(content_lines).strip()
                    sections.append(current_section)

                current_section = Section(
                    title=title_text,
                    level=title_level,
                    page_start=0,
                )
                content_lines = []
            else:
                content_lines.append(line)

        # 保存最后一个章节
        if current_section is not None:
            current_section.content = "\n".join(content_lines).strip()
            sections.append(current_section)

        # 如果没有识别到任何章节，创建单章节
        if not sections and full_text.strip():
            sections.append(Section(
                title="全文",
                level=1,
                content=full_text.strip(),
            ))

        return sections

    # ── Markdown 解析 ─────────────────────────────────────────

    def _parse_markdown(
        self, file_path: str, doc_id: str, file_name: str, file_type: str
    ) -> Document:
        """解析 Markdown 文档"""
        text = Path(file_path).read_text(encoding="utf-8")
        sections: List[Section] = []

        # 按 Markdown 标题拆分
        current_section: Optional[Section] = None
        content_lines: List[str] = []

        for line in text.split("\n"):
            header_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if header_match:
                if current_section is not None:
                    current_section.content = "\n".join(content_lines).strip()
                    sections.append(current_section)
                level = len(header_match.group(1))
                current_section = Section(
                    title=header_match.group(2).strip(),
                    level=level,
                )
                content_lines = []
            else:
                content_lines.append(line)

        if current_section is not None:
            current_section.content = "\n".join(content_lines).strip()
            sections.append(current_section)

        if not sections:
            sections.append(Section(title="全文", level=1, content=text.strip()))

        return Document(
            doc_id=doc_id,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            title=sections[0].title if sections else doc_id,
            pages=[text],
            sections=sections,
            metadata={},
        )

    # ── TXT 解析 ──────────────────────────────────────────────

    def _parse_txt(
        self, file_path: str, doc_id: str, file_name: str, file_type: str
    ) -> Document:
        """解析纯文本文件"""
        text = Path(file_path).read_text(encoding="utf-8")
        sections = self._identify_sections(text, [text])

        return Document(
            doc_id=doc_id,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            title=self._extract_title([text]),
            pages=[text],
            sections=sections,
            metadata={},
        )

    # ── 工具方法 ──────────────────────────────────────────────

    def _extract_title(self, pages_text: List[str]) -> str:
        """从首页文本中提取文档标题"""
        if not pages_text:
            return ""
        first_page = pages_text[0]
        lines = first_page.strip().split("\n")
        # 取第一个非空短行作为标题
        for line in lines[:10]:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                return stripped
        return ""
