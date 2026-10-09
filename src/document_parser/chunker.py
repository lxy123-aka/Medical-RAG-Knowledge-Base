"""
医疗语义分块器 - 基于文档结构的智能分块

设计说明：
1. 不使用固定长度粗暴切分，优先按章节/段落语义边界切分
2. 表格、诊疗方案按章节独立成块，保持完整性
3. 每个块附带摘要信息，存入向量库提升召回质量
4. 超长段落递归二次切分，保证块大小可控
5. 块之间保留少量重叠，避免语义截断

使用方法：
    chunker = MedicalSemanticChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk(document)
    # 每个 chunk 包含 content, summary, metadata 等信息
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .parser import Document, Section, TableData

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Chunk:
    """分块结果"""
    chunk_id: str                          # 分块唯一ID
    doc_id: str                            # 所属文档ID
    chunk_index: int                       # 分块序号
    content: str                           # 分块正文
    summary: str = ""                      # 分块摘要（用于提升召回质量）
    section_title: str = ""                # 所属章节标题
    chunk_type: str = "text"               # 类型: text / table / list
    source_file: str = ""                  # 来源文件名
    page_num: int = 0                      # 起始页码
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_with_summary(self) -> str:
        """带摘要的文本内容（存入向量库时使用，提升语义匹配质量）"""
        if self.summary:
            return f"[摘要] {self.summary}\n\n[正文] {self.content}"
        return self.content


# =============================================================================
# 医疗语义分块器
# =============================================================================

class MedicalSemanticChunker:
    """
    医疗文档语义分块器

    分块策略（优先级从高到低）：
    1. 按章节标题切分 —— 每个章节独立成块
    2. 表格独立成块 —— 表格内容保持完整，不拆分
    3. 段落自然边界 —— 按空行/段落分隔符切分
    4. 超长段落二次切分 —— 在句子边界处切分，避免破坏语义
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        """
        Args:
            max_chunk_size: 单块最大字符数（中文字符）
            overlap: 块间重叠字符数
            min_chunk_size: 最小块大小，低于此值合并到相邻块
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> List[Chunk]:
        """
        对文档进行语义分块

        Args:
            document: 解析后的文档对象

        Returns:
            Chunk 列表
        """
        chunks: List[Chunk] = []
        chunk_index = 0

        logger.info(
            f"开始分块: 文档={document.file_name}, "
            f"章节数={len(document.sections)}, 表格数={len(document.tables)}"
        )

        # 1. 按章节分块
        for section in document.sections:
            # 1a. 章节内的表格独立成块
            for table in section.tables:
                table_text = table.to_markdown()
                if table_text.strip():
                    chunk = Chunk(
                        chunk_id=self._generate_chunk_id(document.doc_id, chunk_index),
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        content=table_text,
                        summary=f"表格: {table.caption or table.headers[0] if table.headers else '无标题表格'}",
                        section_title=section.title,
                        chunk_type="table",
                        source_file=document.file_name,
                        page_num=table.page_num,
                        metadata={"has_table": True},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            # 1b. 章节正文分块
            if section.content.strip():
                section_chunks = self._chunk_section_content(
                    content=section.content,
                    doc_id=document.doc_id,
                    section_title=section.title,
                    source_file=document.file_name,
                    page_start=section.page_start,
                    start_index=chunk_index,
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

        # 2. 处理不属于任何章节的独立表格
        section_table_ids = set()
        for section in document.sections:
            for table in section.tables:
                section_table_ids.add(id(table))

        for table in document.tables:
            if id(table) not in section_table_ids:
                table_text = table.to_markdown()
                if table_text.strip():
                    chunk = Chunk(
                        chunk_id=self._generate_chunk_id(document.doc_id, chunk_index),
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        content=table_text,
                        summary=f"表格: {table.caption or '无标题表格'}",
                        section_title="",
                        chunk_type="table",
                        source_file=document.file_name,
                        page_num=table.page_num,
                        metadata={"has_table": True},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        # 3. 如果没有章节结构，对全文进行段落分块
        if not document.sections and document.full_text.strip():
            fallback_chunks = self._chunk_section_content(
                content=document.full_text,
                doc_id=document.doc_id,
                section_title="",
                source_file=document.file_name,
                page_start=0,
                start_index=chunk_index,
            )
            chunks.extend(fallback_chunks)

        # 4. 为每个块生成摘要
        self._generate_summaries(chunks)

        logger.info(f"分块完成: 共 {len(chunks)} 个块")
        return chunks

    # ── 章节内容分块 ──────────────────────────────────────────

    def _chunk_section_content(
        self,
        content: str,
        doc_id: str,
        section_title: str,
        source_file: str,
        page_start: int,
        start_index: int,
    ) -> List[Chunk]:
        """
        对章节正文进行分块

        策略：
        1. 先按段落（空行分隔）切分
        2. 合并短段落，直到接近 max_chunk_size
        3. 超长段落在句子边界处二次切分
        """
        chunks: List[Chunk] = []

        # 按段落拆分
        paragraphs = self._split_paragraphs(content)

        # 合并短段落
        merged_blocks = self._merge_paragraphs(paragraphs)

        chunk_idx = start_index
        for block in merged_blocks:
            block = block.strip()
            if not block:
                continue

            # 超长块在句子边界处二次切分
            if len(block) > self.max_chunk_size:
                sub_blocks = self._split_long_text(block)
            else:
                sub_blocks = [block]

            for sub_block in sub_blocks:
                sub_block = sub_block.strip()
                if len(sub_block) < self.min_chunk_size and chunks:
                    # 太短的块合并到前一个块
                    prev = chunks[-1]
                    prev.content += "\n\n" + sub_block
                    continue

                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(doc_id, chunk_idx),
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    content=sub_block,
                    section_title=section_title,
                    chunk_type="text",
                    source_file=source_file,
                    page_num=page_start,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    # ── 段落处理工具 ──────────────────────────────────────────

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        按段落边界拆分文本

        段落边界定义：
        - 连续空行
        - 列表项起始（- 或 * 或 数字.）
        - 医学文档特殊分隔符（如 ──、===、***）
        """
        # 先按双换行拆分
        raw_paragraphs = re.split(r"\n\s*\n", text)

        paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检测段落内的列表结构
            if re.search(r"\n[-*]\s+", para) or re.search(r"\n\d+[.、]\s+", para):
                # 包含列表，按列表项进一步拆分
                list_items = self._split_list_items(para)
                paragraphs.extend(list_items)
            else:
                paragraphs.append(para)

        return paragraphs

    def _split_list_items(self, text: str) -> List[str]:
        """拆分包含列表项的文本"""
        items = []
        lines = text.split("\n")
        current_item: List[str] = []

        for line in lines:
            # 检测列表项起始
            is_list_start = bool(re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+[.、]\s+", line))
            if is_list_start and current_item:
                items.append("\n".join(current_item))
                current_item = [line]
            else:
                current_item.append(line)

        if current_item:
            items.append("\n".join(current_item))

        return items if items else [text]

    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        合并短段落，使每个块接近 max_chunk_size

        策略：贪心合并，直到加入下一段落会超过 max_chunk_size
        """
        if not paragraphs:
            return []

        merged: List[str] = []
        current_block = paragraphs[0]

        for para in paragraphs[1:]:
            # 计算合并后长度（加上分隔符）
            combined_len = len(current_block) + 2 + len(para)  # \n\n 分隔
            if combined_len <= self.max_chunk_size:
                current_block += "\n\n" + para
            else:
                merged.append(current_block)
                current_block = para

        if current_block.strip():
            merged.append(current_block)

        return merged

    def _split_long_text(self, text: str) -> List[str]:
        """
        在句子边界处切分超长文本

        句子边界：中文句号(。)、问号(？)、感叹号(！)、分号(；)
        保留 overlap 重叠，避免语义截断
        """
        # 按句子边界拆分
        sentences = re.split(r"(?<=[。？！；\n])", text)
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return [text]

        blocks: List[str] = []
        current_block = ""

        for sentence in sentences:
            combined_len = len(current_block) + len(sentence)
            if combined_len <= self.max_chunk_size:
                current_block += sentence
            else:
                if current_block:
                    blocks.append(current_block)
                    # 保留重叠：取前一块末尾 overlap 个字符
                    if self.overlap > 0 and len(current_block) > self.overlap:
                        overlap_text = current_block[-self.overlap:]
                        current_block = overlap_text + sentence
                    else:
                        current_block = sentence
                else:
                    # 单个句子就超过 max_chunk_size，强制按字符切分
                    for i in range(0, len(sentence), self.max_chunk_size - self.overlap):
                        blocks.append(sentence[i:i + self.max_chunk_size])
                    current_block = ""

        if current_block.strip():
            blocks.append(current_block)

        return blocks

    # ── 摘要生成 ──────────────────────────────────────────────

    def _generate_summaries(self, chunks: List[Chunk]) -> None:
        """
        为每个块生成摘要

        策略（不调用LLM，使用规则提取）：
        1. 表格类型：使用表头和首行作为摘要
        2. 文本类型：提取前100个字符作为摘要
        3. 如果有章节标题，将标题纳入摘要
        """
        for chunk in chunks:
            if chunk.chunk_type == "table":
                # 表格摘要已在创建时设置
                if not chunk.summary:
                    lines = chunk.content.split("\n")
                    chunk.summary = lines[0] if lines else "表格数据"
            else:
                # 文本类型：提取首句 + 章节标题
                first_sentence = self._extract_first_sentence(chunk.content)
                if chunk.section_title:
                    chunk.summary = f"[{chunk.section_title}] {first_sentence}"
                else:
                    chunk.summary = first_sentence

    def _extract_first_sentence(self, text: str, max_len: int = 100) -> str:
        """提取文本首句"""
        # 按句子边界拆分
        sentences = re.split(r"[。？！；\n]", text)
        for s in sentences:
            s = s.strip()
            if s and len(s) > 10:
                return s[:max_len] + ("..." if len(s) > max_len else "")
        # 没有合适句子，取前 max_len 个字符
        return text[:max_len].strip() + ("..." if len(text) > max_len else "")

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int) -> str:
        """生成分块唯一ID"""
        raw = f"{doc_id}_{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
