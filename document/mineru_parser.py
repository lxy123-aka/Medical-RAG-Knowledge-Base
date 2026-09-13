"""
MinerU 文档解析服务 - PDF 高精度还原为 Markdown + 分片入库
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config.settings import get_settings
from core.embedding import get_embedding_model
from core.milvus_schema import CollectionNames
from services.db.milvus_pool import get_milvus_pool
from services.db.postgres_pool import get_postgres_pool

logger = logging.getLogger(__name__)


class MinerUService:
    """
    MinerU 文档解析服务

    功能:
    1. 调用 MinerU API 解析 PDF（表格/公式高精度还原为 Markdown）
    2. 文本分片（按 chunk_size / chunk_overlap）
    3. 生成 Embedding 并存入 Milvus
    4. 元数据存入 PostgreSQL
    """

    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """初始化 HTTP 客户端"""
        settings = get_settings()
        self._http_client = httpx.AsyncClient(
            base_url=settings.mineru_api_base,
            headers={"Authorization": f"Bearer {settings.mineru_api_key}"},
            timeout=300.0,  # 文档解析可能较慢
        )
        logger.info(f"MinerU 服务初始化: {settings.mineru_api_base}")

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()

    async def parse_and_ingest(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        source_type: str = "textbook",
        department: Optional[str] = None,
        permission_level: int = 0,
    ) -> Dict[str, Any]:
        """
        解析文档并入库

        Args:
            file_path: PDF 文件路径
            doc_id: 文档 ID（不指定则自动生成）
            source_type: 来源类型
            department: 所属科室
            permission_level: 权限等级

        Returns:
            入库结果统计
        """
        start_time = time.time()
        settings = get_settings()

        doc_id = doc_id or str(uuid.uuid4())
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"[MinerU] 开始解析文档: {file_path.name} (doc_id={doc_id})")

        # 1. 调用 MinerU 解析 PDF
        markdown_content = await self._parse_pdf(file_path)
        if not markdown_content:
            raise ValueError("MinerU 解析返回空内容")

        logger.info(f"[MinerU] 解析完成，Markdown 长度: {len(markdown_content)}")

        # 2. 文本分片
        chunks = self._split_text(
            markdown_content,
            chunk_size=settings.docrag_chunk_size,
            chunk_overlap=settings.docrag_chunk_overlap,
        )
        logger.info(f"[MinerU] 分片完成: {len(chunks)} 个分片")

        # 3. 生成 Embedding
        embedding_model = get_embedding_model()
        chunk_texts = [c["content"] for c in chunks]
        embeddings = await embedding_model.embed_documents(chunk_texts)

        # 4. 存入 Milvus
        milvus_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            milvus_data.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "content": chunk["content"],
                "embedding": embedding,
                "permission_level": permission_level,
                "source_type": source_type,
                "department": department or "",
                "created_at": datetime.now().isoformat(),
            })

        milvus_pool = get_milvus_pool()
        await milvus_pool.insert(CollectionNames.DOC_RAG, milvus_data)
        logger.info(f"[MinerU] Milvus 入库完成: {len(milvus_data)} 条")

        # 5. 元数据存入 PostgreSQL
        postgres_pool = get_postgres_pool()
        await postgres_pool.execute_query(
            """
            INSERT INTO documents (doc_id, title, source_type, department,
                                   permission_level, page_count, file_path)
            VALUES (:doc_id, :title, :source_type, :department,
                    :permission_level, :page_count, :file_path)
            ON CONFLICT (doc_id) DO UPDATE SET
                updated_at = NOW(),
                page_count = :page_count
            """,
            {
                "doc_id": doc_id,
                "title": file_path.stem,
                "source_type": source_type,
                "department": department or "",
                "permission_level": permission_level,
                "page_count": len(chunks),
                "file_path": str(file_path),
            },
        )

        # 批量插入分片元数据
        chunk_records = [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "content": chunk["content"][:1000],  # 截断存储
                "token_count": chunk.get("token_count", 0),
            }
            for i, chunk in enumerate(chunks)
        ]
        await postgres_pool.execute_many(
            """
            INSERT INTO document_chunks (doc_id, chunk_index, content, token_count)
            VALUES (:doc_id, :chunk_index, :content, :token_count)
            """,
            chunk_records,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        result = {
            "doc_id": doc_id,
            "file_name": file_path.name,
            "markdown_length": len(markdown_content),
            "chunk_count": len(chunks),
            "milvus_inserted": len(milvus_data),
            "postgres_inserted": 1 + len(chunk_records),
            "elapsed_ms": round(elapsed_ms, 2),
        }

        logger.info(f"[MinerU] 文档入库完成: {result}")
        return result

    async def _parse_pdf(self, file_path: Path) -> str:
        """调用 MinerU API 解析 PDF"""
        if not self._http_client:
            await self.initialize()

        try:
            # 上传文件
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/pdf")}
                response = await self._http_client.post(
                    "/api/v1/parse",
                    files=files,
                )

            response.raise_for_status()
            result = response.json()

            # MinerU 返回格式
            markdown = result.get("markdown", "")
            if not markdown:
                logger.warning(f"[MinerU] API 返回空 Markdown: {result}")

            return markdown

        except httpx.HTTPStatusError as e:
            logger.error(f"[MinerU] API 请求失败: {e.response.status_code} {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[MinerU] 解析异常: {e}")
            raise

    def _split_text(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        文本分片

        使用 LangChain 的 RecursiveCharacterTextSplitter
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "；", ";", " "],
            length_function=len,
        )

        chunks = splitter.split_text(text)
        return [
            {
                "content": chunk,
                "token_count": len(chunk),
            }
            for chunk in chunks
        ]


# ── 全局单例 ─────────────────────────────────────────────────
_mineru_service: Optional[MinerUService] = None


def get_mineru_service() -> MinerUService:
    """获取 MinerU 服务全局单例"""
    global _mineru_service
    if _mineru_service is None:
        _mineru_service = MinerUService()
    return _mineru_service
