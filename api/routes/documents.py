"""
API 路由 - 文档管理（上传、解析、入库）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from document.mineru_parser import get_mineru_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["文档管理"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: str = Form(default=None),
    source_type: str = Form(default="textbook"),
    department: str = Form(default=None),
    permission_level: int = Form(default=0),
) -> Dict[str, Any]:
    """
    上传并解析文档

    支持 PDF 文件，通过 MinerU 解析为 Markdown 后分片入库。

    参数:
    - file: PDF 文件
    - doc_id: 文档 ID（可选，不指定则自动生成）
    - source_type: 来源类型 (textbook/record/pharmacopoeia)
    - department: 所属科室
    - permission_level: 权限等级
    """
    # 验证文件类型
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存临时文件
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 调用 MinerU 解析并入库
        mineru_service = get_mineru_service()
        result = await mineru_service.parse_and_ingest(
            file_path=str(temp_path),
            doc_id=doc_id,
            source_type=source_type,
            department=department,
            permission_level=permission_level,
        )

        return {
            "status": "success",
            "message": "文档解析入库成功",
            "data": result,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API] 文档上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


@router.get("/list")
async def list_documents(
    source_type: str = None,
    department: str = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    列出已入库的文档
    """
    from services.db.postgres_pool import get_postgres_pool

    try:
        postgres_pool = get_postgres_pool()

        conditions = []
        params = {"limit": limit, "offset": offset}

        if source_type:
            conditions.append("source_type = :source_type")
            params["source_type"] = source_type
        if department:
            conditions.append("department = :department")
            params["department"] = department

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT doc_id, title, source_type, department,
                   permission_level, page_count, created_at, updated_at
            FROM documents
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """

        results = await postgres_pool.execute_query(sql, params)

        return {
            "status": "success",
            "data": results,
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"[API] 文档列表查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, Any]:
    """
    删除指定文档（同时删除 Milvus 和 PostgreSQL 中的数据）
    """
    from services.db.milvus_pool import get_milvus_pool
    from services.db.postgres_pool import get_postgres_pool
    from core.milvus_schema import CollectionNames

    try:
        # 删除 Milvus 数据
        milvus_pool = get_milvus_pool()
        await milvus_pool.delete(
            CollectionNames.DOC_RAG,
            filter_expr=f'doc_id == "{doc_id}"',
        )

        # 删除 PostgreSQL 数据
        postgres_pool = get_postgres_pool()
        await postgres_pool.execute_query(
            "DELETE FROM document_chunks WHERE doc_id = :doc_id",
            {"doc_id": doc_id},
        )
        await postgres_pool.execute_query(
            "DELETE FROM documents WHERE doc_id = :doc_id",
            {"doc_id": doc_id},
        )

        return {
            "status": "success",
            "message": f"文档 {doc_id} 已删除",
        }

    except Exception as e:
        logger.error(f"[API] 文档删除失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
