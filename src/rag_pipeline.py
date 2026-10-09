"""
医疗RAG知识库 - 项目主入口文件

功能：
1. 文档入库流程（解析 → 分块 → 向量化 → 入库 → 实体抽取 → 图谱构建）
2. 问答流程（调用 LangGraph 多Agent工作流）
3. 命令行交互模式

使用方法：
    # 文档入库
    python -m src.rag_pipeline ingest --file data/raw/医学指南.pdf
    python -m src.rag_pipeline ingest --dir data/raw/

    # 问答
    python -m src.rag_pipeline ask "高血压有哪些常见症状？"

    # 交互模式
    python -m src.rag_pipeline interactive

    # 系统状态
    python -m src.rag_pipeline status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 文档入库流程
# =============================================================================

async def ingest_document(
    file_path: str,
    doc_id: Optional[str] = None,
    source_type: str = "textbook",
    department: str = "",
    extract_entities: bool = True,
) -> Dict[str, Any]:
    """
    文档入库完整流程

    步骤：
    1. PDF/MD/TXT 解析 → 结构化文档
    2. 医疗语义分块 → 带摘要的文本块
    3. 向量化 → 写入 Milvus
    4. 实体抽取 → 写入 Neo4j 图谱

    Args:
        file_path: 文件路径
        doc_id: 文档ID（可选）
        source_type: 来源类型
        department: 所属科室
        extract_entities: 是否抽取实体写入图谱

    Returns:
        入库统计信息
    """
    start_time = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"  文档入库: {file_path}")
    logger.info(f"{'='*60}")

    stats = {"file": file_path, "status": "success"}

    # ── 步骤1: 文档解析 ──────────────────────────────────────
    logger.info("[1/4] 文档解析...")
    try:
        from src.document_parser.parser import MedicalDocumentParser
        parser = MedicalDocumentParser()
        document = parser.parse(file_path, doc_id=doc_id)
        stats["pages"] = len(document.pages)
        stats["sections"] = len(document.sections)
        stats["tables"] = len(document.tables)
        logger.info(
            f"  解析完成: {stats['pages']}页, "
            f"{stats['sections']}章节, {stats['tables']}表格"
        )
    except Exception as e:
        logger.error(f"  文档解析失败: {e}")
        stats["status"] = "failed"
        stats["error"] = str(e)
        return stats

    # ── 步骤2: 语义分块 ──────────────────────────────────────
    logger.info("[2/4] 医疗语义分块...")
    try:
        from src.document_parser.chunker import MedicalSemanticChunker
        settings = get_settings()
        chunker = MedicalSemanticChunker(
            max_chunk_size=settings.docrag_chunk_size,
            overlap=settings.docrag_chunk_overlap,
        )
        chunks = chunker.chunk(document)
        stats["chunks"] = len(chunks)
        logger.info(f"  分块完成: {len(chunks)} 个块")
    except Exception as e:
        logger.error(f"  分块失败: {e}")
        stats["status"] = "failed"
        stats["error"] = str(e)
        return stats

    # ── 步骤3: 向量化入库 ────────────────────────────────────
    logger.info("[3/4] 向量化入库 (Milvus)...")
    try:
        from src.vector_store.store import VectorStore
        from core.embedding import get_embedding_model

        store = VectorStore()
        await store.initialize()
        embedding_model = get_embedding_model()
        if embedding_model.mode == "uninitialized":
            await embedding_model.initialize()

        ingest_result = await store.ingest_chunks(
            chunks=chunks,
            embedding_model=embedding_model,
            doc_id=document.doc_id,
            source_type=source_type,
            department=department,
        )
        stats["vector_inserted"] = ingest_result.get("inserted", 0)
        logger.info(f"  向量入库: {stats['vector_inserted']} 条")
    except Exception as e:
        logger.error(f"  向量入库失败: {e}")
        stats["vector_error"] = str(e)

    # ── 步骤4: 实体抽取入库 ──────────────────────────────────
    if extract_entities:
        logger.info("[4/4] 实体抽取 → 图谱构建 (Neo4j)...")
        try:
            from src.entity_extract.extractor import MedicalEntityExtractor
            extractor = MedicalEntityExtractor()
            await extractor.initialize()

            extraction = await extractor.extract_from_chunks(chunks)
            stats["entities"] = len(extraction.entities)
            stats["relations"] = len(extraction.relations)

            if extraction.entities or extraction.relations:
                write_stats = await extractor.write_to_graph(extraction)
                stats["entities_upserted"] = write_stats.get("entities_upserted", 0)
                stats["relations_upserted"] = write_stats.get("relations_upserted", 0)

            logger.info(
                f"  实体抽取: {stats.get('entities', 0)} 个, "
                f"关系: {stats.get('relations', 0)} 条"
            )
        except Exception as e:
            logger.error(f"  实体抽取失败: {e}")
            stats["entity_error"] = str(e)
    else:
        logger.info("[4/4] 跳过实体抽取")

    elapsed = (time.time() - start_time) * 1000
    stats["elapsed_ms"] = elapsed
    logger.info(f"文档入库完成, 总耗时: {elapsed:.0f}ms")
    return stats


async def ingest_directory(
    dir_path: str,
    source_type: str = "textbook",
    department: str = "",
    extract_entities: bool = True,
) -> List[Dict[str, Any]]:
    """批量入库目录下所有文档"""
    dir_p = Path(dir_path)
    if not dir_p.exists():
        logger.error(f"目录不存在: {dir_path}")
        return []

    supported_ext = {".pdf", ".md", ".txt"}
    files = [
        f for f in dir_p.iterdir()
        if f.is_file() and f.suffix.lower() in supported_ext
    ]

    if not files:
        logger.warning(f"目录中没有支持的文档: {dir_path}")
        return []

    logger.info(f"发现 {len(files)} 个文档待入库")
    results = []
    for file in files:
        result = await ingest_document(
            str(file),
            source_type=source_type,
            department=department,
            extract_entities=extract_entities,
        )
        results.append(result)

    return results


# =============================================================================
# 问答流程
# =============================================================================

async def ask_question(query: str) -> Dict[str, Any]:
    """
    医疗问答

    Args:
        query: 用户问题

    Returns:
        问答结果
    """
    from src.agent.graph import run_medical_rag
    return await run_medical_rag(query)


# =============================================================================
# 系统状态
# =============================================================================

async def show_status() -> Dict[str, Any]:
    """显示系统各组件状态"""
    status = {"components": {}}

    # Milvus
    try:
        from src.vector_store.store import VectorStore
        store = VectorStore()
        await store.initialize()
        stats = await store.get_stats()
        status["components"]["milvus"] = {"status": "ok", "stats": stats}
    except Exception as e:
        status["components"]["milvus"] = {"status": "error", "error": str(e)}

    # Neo4j
    try:
        from src.graph_store.store import GraphStore
        store = GraphStore()
        await store.initialize()
        stats = await store.get_graph_stats()
        status["components"]["neo4j"] = {"status": "ok", "stats": stats}
    except Exception as e:
        status["components"]["neo4j"] = {"status": "error", "error": str(e)}

    # Embedding
    try:
        from core.embedding import get_embedding_model
        model = get_embedding_model()
        if model.mode == "uninitialized":
            await model.initialize()
        status["components"]["embedding"] = {
            "status": "ok",
            "mode": model.mode,
            "dimension": model.dimension,
        }
    except Exception as e:
        status["components"]["embedding"] = {"status": "error", "error": str(e)}

    return status


# =============================================================================
# 交互模式
# =============================================================================

async def interactive_mode():
    """交互式问答模式"""
    print("\n" + "=" * 60)
    print("  医疗RAG知识库 - 交互式问答")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'status' 查看系统状态")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("🏥 请输入医学问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if query.lower() == "status":
            status = await show_status()
            for comp, info in status["components"].items():
                print(f"  {comp}: {info['status']}")
            continue

        result = await ask_question(query)

        print("\n" + "-" * 60)
        print(f"📋 回答:\n{result.get('final_answer', '无回答')}")
        print(f"\n📚 引用文档:")
        for doc in result.get("source_documents", []):
            print(f"  - {doc.get('doc_id', '未知')} ({doc.get('source_type', '')})")
        print(f"\n⏱️  耗时: {result.get('elapsed_ms', 0):.0f}ms")
        print(f"✅ 事实校验: {'通过' if result.get('verification_passed') else '未通过'}")
        if result.get("verification_issues"):
            print(f"⚠️  校验问题:")
            for issue in result["verification_issues"]:
                print(f"  - {issue}")
        print("-" * 60 + "\n")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="医疗RAG知识库系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── 文档入库 ──────────────────────────────────────────────
    ingest_parser = subparsers.add_parser("ingest", help="文档入库")
    ingest_parser.add_argument("--file", type=str, help="单个文件路径")
    ingest_parser.add_argument("--dir", type=str, help="目录路径（批量入库）")
    ingest_parser.add_argument("--doc-id", type=str, default=None, help="文档ID")
    ingest_parser.add_argument(
        "--source-type", type=str, default="textbook",
        choices=["textbook", "record", "pharmacopoeia"],
        help="来源类型",
    )
    ingest_parser.add_argument("--department", type=str, default="", help="所属科室")
    ingest_parser.add_argument(
        "--no-entity", action="store_true", help="跳过实体抽取"
    )

    # ── 问答 ──────────────────────────────────────────────────
    ask_parser = subparsers.add_parser("ask", help="单次问答")
    ask_parser.add_argument("query", type=str, help="问题文本")

    # ── 交互模式 ──────────────────────────────────────────────
    subparsers.add_parser("interactive", help="交互式问答")

    # ── 系统状态 ──────────────────────────────────────────────
    subparsers.add_parser("status", help="查看系统状态")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 运行异步命令
    if args.command == "ingest":
        if args.file:
            result = asyncio.run(ingest_document(
                args.file,
                doc_id=args.doc_id,
                source_type=args.source_type,
                department=args.department,
                extract_entities=not args.no_entity,
            ))
            print(f"\n入库结果: {result}")
        elif args.dir:
            results = asyncio.run(ingest_directory(
                args.dir,
                source_type=args.source_type,
                department=args.department,
                extract_entities=not args.no_entity,
            ))
            for r in results:
                print(f"  {r['file']}: {r['status']}")
        else:
            print("请指定 --file 或 --dir")

    elif args.command == "ask":
        result = asyncio.run(ask_question(args.query))
        print(f"\n回答: {result.get('final_answer', '无回答')}")
        print(f"耗时: {result.get('elapsed_ms', 0):.0f}ms")

    elif args.command == "interactive":
        asyncio.run(interactive_mode())

    elif args.command == "status":
        status = asyncio.run(show_status())
        print("\n系统状态:")
        for comp, info in status["components"].items():
            status_icon = "✓" if info["status"] == "ok" else "✗"
            print(f"  {status_icon} {comp}: {info['status']}")
            if "error" in info:
                print(f"    错误: {info['error']}")


if __name__ == "__main__":
    main()
