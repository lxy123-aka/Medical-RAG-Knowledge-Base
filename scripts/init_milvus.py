"""
Milvus 向量库建表脚本（含 Schema 集中管理与维度一致检查）
"""

import asyncio
import logging
import sys
import os

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from core.milvus_schema import (
    CollectionNames,
    EMBEDDING_DIM,
    SCHEMA_REGISTRY,
    validate_dimension_consistency,
    INDEX_PARAMS,
)
from services.db.milvus_pool import get_milvus_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主入口"""
    logger.info("=" * 50)
    logger.info("  Milvus 向量库初始化")
    logger.info("=" * 50)

    # 1. 维度一致性检查
    logger.info("检查 Embedding 维度一致性...")
    try:
        validate_dimension_consistency()
        logger.info(f"  ✓ 所有集合 Embedding 维度一致: {EMBEDDING_DIM}")
    except ValueError as e:
        logger.error(f"  ✗ {e}")
        sys.exit(1)

    # 2. 初始化连接
    pool = get_milvus_pool()
    await pool.initialize()

    # 3. 创建集合
    logger.info("创建集合...")
    await pool.ensure_collections()

    # 4. 验证
    logger.info("验证集合创建结果...")
    from pymilvus import utility
    for name in SCHEMA_REGISTRY.keys():
        try:
            exists = utility.has_collection(name)
            if exists:
                logger.info(f"  ✓ 集合 {name} 已创建")
            else:
                logger.error(f"  ✗ 集合 {name} 未找到")
        except Exception as e:
            logger.warning(f"  ⚠ 验证集合 {name} 失败: {e}")

    logger.info("=" * 50)
    logger.info("  Milvus 初始化完成！")
    logger.info("=" * 50)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
