# 医疗RAG知识库 - 混合检索智能问答系统

> 向量检索 + 医学知识图谱检索，用于医学指南、医典文档问答，重点抑制医疗幻觉，回答附带原文溯源。

## 项目概述

本项目是一个基于 **LangGraph 多Agent编排** 的医疗智能问答系统，采用 **Milvus 向量检索 + Neo4j 知识图谱** 双通道混合检索架构。

### 核心特性

- **混合检索**：Milvus 向量语义召回 + Neo4j 图谱关联查询
- **幻觉抑制**：独立事实校验Agent，识别矛盾医学信息
- **原文溯源**：回答强制附带文档来源引用
- **免责声明**：所有回答强制标注【仅供学习参考，不能替代执业医师诊断】
- **语义分块**：医疗文档按章节/段落语义切分，非固定长度粗暴切分
- **LangGraph Agent**：问题改写 → 实体抽取 → 并行检索 → 答案生成 → 事实校验

## 目录结构

```
E:\Medical-RAG-Knowledge-Base
├── data/
│   ├── raw/              # 原始医学文档 (pdf/md/txt)
│   └── processed/        # 清洗分段后的文本
├── config/               # 配置文件 (Pydantic Settings)
│   └── settings.py       # 全局配置，.env 只读一次
├── src/
│   ├── document_parser/  # PDF解析、表格抽取、医疗语义分块
│   │   ├── parser.py     # 文档解析器 (PDF/MD/TXT)
│   │   └── chunker.py    # 医疗语义分块器
│   ├── entity_extract/   # 医学实体抽取、关系抽取
│   │   └── extractor.py  # LLM驱动的实体/关系抽取
│   ├── vector_store/     # Milvus向量入库、检索
│   │   └── store.py      # 向量存储封装
│   ├── graph_store/      # Neo4j图谱构建、Cypher查询
│   │   └── store.py      # 图谱存储封装
│   ├── agent/            # LangGraph多Agent编排
│   │   ├── state.py      # Agent状态定义
│   │   ├── nodes.py      # Agent节点(改写/抽取/检索/生成/校验)
│   │   └── graph.py      # LangGraph图构建与执行
│   └── rag_pipeline.py   # 项目主入口文件
├── core/                 # 核心基础设施
│   ├── embedding.py      # Embedding模型 (本地优先/API降级)
│   ├── milvus_schema.py  # Milvus Schema定义
│   ├── neo4j_schema.py   # Neo4j Schema定义
│   └── safe_json.py      # JSON安全解析工具
├── services/             # 服务层
│   ├── db/               # 数据库连接池
│   │   ├── milvus_pool.py
│   │   ├── neo4j_pool.py
│   │   └── postgres_pool.py
│   └── llm/              # LLM服务 (Qwen)
│       └── llm_service.py
├── tests/                # 测试模块
│   ├── test_document_parser.py  # 文档解析/分块测试
│   ├── test_retrieval.py        # 召回测试
│   └── test_hallucination.py    # 幻觉对抗测试
├── .env                  # 环境变量配置
├── docker-compose.yml    # Milvus + Neo4j 容器编排
├── requirements.txt      # Python依赖
└── main.py               # FastAPI服务入口
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 大模型 | Qwen (OpenAI兼容接口) |
| Agent编排 | LangGraph |
| 向量数据库 | Milvus 2.4 |
| 图数据库 | Neo4j 5.15 |
| Embedding | bge-large-zh-v1.5 (本地优先) |
| 文档解析 | PyMuPDF |
| Web框架 | FastAPI |
| 容器化 | Docker Compose |

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 配置环境变量
# 编辑 .env 文件，填写 LLM_API_KEY、NEO4J_PASSWORD、MILVUS_PASSWORD 等
```

### 2. 启动基础设施

```bash
# Docker 一键启动 Milvus + Neo4j
docker-compose up -d

# 等待服务就绪 (约 2-3 分钟)
docker-compose ps
```

### 3. 文档入库

```bash
# 入库单个文档
python -m src.rag_pipeline ingest --file data/raw/高血压指南.pdf

# 批量入库整个目录
python -m src.rag_pipeline ingest --dir data/raw/ --source-type textbook

# 跳过实体抽取（仅向量入库）
python -m src.rag_pipeline ingest --file data/raw/文档.pdf --no-entity
```

### 4. 问答

```bash
# 单次问答
python -m src.rag_pipeline ask "高血压有哪些常见症状？"

# 交互式问答
python -m src.rag_pipeline interactive

# 查看系统状态
python -m src.rag_pipeline status
```

### 5. 运行测试

```bash
# 运行文档解析测试
pytest tests/test_document_parser.py -v

# 运行幻觉对抗测试
pytest tests/test_hallucination.py -v

# 运行全部测试
pytest tests/ -v
```

## LangGraph Agent 工作流

```
用户提问
  │
  ▼
┌─────────────────┐
│ 问题改写 Agent   │  口语→术语标准化 + 子查询拆分
└────────┬────────┘
         │
┌────────▼────────┐
│ 实体抽取 Agent   │  提取疾病/症状/药物/检查
└────────┬────────┘
         │
┌────────▼────────────────────┐
│        并行检索              │
│  ┌──────────┐ ┌──────────┐  │
│  │Milvus    │ │Neo4j     │  │
│  │向量召回   │ │图谱关联   │  │
│  └────┬─────┘ └────┬─────┘  │
└───────┼────────────┼────────┘
        │            │
┌───────▼────────────▼────────┐
│       上下文拼接             │
└───────────┬─────────────────┘
            │
┌───────────▼─────────────────┐
│    LLM 答案生成             │  基于参考资料 + 来源标注
└───────────┬─────────────────┘
            │
┌───────────▼─────────────────┐
│    事实校验 Agent            │  矛盾/幻觉/错误用药检测
└───────────┬─────────────────┘
            │
┌───────────▼─────────────────┐
│  输出答案 + 引用来源         │
│  + 免责声明                 │
└─────────────────────────────┘
```

## 分块策略说明

本项目不使用固定长度粗暴切分，采用以下语义分块策略：

1. **章节优先**：按文档标题层级（第X章、1.1、一、）切分
2. **表格独立**：表格完整保留为独立块，不拆分
3. **段落合并**：短段落贪心合并至接近 max_chunk_size
4. **句子边界**：超长段落在句号/问号/分号处切分
5. **摘要增强**：每个块附带摘要，以 `[摘要]...[正文]...` 格式存入向量库

## Windows 开发注意事项

1. **WSL 挂载 E 盘性能**：WSL 挂载 `/mnt/e/` 性能较差，建议在 Windows 原生 Python 环境运行
2. **Docker Desktop**：确保 Docker Desktop 已启动，且配置了足够的内存（建议 8GB+）
3. **路径分隔符**：代码中使用 `pathlib.Path` 处理路径，自动兼容 Windows/Linux

## 配置说明

所有配置通过 `.env` 文件管理，禁止硬编码密钥与参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_API_KEY` | Qwen API Key | 必填 |
| `LLM_API_BASE` | LLM API 地址 | - |
| `LLM_MODEL_NAME` | 模型名称 | qwen-plus |
| `NEO4J_PASSWORD` | Neo4j 密码 | 必填 |
| `MILVUS_PASSWORD` | Milvus 密码 | 必填 |
| `DOCRAG_TOP_K` | 向量召回数量 | 5 |
| `DOCRAG_MIN_SCORE` | 最低相似度阈值 | 0.3 |
| `DOCRAG_CHUNK_SIZE` | 分块最大字符数 | 512 |
| `CHANNEL_DOC_RAG_ENABLED` | 启用向量检索通道 | true |
| `CHANNEL_GRAPH_RAG_ENABLED` | 启用图谱检索通道 | true |

## 免责声明

**本系统所有回答仅供学习参考，不能替代执业医师诊断。**

系统禁止输出可直接用于临床诊断的结论，所有回答强制标注免责声明。
