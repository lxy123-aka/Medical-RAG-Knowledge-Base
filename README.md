# 医典 RAG 多维知识库系统

> 为 AI 智能体打通三通道工程化检索方案：DocRAG（向量检索）、GraphRAG（知识图谱）、SQLRAG（关系数据库）

## 项目介绍

医典 RAG 多维知识库系统是从智慧问诊系统下沉的基础设施组件，提供完整的 RAG（Retrieval-Augmented Generation）检索增强生成能力。系统支持医学教材、病历、药典文档的批量入库，通过三通道并行检索和智能融合，为上层 AI 智能体提供高质量的医学知识检索服务。

### 核心特性

- **三通道并行检索**：DocRAG（Milvus 向量库）、GraphRAG（Neo4j 知识图谱）、SQLRAG（PostgreSQL 关系数据库）
- **动态 HyDE + Rewrite**：假设文档增强检索 + 查询重写（口语→医学术语标准化 + 子查询拆分 + 意图识别路由）
- **NL2Cypher 错误反馈重试**：图谱查询失败后自动修正重生成，目标成功率 ~90%
- **TruLens + OpenTelemetry 评估体系**：RAG 三元指标（Context Relevance、Groundedness、Answer Relevance）按通道分别评估
- **权限过滤**：基于数据 meta 字段实现权限过滤切片
- **配置驱动**：通道开关、阈值、采样率全部走配置，不硬编码

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI 应用层                            │
│  /api/v1/query/search  /api/v1/documents  /api/v1/health       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       融合引擎 (FusionEngine)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Query    │  │ asyncio  │  │ 结果融合  │                      │
│  │ Rewrite  │→ │ .gather  │→ │ + 降级   │→ LLM 生成答案         │
│  │ + HyDE   │  │ 并行检索  │  │          │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└───────┬─────────────┬─────────────┬─────────────────────────────┘
        │             │             │
┌───────▼──────┐ ┌────▼───────┐ ┌──▼──────────┐
│   DocRAG     │ │  GraphRAG  │ │   SQLRAG    │
│              │ │            │ │             │
│ Milvus 向量  │ │ Neo4j 图谱 │ │ PostgreSQL  │
│ + HyDE       │ │ + NL2Cypher│ │ + NL2SQL    │
│ + Reranker   │ │ + 重试机制  │ │             │
└──────────────┘ └────────────┘ └─────────────┘
        │             │             │
┌───────▼─────────────▼─────────────▼─────────────────────────────┐
│                     评估体系 (TruLens + OTEL)                     │
│  Context Relevance | Groundedness | Answer Relevance             │
└─────────────────────────────────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Web 框架 | FastAPI | API 服务 |
| RAG 框架 | LangChain + LlamaIndex | 文档处理、分片 |
| 文档解析 | MinerU | PDF 高精度还原为 Markdown |
| 向量数据库 | Milvus | DocRAG 向量检索 |
| 图数据库 | Neo4j | GraphRAG 知识图谱 |
| 关系数据库 | PostgreSQL | SQLRAG 关系数据 |
| Embedding | bge-large-zh-v1.5 | 文本向量化（本地优先） |
| LLM | Qwen 系列 | 生成、重写、NL2Cypher |
| 评估 | TruLens | RAG 三元指标评估 |
| 可观测性 | OpenTelemetry | 分布式追踪 |
| 容器编排 | Docker Compose | 一键部署 |

## 项目目录结构

```
Medical-RAG-Knowledge-Base/
├── main.py                          # FastAPI 启动入口
├── .env.example                     # 环境变量示例
├── requirements.txt                 # Python 依赖
├── docker-compose.yml               # 容器编排
│
├── config/                          # 配置层
│   ├── settings.py                  # Pydantic Settings（启动时一次性校验）
│   └── constants.py                 # 全局常量
│
├── core/                            # 核心工具层
│   ├── neo4j_schema.py             # Neo4j Schema 共享常量（规则1）
│   ├── milvus_schema.py            # Milvus Schema 集中管理（规则3）
│   ├── safe_json.py                # safe_json_parse 工具（规则6）
│   └── embedding.py                # Embedding 加载（本地优先，规则10）
│
├── models/                          # 数据模型
│   └── schemas.py                  # Pydantic 请求/响应模型
│
├── services/                        # 服务层
│   ├── db/
│   │   ├── neo4j_pool.py           # Neo4j 连接池（规则2/4）
│   │   ├── milvus_pool.py          # Milvus 连接池（规则3/4）
│   │   └── postgres_pool.py        # PostgreSQL 连接池（规则4）
│   └── llm/
│       └── llm_service.py          # LLM 统一封装
│
├── rag/                             # RAG 三通道
│   ├── docrag/
│   │   └── channel.py              # DocRAG 向量检索通道
│   ├── graphrag/
│   │   └── channel.py              # GraphRAG 知识图谱通道
│   └── sqlrag/
│       └── channel.py              # SQLRAG 关系数据库通道
│
├── fusion/                          # 融合层
│   └── fusion_engine.py            # 并行检索 + 降级 + 融合
│
├── document/                        # 文档处理
│   └── mineru_parser.py            # MinerU 解析 + 分片 + 入库
│
├── evaluation/                      # 评估体系
│   └── trulens_eval.py             # TruLens RAG 三元评估
│
├── observability/                   # 可观测性
│   └── otel_tracing.py             # OpenTelemetry 追踪
│
├── api/                             # API 路由
│   └── routes/
│       ├── query.py                # 检索查询接口
│       ├── documents.py            # 文档管理接口
│       ├── health.py               # 健康检查接口
│       └── evaluation.py           # 评估接口
│
├── scripts/                         # 初始化脚本
│   ├── init_neo4j.py               # Neo4j 图谱初始化
│   ├── init_milvus.py              # Milvus 向量库初始化
│   ├── init_postgres.sql           # PostgreSQL 建表 SQL
│   └── verify.py                   # 验证清单脚本（规则17）
│
├── tests/                           # 测试用例
│   └── test_integration.py         # 集成测试
│
└── data/                            # 数据目录
    ├── neo4j/
    ├── milvus/
    └── postgres/
```

## 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- Node.js（可选，用于前端）

### 1. 克隆项目并配置环境

```powershell
# Windows PowerShell
cd E:\Medical-RAG-Knowledge-Base

# 创建虚拟环境
python -m venv venv
.\venv\Scripts\Activate

# 安装依赖
pip install -r requirements.txt

# 复制环境变量配置
Copy-Item .env.example .env
# 编辑 .env 文件，填入实际的 API Key 和密码
```

### 2. 启动基础设施（Docker）

```powershell
# 启动所有数据库服务
docker-compose up -d

# 等待服务就绪（约 2-3 分钟）
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 3. 初始化数据库

```powershell
# 初始化 Neo4j 知识图谱
python scripts/init_neo4j.py

# 初始化 Milvus 向量库
python scripts/init_milvus.py

# PostgreSQL 会自动通过 docker-compose 初始化
```

### 4. 启动应用

```powershell
# 开发模式（热重载）
python main.py

# 或使用 uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 访问 API 文档

打开浏览器访问：http://localhost:8000/docs

## API 接口

### 多通道检索

```http
POST /api/v1/query/search
Content-Type: application/json

{
    "query": "高血压的常见症状和治疗方法",
    "channels": ["docrag", "graphrag", "sqlrag"],
    "top_k": 5,
    "permission_level": 0,
    "enable_hyde": true,
    "enable_rewrite": true,
    "enable_reranker": true
}
```

### 单通道检索

```http
POST /api/v1/query/search/single?channel=docrag
Content-Type: application/json

{
    "query": "高血压的症状",
    "top_k": 5
}
```

### 文档上传

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <PDF文件>
source_type: textbook
department: 心血管内科
permission_level: 0
```

### 健康检查

```http
GET /api/v1/health/
```

### 评估

```http
POST /api/v1/evaluation/evaluate
Content-Type: application/json

{
    "query": "高血压的症状",
    "ground_truth": "头痛、头晕、心悸..."
}
```

## 评估体系

### TruLens RAG 三元指标

| 指标 | 说明 |
|------|------|
| Context Relevance | 检索到的上下文与用户问题的相关性 |
| Groundedness | 生成的答案是否基于检索到的上下文 |
| Answer Relevance | 生成的答案与用户问题的相关性 |

### 采样率配置

- 开发阶段：`TRULENS_SAMPLE_RATE=1.0`（100%）
- 线上环境：`TRULENS_SAMPLE_RATE=0.0001`（0.01%）

### 通道对比实验

```http
POST /api/v1/evaluation/compare?query=高血压的症状
```

## 验证清单

运行验证脚本检查系统完整性：

```powershell
python scripts/verify.py
```

检查项目：
1. ✓ Python 文件语法检查
2. ✓ Neo4j Schema 属性名一致性
3. ✓ Milvus Embedding 维度一致性
4. ✓ 环境变量/配置检查
5. ✓ JSON 解析容错测试
6. ✓ 数据库连通性检查

## 开发规范落实清单

| 规则 | 描述 | 落实位置 |
|------|------|---------|
| 1 | Neo4j 属性名全局强一致 | `core/neo4j_schema.py` + 启动校验 |
| 2 | Neo4j 禁止 fetch(-1) | `services/db/neo4j_pool.py` |
| 3 | Milvus DataType 枚举 + 维度一致 | `core/milvus_schema.py` |
| 4 | 数据库健康检查 + 自动重连 | 各 `*_pool.py` |
| 5 | 关系属性完整性断言 | `scripts/init_neo4j.py` |
| 6 | safe_json_parse 统一解析 | `core/safe_json.py` |
| 7 | LLM 结构化输出容错 | `core/safe_json.py` |
| 8 | 空结果降级逻辑 | `fusion/fusion_engine.py` |
| 9 | min_score 阈值按集合调优 | `config/settings.py` |
| 10 | Embedding 本地优先 | `core/embedding.py` |
| 13 | .env 只读一次 | `config/settings.py` |
| 14 | PowerShell 兼容 | README 命令 |
| 15 | NL2Cypher 重试 + Schema 注入 | `rag/graphrag/channel.py` |
| 16 | TruLens 从第一天埋观测点 | `evaluation/trulens_eval.py` |
| 17 | 验证清单脚本 | `scripts/verify.py` |
| 18 | 单例连接池 lifespan 初始化 | `main.py` lifespan |
| 19 | 禁止静默失效 | 全项目 |

## Shell 兼容性说明

本项目所有命令均兼容 **Windows PowerShell** 语法：
- 使用 `;` 而非 `&&` 连接命令
- 使用 `Copy-Item` 而非 `cp`
- 虚拟环境激活使用 `.\venv\Scripts\Activate`

## License

MIT
