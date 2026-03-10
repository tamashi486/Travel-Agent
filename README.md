# 智能旅行助手

基于 **LangGraph** 框架构建的多智能体旅行规划助手，集成高德地图 MCP Server 与 Unsplash 图片服务，通过 SSE 实时流式推送 Agent 执行进度，提供个性化旅行计划生成。

## 功能特点

- **LangGraph 多智能体协作**: 使用 `StateGraph` 编排景点搜索、天气查询、酒店推荐、行程规划四个节点，天气与酒店节点**并行执行**，支持 5 种执行模式（sequential / parallel_react / parallel_direct / parallel_agent / parallel_cache）
- **Agent 间协作**: `parallel_agent` 模式中酒店 Agent 读取景点 Agent 输出推荐附近酒店，天气 Agent 输出室内/室外活动建议
- **五模式 Benchmark 测评**: 5 种执行模式对比，三层指标体系（可靠性 / 速度 / 质量），13 项规则评测 + 3 项协作指标，20 条请求自动化测评，断点续传
- **Amap MCP Server 解耦**: 独立 MCP Server 进程封装高德地图 API（search_poi / get_weather / geocode / plan_route / get_poi_detail），通过 stdio 协议与 Agent 通信
- **ReAct 工具调用 + 重试**: Sequential 模式基于 `create_react_agent`，parallel_direct 模式直接 `tool.ainvoke()`（零 LLM 开销），parallel_agent 模式 tool + LLM 分析（平衡速度与质量），单节点失败自动降级不阻断管线
- **SSE 实时进度流**: 前端通过 Server-Sent Events 实时接收 Agent 各阶段进度，LLM 生成期间心跳保持连接
- **Redis 行程缓存**: 缓存层位于路由层，相同城市+日期+偏好的请求命中缓存后跳过全部 Agent 执行（TTL 24h）；`REDIS_URL` 为空时自动禁用
- **结构化日志 + 请求追踪**: 全链路 `logging` 模块，每个请求携带唯一 `request_id`
- **接口限流 + 输入校验**: 内存级 Rate Limiting（默认每 IP 每分钟 10 次），Pydantic 字段级安全校验
- **Unsplash 景点图片**: 批量异步获取景点真实照片，串行 + 延迟策略规避速率限制
- **现代化前端**: Vue 3 + TypeScript + Vite + Ant Design Vue，支持行程编辑、地图展示、PNG/PDF 导出

## 技术栈

### 后端
- **多智能体框架**: LangGraph (`StateGraph` + `create_react_agent` / 直接 tool.ainvoke / tool + LLM 分析协作)
- **LLM 接入**: LangChain / langchain-openai (`ChatOpenAI`，兼容 OpenAI 接口格式)
- **MCP 工具适配**: langchain-mcp-adapters + 自建 Amap MCP Server (FastMCP)
- **API 服务**: FastAPI + Uvicorn（lifespan 生命周期管理）
- **缓存**: Redis（行程结果精确缓存，`REDIS_URL` 为空时自动禁用）
- **中间件**: RequestID 追踪、Rate Limiting、CORS
- **测试**: pytest + pytest-asyncio（单元测试 + 集成测试）

### 前端
- **框架**: Vue 3 + TypeScript
- **构建工具**: Vite
- **UI 组件库**: Ant Design Vue
- **地图服务**: 高德地图 JavaScript API
- **HTTP 客户端**: Axios + Fetch (SSE)
- **导出**: html2canvas + jsPDF

### MCP Server
- **框架**: FastMCP (mcp[cli])
- **通信**: stdio 协议
- **数据源**: 高德地图 REST API (httpx)

## LangGraph 工作流

```
用户请求
   │
   ▼
search_attractions    ──→    weather_and_hotels    ──→    generate_plan
(直接 tool.ainvoke)          ┌─ query_weather ─┐          (ChatLLM)
 MCP: search_poi            │  MCP: get_weather │          整合生成 JSON
                            ├─ search_hotels ──┤
                            │  MCP: search_poi  │
                            └─────────────────┘
                              ↑ 并行执行 ↑
```

三阶段管线：景点搜索 → 天气+酒店（并行） → 行程规划。数据采集节点直接调用 MCP tool（无 LLM 推理开销），仅 generate_plan 使用 LLM，共享 `TripPlannerState`。

## 项目结构

```
trip-planner/
├── .env                        # 全局环境变量（不提交，参考 .env.example）
├── .env.example                # 配置示例
├── mcp_servers.json            # MCP Server 声明（兼容 Claude Desktop / Cursor 格式）
├── diagnose_llm.py             # LLM API 健康诊断脚本（延迟 / 限流 / 稳定性）
├── backend/                    # 后端服务
│   ├── run.py                  # 入口脚本
│   ├── app/
│   │   ├── agents/
│   │   │   └── trip_planner_agent.py   # LangGraph StateGraph + 5 种执行模式
│   │   ├── api/
│   │   │   ├── main.py                 # FastAPI lifespan + 中间件
│   │   │   └── routes/
│   │   │       └── trip.py             # 旅行规划 / SSE / 图片端点 + 缓存路由
│   │   ├── services/
│   │   │   ├── llm_service.py          # ChatOpenAI 单例
│   │   │   ├── photo_service.py        # Unsplash 图片搜索
│   │   │   ├── cache.py                # Redis 行程缓存（路由层，精确匹配 + TTL）
│   │   │   └── progress.py             # SSE 进度事件管理
│   │   ├── models/
│   │   │   └── schemas.py              # Pydantic 数据模型 + 输入校验
│   │   └── config.py                   # 配置管理 + 日志初始化
│   └── tests/                          # 单元测试 + 集成测试
│       ├── test_schemas.py             # Pydantic 输入校验（15 个 case）
│       ├── test_cache.py               # Redis 缓存逻辑（12 个 case）
│       ├── test_rules.py               # Benchmark 规则评测函数（34 个 case）
│       └── test_agent_integration.py   # Agent 集成测试（4 个 case）
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── services/api.ts             # API 客户端 + SSE 流解析
│   │   ├── types/index.ts              # TypeScript 类型定义
│   │   └── views/
│   │       ├── Home.vue                # 表单 + SSE 进度条
│   │       └── Result.vue              # 行程展示 / 编辑 / 导出
│   ├── package.json
│   └── vite.config.ts
├── mcp-server/                 # 独立 Amap MCP Server
│   └── server.py               # FastMCP + 高德 REST API（5 个工具）
├── benchmark/                  # Benchmark 测评框架
│   ├── benchmark_trip.py       # 测评 Harness（五模式 / 三层指标 / 断点续传）
│   └── requests_20.json        # 20 条固定测试数据集
└── benchmark.md                # Benchmark 方案与结果分析文档
```

## 快速开始

### 前提条件

- Python 3.10+
- Node.js 18+
- 高德地图 API 密钥（Web 服务 API）
- LLM API 密钥（OpenAI / DeepSeek / SiliconFlow 等 OpenAI 兼容接口）
- Redis（可选，用于行程缓存）
- Unsplash Access Key（可选，用于景点图片）

### 环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

关键字段：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o
AMAP_API_KEY=your_amap_api_key
REDIS_URL=redis://localhost:6379   # 留空则禁用缓存
PORT=8001
```

### 后端启动

```bash
cd backend
pip install -r ../requirements.txt
python run.py
```

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`

## API 文档

启动后端后访问 `http://localhost:8001/docs`

主要端点：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/trip/plan` | POST | 同步生成旅行计划，支持 `execution_mode` 参数 |
| `/api/trip/plan/stream` | POST | SSE 流式推送，推荐前端使用 |
| `/api/trip/photos` | POST | 批量获取景点 Unsplash 图片 |
| `/api/trip/health` | GET | 服务健康检查 |

`execution_mode` 可选值：`sequential` / `parallel_react` / `parallel_direct` / `parallel_agent` / `parallel_cache`（默认）

## 工业级特性

| 特性 | 实现 |
|---|---|
| **结构化日志** | Python `logging`，统一格式 `时间 \| 级别 \| 模块 \| 消息` |
| **请求追踪** | `RequestIDMiddleware`，每个请求分配唯一 ID，贯穿全链路 |
| **接口限流** | `RateLimitMiddleware`，同一 IP 每分钟 N 次（可通过 `RATE_LIMIT_PER_MINUTE` 配置） |
| **Agent 并行化** | 天气+酒店节点 `asyncio.gather` 并行，`Semaphore` 控制 MCP 并发 |
| **节点级容错** | 单节点失败自动降级，不阻断整条管线 |
| **MCP 单例连接** | MCP 子进程全局复用，崩溃自动重连 |
| **LLM JSON 强制输出** | `response_format: json_object`，解析失败后 LLM 修复重试 |
| **Redis 行程缓存** | 路由层精确匹配（TTL 24h），命中时跳过全部 Agent；`REDIS_URL` 为空自动禁用 |
| **SSE 实时进度** | Agent 执行到 UI 进度条全链路闭环，心跳保持连接 |
| **单元测试** | pytest 覆盖 Pydantic 校验 / Redis 缓存 / 规则评测函数 / Agent 集成（65 个 case）|
| **Benchmark 框架** | 五模式对比，三层指标体系，HTTP 429 指数退避重试，断点续传，结构化日志 |

## Benchmark 测评

内置五模式对比测评框架，从三个维度评估不同多智能体架构。

### 五种测评模式

| 模式 | Agent 结构 | 缓存 | 核心特点 |
|---|---|---|---|
| `sequential` | 单 ReAct Agent | ❌ | 持有全部 MCP tools，自主决定调用顺序 |
| `parallel_react` | 3 个独立 ReAct Agent | ❌ | 景点/天气/酒店各自推理，天气‖酒店并行 |
| `parallel_direct` | 直接 tool.ainvoke | ❌ | 零 LLM 推理开销，确定性工具调用 |
| `parallel_agent` | tool + LLM 分析 + Agent 协作 | ❌ | 酒店 Agent 参考景点输出推荐附近区域 |
| `parallel_cache` | parallel_agent + Redis | ✅ | 预热写缓存，统计轮直接返回（< 15ms） |

### 三层指标体系

Benchmark 指标分三层，fallback 记录（LLM 失败时的降级输出）不污染质量和速度数字：

**第一层：可靠性（所有请求均计入）**

| 指标 | 定义 |
|---|---|
| `llm_work_rate` | `(success 且非 fallback) / total`，核心指标 |
| `success_rate` | HTTP 200 且 success=True（含 fallback） |
| `fallback_count` | fallback 触发次数 |
| `error_429_count` | 限流失败次数（区分 429 vs LLM 失败） |

**第二层：速度（按结果拆分）**

| 指标 | 定义 |
|---|---|
| `p50/p95_llm_ms` | 非 fallback 成功记录的延迟分位数 |
| `avg_fallback_ms` | fallback 记录的平均延迟（"等待失败"的时间） |
| `avg_latency_ms` | 全部记录均值（整体吞吐参考） |

**第三层：计划质量（仅 LLM 成功子集，标注 n= 样本量）**

| 指标 | 定义 |
|---|---|
| `llm_rule_pass_rate` | 13 项规则通过率（合法性/完整性/约束/一致性/协作质量） |
| `pref_match_ratio` | 景点匹配用户偏好关键词的比例 |
| `avg_hotel_km` | 酒店到最近景点的平均 haversine 距离（km） |
| `rainy_indoor_rate` | 雨天安排室内景点的天数占比 |

### 测试数据集

20 条请求（15 独立 + 5 重复），覆盖 15 个城市、1~5 天行程、多种偏好标签。重复请求（#16~#20 = #1/#3/#5/#7/#10）用于验证缓存命中率。

### 运行方式

```bash
# 前置：启动后端和 Redis
brew services start redis
cd backend && python run.py

# 推荐：用 caffeinate 防止 macOS 休眠中断测评
caffeinate -i python benchmark/benchmark_trip.py --base-url http://localhost:8001

# 仅运行指定模式
python benchmark/benchmark_trip.py --base-url http://localhost:8001 --modes sequential parallel_direct

# 断点续传
python benchmark/benchmark_trip.py --base-url http://localhost:8001 --resume benchmark/results/<timestamp>
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--base-url` | `http://localhost:8001` | 后端地址 |
| `--timeout` | 600 | 单条请求超时（秒） |
| `--delay` | 3.0 | 请求间隔（秒），避免触发限流 |
| `--modes` | 全部 5 种 | 指定测试模式 |
| `--resume` | — | 从指定目录断点续传 |

输出到 `benchmark/results/{timestamp}/`：
- `report.md` — 三层指标 Markdown 报告
- `raw_results.jsonl` — 每条请求原始数据
- `benchmark.log` — 完整 DEBUG 日志

详细方案见 [benchmark.md](benchmark.md)。

## 单元测试

```bash
cd backend
pytest tests/ -v
```

覆盖范围：

| 测试文件 | 覆盖内容 | case 数 |
|---|---|---|
| `test_schemas.py` | Pydantic 输入校验（城市、日期、天数、XSS 过滤） | 15 |
| `test_cache.py` | Redis 缓存键生成、命中、降级 | 12 |
| `test_rules.py` | Benchmark 13 项规则评测函数（haversine 距离、天气匹配） | 34 |
| `test_agent_integration.py` | parallel_direct / fallback / MCP 工具调用 | 4 |

## LLM API 诊断

遇到 LLM 响应慢或超时，可用诊断脚本排查：

```bash
python diagnose_llm.py
```

输出：单次延迟、连发 3 次稳定性、限流检测（429 / 超时 / 连接失败分类）。

## 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — 多智能体编排框架
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用开发框架
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) — MCP Server 框架
- [高德地图开放平台](https://lbs.amap.com/) — 地图数据与路线规划
- [Unsplash](https://unsplash.com/) — 景点图片
