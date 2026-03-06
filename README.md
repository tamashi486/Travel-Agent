# 智能旅行助手 🌍✈️

基于 **LangGraph** 框架构建的多智能体旅行规划助手，集成高德地图 MCP Server 与 Unsplash 图片服务，通过 SSE 实时流式推送 Agent 执行进度，提供个性化旅行计划生成。

## ✨ 功能特点

- 🤖 **LangGraph 多智能体协作**: 使用 `StateGraph` 编排景点搜索、天气查询、酒店推荐、行程规划四个专业 Agent，天气与酒店节点**并行执行**
- 🗺️ **Amap MCP Server 解耦**: 独立 MCP Server 进程封装高德地图 API（search_poi / get_weather / geocode / plan_route / get_poi_detail），通过 stdio 协议与 Agent 通信
- 🧠 **ReAct 工具调用 + 重试**: 每个 Agent 节点基于 `create_react_agent`，支持指数退避重试，单节点失败自动降级不阻断管线
- 📡 **SSE 实时进度流**: 前端通过 Server-Sent Events 实时接收 Agent 各阶段进度，LLM 生成期间心跳保持连接
- ⚡ **Redis 行程缓存**: 缓存层位于路由层，相同城市+日期+偏好的请求命中缓存后跳过全部 Agent 执行（TTL 24h）；`REDIS_URL` 为空时自动禁用
- 📊 **结构化日志 + 请求追踪**: 全链路 `logging` 模块替代 print，每个请求携带唯一 `request_id`
- 🛡️ **接口限流 + 输入校验**: 内存级 Rate Limiting（每 IP 每分钟 10 次规划请求），Pydantic 字段级安全校验
- 📸 **Unsplash 景点图片**: 批量异步获取景点真实照片，串行 + 延迟策略规避速率限制
- 🎨 **现代化前端**: Vue 3 + TypeScript + Vite + Ant Design Vue，支持行程编辑、地图展示、PNG/PDF 导出

## 🏗️ 技术栈

### 后端
- **多智能体框架**: LangGraph (`StateGraph` + `create_react_agent`)
- **LLM 接入**: LangChain / langchain-openai (`ChatOpenAI`，兼容 DeepSeek 等 OpenAI 格式接口)
- **MCP 工具适配**: langchain-mcp-adapters + 自建 Amap MCP Server (FastMCP)
- **API 服务**: FastAPI + Uvicorn（lifespan 生命周期管理）
- **缓存**: Redis（行程结果精确缓存，`REDIS_URL` 为空时自动禁用）
- **中间件**: RequestID 追踪、Rate Limiting、CORS
- **日志**: Python logging 结构化日志

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

## 🔄 LangGraph 工作流

```
用户请求
   │
   ▼
search_attractions    ──→    weather_and_hotels    ──→    generate_plan
(ReAct Agent)               ┌─ query_weather ─┐          (ChatOpenAI)
 MCP: search_poi            │  (ReAct Agent)  │          整合生成 JSON
                            │  MCP: get_weather│
                            ├─ search_hotels ──┤
                            │  (ReAct Agent)  │
                            │  MCP: search_poi │
                            └─────────────────┘
                              ↑ 并行执行 ↑
```

三阶段管线：景点搜索 → 天气+酒店（并行） → 行程规划（流式 token 输出），共享 `TripPlannerState`。

## 📁 项目结构

```
trip-planner/
├── .env                        # 全局环境变量（唯一配置源）
├── mcp_servers.json            # MCP Server 声明（兼容 Claude Desktop / Cursor 格式）
├── backend/                    # 后端服务
│   ├── run.py                  # 入口脚本
│   └── app/
│       ├── agents/
│       │   └── trip_planner_agent.py   # LangGraph StateGraph + 并行节点
│       ├── api/
│       │   ├── main.py                 # FastAPI lifespan + 中间件
│       │   └── routes/
│       │       └── trip.py             # 旅行规划 / SSE / 图片端点
│       ├── services/
│       │   ├── llm_service.py          # ChatOpenAI 单例
│       │   ├── photo_service.py        # Unsplash 图片搜索
│       │   ├── cache.py                # Redis 行程缓存（路由层调用，精确匹配 + TTL）
│       │   └── progress.py             # SSE 进度事件管理（含流式 token 事件）
│       ├── models/
│       │   └── schemas.py              # Pydantic 数据模型 + 输入校验
│       └── config.py                   # 配置管理 + 日志初始化
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── services/api.ts             # API 客户端 + SSE 流解析
│   │   ├── types/index.ts              # TypeScript 类型定义
│   │   └── views/
│   │       ├── Home.vue                # 表单 + SSE 进度条
│   │       └── Result.vue              # 行程展示 / 编辑 / 导出
│   ├── package.json
│   └── vite.config.ts
└── mcp-server/                 # 独立 Amap MCP Server
    └── server.py               # FastMCP + 高德 REST API
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 18+
- 高德地图 API 密钥（Web 服务 API Key + Web 端 JS API Key）
- LLM API 密钥（OpenAI / DeepSeek / SiliconFlow 等 OpenAI 兼容接口）
- Unsplash Access Key（可选，用于景点图片）

### 环境变量

在项目根目录创建 `.env` 文件：

```env
# 高德地图
AMAP_API_KEY=your_amap_api_key

# LLM
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o
LLM_TIMEOUT=300

# Redis 缓存（可选，为空则自动禁用）
REDIS_URL=redis://localhost:6379/0

# Unsplash（可选）
UNSPLASH_ACCESS_KEY=your_unsplash_key

# 日志级别
LOG_LEVEL=INFO
```

### 后端启动

```bash
cd backend
python run.py
```

### 前端启动

```bash
cd frontend
npm run dev
```

访问 `http://localhost:5173`

## 📝 使用指南

1. 在首页填写旅行信息（目的地、日期、交通、住宿、偏好标签）
2. 点击"生成旅行计划"，实时查看 SSE 进度
3. LangGraph 执行流程：
   - **景点搜索 Agent** → 调用 MCP `search_poi`
   - **天气查询 + 酒店搜索 Agent**（并行） → 调用 MCP `get_weather` / `search_poi`
   - **行程规划 Agent** → 整合信息生成完整 JSON 行程
4. 查看结果：每日行程、地图标记、天气预报、预算汇总
5. 支持行程编辑、PNG/PDF 导出

## 📄 API 文档

启动后端服务后，访问 `http://localhost:8000/docs` 查看完整 API 文档。

主要端点：
- `POST /api/trip/plan` — 生成旅行计划（同步）
- `POST /api/trip/plan/stream` — 生成旅行计划（SSE 流式，推荐）
- `POST /api/trip/photos` — 批量获取景点图片
- `GET /api/trip/health` — 服务健康检查

## 🔧 工业级特性

| 特性 | 实现 |
|------|------|
| **结构化日志** | Python `logging` 模块，统一格式 `时间 \| 级别 \| 模块 \| 消息` |
| **请求追踪** | `RequestIDMiddleware`，每个请求分配唯一 ID，贯穿全链路 |
| **接口限流** | `RateLimitMiddleware`，同一 IP 每分钟 10 次规划请求 |
| **Agent 并行化** | 天气查询与酒店搜索并行执行，减少总延迟 |
| **指数退避重试** | 每个 Agent 节点最多重试 2 次，间隔指数增长 |
| **节点级容错** | 单节点失败自动降级，不阻断整条管线 |
| **Redis 行程缓存** | 路由层精确匹配缓存（TTL 24h），命中时跳过全部 Agent 执行；`REDIS_URL` 为空自动禁用 |
| **SSE 实时进度** | Agent 执行到 UI 进度条的全链路闭环，心跳保持连接 |
| **MCP 配置解耦** | `mcp_servers.json` 标准格式，零代码切换工具 Server |

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — 多智能体编排框架
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用开发框架
- [FastMCP](https://github.com/modelcontextprotocol/python-sdk) — MCP Server 框架
- [高德地图开放平台](https://lbs.amap.com/) — 地图服务
- [Unsplash](https://unsplash.com/) — 景点图片

---

**智能旅行助手** - 让旅行计划变得简单而智能 🌈

