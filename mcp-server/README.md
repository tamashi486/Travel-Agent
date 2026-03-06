# Amap MCP Server

基于 [FastMCP](https://github.com/jlowin/fastmcp) 构建的高德地图 MCP 工具服务。  
通过 **stdio** 协议与 LangGraph Agent 通信，实现 Amap API 与 Agent 的彻底解耦。

## 架构位置

```
frontend (Vue) ──HTTP──> backend (FastAPI + LangGraph) ──MCP stdio──> mcp-server ──REST──> 高德地图 API
```

## 提供的工具

| 工具名 | 说明 |
|---|---|
| `search_poi` | 按关键词搜索 POI（景点、餐厅、酒店等） |
| `get_weather` | 查询城市未来天气预报 |
| `geocode` | 地址转经纬度坐标 |
| `plan_route` | 步行 / 驾车 / 公交路线规划 |
| `get_poi_detail` | 根据 POI ID 获取详细信息 |

## 快速开始

### 1. 安装依赖

```bash
cd mcp-server
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入高德 API Key
```

### 3. 直接运行（调试用）

```bash
python server.py
```

服务启动后通过 stdin/stdout 接收 MCP 协议消息。正常情况下由 backend agent 自动以子进程方式启动。

## 被 Backend 调用方式

`backend/app/agents/trip_planner_agent.py` 通过 `MultiServerMCPClient` 以 stdio 方式启动本服务：

```python
MultiServerMCPClient({
    "amap": {
        "command": "python",
        "args": ["/path/to/mcp-server/server.py"],
        "env": {"AMAP_API_KEY": "..."},
        "transport": "stdio",
    }
})
```

`AMAP_API_KEY` 由 backend `.env` 统一管理，运行时通过子进程 env 传入，无需在 mcp-server 单独配置。
