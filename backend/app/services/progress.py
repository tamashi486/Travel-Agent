"""
进度事件管理模块
================
基于 asyncio.Queue 的进度事件系统，供 Agent 节点向 SSE 端点推送实时进度。

使用方式:
    emitter = ProgressEmitter()
    await emitter.emit(ProgressStep.SEARCH_ATTRACTIONS)
    ...
    await emitter.emit_done(trip_plan_dict)
"""

import asyncio
import json
from typing import Any
from enum import Enum


class ProgressStep(str, Enum):
    """Agent 执行步骤枚举"""
    INIT = "init"
    MCP_CONNECT = "mcp_connect"
    SEARCH_ATTRACTIONS = "search_attractions"
    QUERY_WEATHER = "query_weather"
    SEARCH_HOTELS = "search_hotels"
    GENERATE_PLAN = "generate_plan"
    STREAM_CHUNK = "stream_chunk"  # LLM 流式输出块
    PARSE_RESULT = "parse_result"
    DONE = "done"
    ERROR = "error"


# 每个步骤对应的默认百分比和提示信息
STEP_CONFIG = {
    ProgressStep.INIT:                {"percent": 5,   "message": "🚀 正在初始化AI规划引擎..."},
    ProgressStep.MCP_CONNECT:         {"percent": 10,  "message": "🔌 正在连接地图数据服务..."},
    ProgressStep.SEARCH_ATTRACTIONS:  {"percent": 25,  "message": "🔍 正在搜索目的地景点信息..."},
    ProgressStep.QUERY_WEATHER:       {"percent": 45,  "message": "🌤️ 正在获取天气预报数据..."},
    ProgressStep.SEARCH_HOTELS:       {"percent": 65,  "message": "🏨 正在查找合适的住宿推荐..."},
    ProgressStep.GENERATE_PLAN:       {"percent": 80,  "message": "✍️ AI正在生成详细行程规划..."},
    ProgressStep.PARSE_RESULT:        {"percent": 92,  "message": "📋 正在整理行程数据..."},
    ProgressStep.DONE:                {"percent": 100, "message": "✅ 旅行计划生成完成!"},
    ProgressStep.ERROR:               {"percent": -1,  "message": "❌ 生成失败"},
}


class ProgressEmitter:
    """
    进度事件发射器。

    Agent 各节点通过 emit() 推送进度，SSE handler 从 queue 消费事件。
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    async def emit(self, step: ProgressStep, detail: str = "", percent_override: int = -1):
        """发射一个进度事件，percent_override > 0 时覆盖默认百分比"""
        cfg = STEP_CONFIG.get(step, {"percent": 0, "message": ""})
        pct = percent_override if percent_override > 0 else cfg["percent"]
        await self.queue.put({
            "step": step.value,
            "percent": pct,
            "message": detail or cfg["message"],
        })

    async def emit_stream_chunk(self, text: str):
        """发射 LLM 流式 token 块，供前端实时展示生成过程"""
        await self.queue.put({
            "step": ProgressStep.STREAM_CHUNK.value,
            "percent": 85,
            "message": "✨ AI 规划中...",
            "streamText": text,
        })

    async def emit_done(self, data: Any):
        """发射完成事件（携带最终数据）"""
        await self.queue.put({
            "step": ProgressStep.DONE.value,
            "percent": 100,
            "message": STEP_CONFIG[ProgressStep.DONE]["message"],
            "data": data,
        })

    async def emit_error(self, error: str):
        """发射错误事件"""
        await self.queue.put({
            "step": ProgressStep.ERROR.value,
            "percent": -1,
            "message": f"❌ {error}",
            "error": error,
        })
