"""
LangGraph 多智能体旅行规划系统
================================
MCP Server 配置从项目根目录 mcp_servers.json 读取，
格式与 Claude Desktop / Cursor 对齐。

数据流:
  TripRequest → mcp_servers.json → MultiServerMCPClient
              → LangGraph (LLM + MCP Tools) → TripPlan

LangGraph 节点:
  search_attractions → [query_weather // search_hotels] → generate_plan
                        (并行执行)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any, List

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage

from ..services.llm_service import get_llm
from ..services.progress import ProgressEmitter, ProgressStep
from ..models.schemas import (
    TripRequest, TripPlan, DayPlan, Attraction,
    Meal, WeatherInfo, Location, Hotel,
)


# 项目根目录： backend/app/agents/ → backend/app/ → backend/ → trip-planner/
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_MCP_SERVERS_JSON = _PROJECT_ROOT / "mcp_servers.json"


def _load_mcp_client() -> MultiServerMCPClient:
    """
    读取 mcp_servers.json，将标准 mcpServers 格式转换为
    MultiServerMCPClient 小期含的注册形式。

    支持的占位符（将在运行时替换）:
      ${变量名}  → os.environ 中对应的值
      ${PYTHON}    → sys.executable（当前运行时的 Python）

    args 中的相对路径基于项目根目录解析。
    """
    if not _MCP_SERVERS_JSON.exists():
        raise FileNotFoundError(f"mcp_servers.json 未找到: {_MCP_SERVERS_JSON}")

    raw = json.loads(_MCP_SERVERS_JSON.read_text(encoding="utf-8"))
    servers_raw: Dict[str, Any] = raw.get("mcpServers", {})

    def _sub(value: str) -> str:
        """${PYTHON} 或 ${ENV_VAR} 替换"""
        if "${PYTHON}" in value:
            value = value.replace("${PYTHON}", sys.executable)
        # 替换其余 ${VAR}
        import re
        for match in re.findall(r"\$\{([^}]+)\}", value):
            env_val = os.environ.get(match, "")
            value = value.replace(f"${{{match}}}", env_val)
        return value

    servers: Dict[str, Any] = {}
    for name, cfg in servers_raw.items():
        command = _sub(cfg.get("command", ""))

        # 转换 args：相对路径基于项目根目录
        args = []
        for arg in cfg.get("args", []):
            resolved = _sub(arg)
            p = Path(resolved)
            if not p.is_absolute():
                resolved = str(_PROJECT_ROOT / p)
            args.append(resolved)

        # env 替换
        env = {k: _sub(v) for k, v in cfg.get("env", {}).items()}

        # transport 默认 stdio
        transport = cfg.get("transport", "stdio")

        servers[name] = {
            "command": command,
            "args": args,
            "env": env,
            "transport": transport,
        }

    logger.info("加载 MCP 配置: %s，共 %d 个 server: %s", _MCP_SERVERS_JSON.name, len(servers), list(servers))
    return MultiServerMCPClient(servers)


# ============================================================
# Prompts
# ============================================================

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。
请使用 search_poi 工具搜索目标城市的景点信息（如博物馆、公园、古迹等）。
搜索完毕后，汇总景点名称、地址、坐标等关键信息供后续规划使用。
不要编造任何景点信息，必须基于工具返回的真实数据。"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。
请使用 get_weather 工具查询目标城市的天气预报。
汇总每天的天气、气温、风向等信息供行程规划参考。
不要编造天气数据，必须基于工具返回的真实结果。"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。
请使用 search_poi 工具搜索目标城市的酒店信息。
汇总酒店名称、地址、价格区间、评分等信息供行程规划参考。
不要编造酒店信息，必须基于工具返回的真实数据。"""

PLANNER_SYSTEM_PROMPT = "你是行程规划专家。只返回 JSON，不要任何额外文字或解释。"

PLANNER_OUTPUT_SCHEMA = '''{"city":"","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD",
 "days":[{"date":"YYYY-MM-DD","day_index":0,"description":"","transportation":"","accommodation":"",
   "hotel":{"name":"","address":"","location":{"longitude":0,"latitude":0},"price_range":"","rating":"","distance":"","type":"","estimated_cost":0},
   "attractions":[{"name":"","address":"","location":{"longitude":0,"latitude":0},"visit_duration":120,"description":"","category":"","ticket_price":0}],
   "meals":[{"type":"breakfast|lunch|dinner","name":"","description":"","estimated_cost":0}]}],
 "weather_info":[{"date":"YYYY-MM-DD","day_weather":"","night_weather":"","day_temp":25,"night_temp":15,"wind_direction":"","wind_power":""}],
 "overall_suggestions":"",
 "budget":{"total_attractions":0,"total_hotels":0,"total_meals":0,"total_transportation":0,"total":0}}
要求:1.每天 2-3 个景点+早中晚三餐+一个酒店 2.天气temperature为纯数字 3.经纬度必须真实准确 4.必须包含完整预算'''


# 上下文字数上限（截断前几个 agent 返回的信息，减少 input token）
_CONTEXT_MAX_CHARS = 1500
# 节点级重试次数
_NODE_MAX_RETRIES = 2


# ============================================================
# LangGraph State
# ============================================================

class TripPlannerState(TypedDict):
    request: TripRequest
    attractions_info: str
    weather_info: str
    hotels_info: str
    trip_plan: Optional[TripPlan]


# ============================================================
# Graph Builder
# ============================================================

def _extract_last_message(result: Dict[str, Any]) -> str:
    """从 ReAct agent 输出中提取最后一条 AI 消息的文本"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _build_langgraph(tools: List, progress_emitter: Optional[ProgressEmitter] = None) -> Any:
    """构建并编译 LangGraph，支持进度回调、节点级容错与重试、天气/酒店并行执行"""
    llm = get_llm()

    attraction_agent = create_react_agent(llm, tools, prompt=ATTRACTION_AGENT_PROMPT)
    weather_agent    = create_react_agent(llm, tools, prompt=WEATHER_AGENT_PROMPT)
    hotel_agent      = create_react_agent(llm, tools, prompt=HOTEL_AGENT_PROMPT)

    async def _invoke_with_retry(agent, query: str, timeout: int, node_name: str) -> str:
        """带指数退避重试的 agent 调用"""
        last_err = None
        for attempt in range(1, _NODE_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": [HumanMessage(content=query)]}),
                    timeout=timeout,
                )
                return _extract_last_message(result)
            except asyncio.TimeoutError:
                last_err = f"超时({timeout}s)"
                logger.warning("[%s] 第%d次尝试超时(%ds)", node_name, attempt, timeout)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e!r}"
                logger.warning("[%s] 第%d次尝试失败: %s", node_name, attempt, last_err)
            if attempt < _NODE_MAX_RETRIES:
                backoff = 2 ** attempt
                logger.info("[%s] %ds后重试...", node_name, backoff)
                await asyncio.sleep(backoff)
        return ""  # 所有重试耗尽，返回空串触发降级

    async def search_attractions(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.SEARCH_ATTRACTIONS)
        req = state["request"]
        keyword = req.preferences[0] if req.preferences else "景点"
        query = f"请搜索 {req.city} 的 {keyword} 相关景点，关键词={keyword}，城市={req.city}"
        logger.info("[搜索景点] %s", query)
        info = await _invoke_with_retry(attraction_agent, query, 120, "搜索景点")
        if not info:
            info = f"{req.city}的热门景点信息（工具调用失败，请根据已有知识推荐）"
            logger.warning("[搜索景点] 所有重试耗尽，使用降级策略")
        logger.debug("[搜索景点] 结果(前300字): %s", info[:300])
        return {"attractions_info": info}

    async def query_weather(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.QUERY_WEATHER)
        req = state["request"]
        query = f"请查询 {req.city} 的天气预报"
        logger.info("[查询天气] %s", query)
        info = await _invoke_with_retry(weather_agent, query, 120, "查询天气")
        if not info:
            info = f"{req.city}近期天气预报（工具调用失败，请根据季节常识推荐）"
            logger.warning("[查询天气] 所有重试耗尽，使用降级策略")
        logger.debug("[查询天气] 结果(前300字): %s", info[:300])
        return {"weather_info": info}

    async def search_hotels(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.SEARCH_HOTELS)
        req = state["request"]
        query = f"请搜索 {req.city} 的 {req.accommodation} 酒店"
        logger.info("[搜索酒店] %s", query)
        info = await _invoke_with_retry(hotel_agent, query, 120, "搜索酒店")
        if not info:
            info = f"{req.city}的{req.accommodation}推荐（工具调用失败，请根据已有知识推荐）"
            logger.warning("[搜索酒店] 所有重试耗尽，使用降级策略")
        logger.debug("[搜索酒店] 结果(前300字): %s", info[:300])
        return {"hotels_info": info}

    async def weather_and_hotels(state: TripPlannerState) -> Dict:
        """并行执行天气查询和酒店搜索，合并结果"""
        weather_task = asyncio.create_task(query_weather(state))
        hotel_task = asyncio.create_task(search_hotels(state))
        weather_result, hotel_result = await asyncio.gather(weather_task, hotel_task)
        return {**weather_result, **hotel_result}

    async def generate_plan(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.GENERATE_PLAN)
        req = state["request"]

        # 截断上下文，控制 input token 规模
        attr_ctx  = state['attractions_info'][:_CONTEXT_MAX_CHARS]
        wthr_ctx  = state['weather_info'][:_CONTEXT_MAX_CHARS]
        hotel_ctx = state['hotels_info'][:_CONTEXT_MAX_CHARS]

        user_content = f"""生成 {req.city} {req.travel_days} 天旅行计划:
城市={req.city} 日期={req.start_date}~{req.end_date} 天数={req.travel_days}
交通={req.transportation} 住宿={req.accommodation}
偏好={', '.join(req.preferences) if req.preferences else '无'}
{f'额外要求: {req.free_text_input}' if req.free_text_input else ''}

景点: {attr_ctx}
天气: {wthr_ctx}
酒店: {hotel_ctx}

输出格式（严格遵守，只返回 JSON 不要其他文字）:
{PLANNER_OUTPUT_SCHEMA}"""
        logger.info("[生成行程] 调用规划 LLM（上下文约 %d 字）", len(user_content))

        # 心跳任务：在等待 LLM 生成时定期推送进度，避免前端卡死在 80%
        heartbeat_task = None
        if progress_emitter:
            heartbeat_msgs = [
                "✨ AI 正在组织每日行程...",
                "📝 正在安排景点参观顺序...",
                "🍴 正在筛选餐饮推荐...",
                "💰 正在计算行程预算...",
                "🔄 AI 深度规划中，请稍候...",
            ]
            async def _heartbeat():
                try:
                    for i, msg in enumerate(heartbeat_msgs):
                        await asyncio.sleep(8)
                        pct = min(80 + (i + 1) * 2, 90)
                        await progress_emitter.emit(ProgressStep.GENERATE_PLAN, msg, percent_override=pct)
                    while True:
                        await asyncio.sleep(15)
                        await progress_emitter.emit(ProgressStep.GENERATE_PLAN, "⏳ AI 正在生成完整行程 JSON...", percent_override=91)
                except asyncio.CancelledError:
                    pass
            heartbeat_task = asyncio.create_task(_heartbeat())

        try:
            llm_inst = get_llm()
            response = await asyncio.wait_for(
                llm_inst.ainvoke([
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]),
                timeout=300,
            )
            plan_text = response.content
            logger.debug("[生成行程] 规划结果(前300字): %s", plan_text[:300])
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.PARSE_RESULT)
            trip_plan = _parse_plan(plan_text, req)
        except asyncio.TimeoutError:
            logger.warning("[生成行程] LLM 调用超时(300s)，使用备用方案")
            trip_plan = _fallback_plan(req)
        except Exception as e:
            logger.warning("[生成行程] 规划失败 [%s]: %r，使用备用方案", type(e).__name__, e)
            trip_plan = _fallback_plan(req)
        finally:
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
        return {"trip_plan": trip_plan}

    workflow = StateGraph(TripPlannerState)
    workflow.add_node("search_attractions", search_attractions)
    workflow.add_node("weather_and_hotels", weather_and_hotels)
    workflow.add_node("generate_plan",      generate_plan)

    workflow.set_entry_point("search_attractions")
    workflow.add_edge("search_attractions", "weather_and_hotels")
    workflow.add_edge("weather_and_hotels", "generate_plan")
    workflow.add_edge("generate_plan",      END)

    return workflow.compile()


# ============================================================
# Helpers
# ============================================================

def _parse_plan(response: str, request: TripRequest) -> TripPlan:
    """从 LLM 响应中解析 TripPlan"""
    try:
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end   = response.find("```", json_start)
            json_str   = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end   = response.find("```", json_start)
            json_str   = response[json_start:json_end].strip()
        elif "{" in response and "}" in response:
            json_start = response.find("{")
            json_end   = response.rfind("}") + 1
            json_str   = response[json_start:json_end]
        else:
            raise ValueError("响应中未找到 JSON 数据")

        data = json.loads(json_str)
        return TripPlan(**data)
    except Exception as e:
        logger.warning("解析行程失败: %s，使用备用计划", e)
        return _fallback_plan(request)


def _fallback_plan(request: TripRequest) -> TripPlan:
    """备用计划（当 LLM 或解析失败时）"""
    from datetime import datetime, timedelta

    start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    days = []
    for i in range(request.travel_days):
        current_date = start_date + timedelta(days=i)
        days.append(DayPlan(
            date=current_date.strftime("%Y-%m-%d"),
            day_index=i,
            description=f"第{i+1}天行程",
            transportation=request.transportation,
            accommodation=request.accommodation,
            attractions=[
                Attraction(
                    name=f"{request.city}景点{j+1}",
                    address=f"{request.city}市",
                    location=Location(
                        longitude=116.4 + i * 0.01 + j * 0.005,
                        latitude=39.9  + i * 0.01 + j * 0.005,
                    ),
                    visit_duration=120,
                    description=f"{request.city}著名景点",
                    category="景点",
                )
                for j in range(2)
            ],
            meals=[
                Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                Meal(type="lunch",     name=f"第{i+1}天午餐", description="午餐推荐"),
                Meal(type="dinner",    name=f"第{i+1}天晚餐", description="晚餐推荐"),
            ],
        ))

    return TripPlan(
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        days=days,
        weather_info=[],
        overall_suggestions=(
            f"为您规划的{request.city}{request.travel_days}日游行程，"
            "建议提前查看各景点开放时间。"
        ),
    )


# ============================================================
# Main Planner
# ============================================================

class LangGraphTripPlanner:
    """
    基于 LangGraph 的多智能体旅行规划系统。

    MCP Server 配置从项目根目录 mcp_servers.json 读取，
    格式与 Claude Desktop / Cursor 对齐，无需改代码即可切换工具。
    """

    def __init__(self):
        logger.info("LangGraph 旅行规划系统初始化 | MCP 配置=%s", _MCP_SERVERS_JSON)

    async def plan_trip(self, request: TripRequest, progress_emitter: Optional[ProgressEmitter] = None) -> TripPlan:
        """异步执行多智能体协作，生成旅行计划，支持 SSE 进度回调"""
        logger.info("LangGraph 旅行规划启动 | 目的地=%s | 日期=%s~%s",
                    request.city, request.start_date, request.end_date)

        if progress_emitter:
            await progress_emitter.emit(ProgressStep.INIT)

        mcp_client = None
        try:
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.MCP_CONNECT)

            mcp_client = _load_mcp_client()

            tools = await mcp_client.get_tools()
            logger.info("MCP 工具加载完成，共 %d 个: %s", len(tools), [t.name for t in tools])

            graph = _build_langgraph(tools, progress_emitter)
            initial_state: TripPlannerState = {
                "request":          request,
                "attractions_info": "",
                "weather_info":     "",
                "hotels_info":      "",
                "trip_plan":        None,
            }
            final_state = await graph.ainvoke(initial_state)

            trip_plan = final_state.get("trip_plan") or _fallback_plan(request)

        except Exception as e:
            logger.error("Agent 管线异常: %s", e, exc_info=True)
            trip_plan = _fallback_plan(request)

        finally:
            # 最佳努力清理 MCP 客户端（关闭子进程连接）
            if mcp_client is not None:
                for method_name in ("close", "aclose", "disconnect"):
                    cleanup = getattr(mcp_client, method_name, None)
                    if cleanup and callable(cleanup):
                        try:
                            ret = cleanup()
                            if asyncio.iscoroutine(ret) or asyncio.isfuture(ret):
                                await ret
                        except Exception:
                            pass
                        break

        logger.info("旅行计划生成完成!")
        return trip_plan


# ============================================================
# Singleton
# ============================================================

_planner_instance: Optional[LangGraphTripPlanner] = None


def get_trip_planner_agent() -> LangGraphTripPlanner:
    """获取 LangGraph 旅行规划系统单例"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = LangGraphTripPlanner()
    return _planner_instance
