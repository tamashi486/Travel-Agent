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


def _parse_mcp_servers_config() -> Dict[str, Any]:
    """解析 mcp_servers.json 配置，返回 MultiServerMCPClient 构造参数"""
    if not _MCP_SERVERS_JSON.exists():
        raise FileNotFoundError(f"mcp_servers.json 未找到: {_MCP_SERVERS_JSON}")

    raw = json.loads(_MCP_SERVERS_JSON.read_text(encoding="utf-8"))
    servers_raw: Dict[str, Any] = raw.get("mcpServers", {})

    import re as _re

    def _sub(value: str) -> str:
        """${PYTHON} 或 ${ENV_VAR} 替换"""
        if "${PYTHON}" in value:
            value = value.replace("${PYTHON}", sys.executable)
        for match in _re.findall(r"\$\{([^}]+)\}", value):
            env_val = os.environ.get(match, "")
            value = value.replace(f"${{{match}}}", env_val)
        return value

    servers: Dict[str, Any] = {}
    for name, cfg in servers_raw.items():
        command = _sub(cfg.get("command", ""))
        args = []
        for arg in cfg.get("args", []):
            resolved = _sub(arg)
            p = Path(resolved)
            if not p.is_absolute():
                resolved = str(_PROJECT_ROOT / p)
            args.append(resolved)
        env = {k: _sub(v) for k, v in cfg.get("env", {}).items()}
        transport = cfg.get("transport", "stdio")
        servers[name] = {"command": command, "args": args, "env": env, "transport": transport}

    logger.info("加载 MCP 配置: %s，共 %d 个 server: %s", _MCP_SERVERS_JSON.name, len(servers), list(servers))
    return servers


# ============================================================
# MCP Client 单例（避免每次请求创建/销毁子进程）
# ============================================================

_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools: Optional[List] = None
_mcp_lock = asyncio.Lock()
# 控制 MCP 工具调用并发，避免 stdio 管道竞争
_mcp_semaphore = asyncio.Semaphore(2)


async def _get_mcp_tools() -> List:
    """获取 MCP 工具列表（单例，首次调用时创建 MCP 子进程）"""
    global _mcp_client, _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools
    async with _mcp_lock:
        if _mcp_tools is not None:
            return _mcp_tools
        try:
            servers = _parse_mcp_servers_config()
            _mcp_client = MultiServerMCPClient(servers)
            _mcp_tools = await _mcp_client.get_tools()
            logger.info("MCP 单例初始化完成，工具数: %d, 名称: %s",
                        len(_mcp_tools), [t.name for t in _mcp_tools])
        except Exception:
            _mcp_client = None
            _mcp_tools = None
            raise
        return _mcp_tools


async def _reset_mcp_client():
    """重置 MCP 单例（用于子进程崩溃后重建连接）"""
    global _mcp_client, _mcp_tools
    async with _mcp_lock:
        if _mcp_client is not None:
            for method_name in ("close", "aclose", "disconnect"):
                cleanup = getattr(_mcp_client, method_name, None)
                if cleanup and callable(cleanup):
                    try:
                        ret = cleanup()
                        if asyncio.iscoroutine(ret) or asyncio.isfuture(ret):
                            await ret
                    except Exception:
                        pass
                    break
        _mcp_client = None
        _mcp_tools = None
        logger.info("MCP 单例已重置")


async def shutdown_mcp_client():
    """应用关闭时清理 MCP 子进程（在 FastAPI lifespan 中调用）"""
    await _reset_mcp_client()


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

# 单 Agent 模式：一个 ReAct Agent 完成所有工具调用 + 行程规划
SINGLE_AGENT_PROMPT = f"""你是全能旅行规划师。请严格按以下步骤执行：

**Step 1 — 搜索景点**
使用 search_poi 工具搜索目标城市的景点（关键词如博物馆、公园、古迹等），至少获取 5 个景点。

**Step 2 — 查询天气**
使用 get_weather 工具查询目标城市的天气预报。

**Step 3 — 搜索酒店**
使用 search_poi 工具搜索目标城市的酒店（关键词"酒店"）。

**Step 4 — 生成行程 JSON**
基于 Step 1-3 获取的真实数据，生成完整的旅行计划。

重要规则：
- 不要编造任何信息，必须基于工具返回的真实数据
- 必须先完成全部 3 个工具调用步骤，再生成最终 JSON
- 最终只返回 JSON，不要任何额外文字、解释或 markdown 标记

输出格式（严格遵守）:
{PLANNER_OUTPUT_SCHEMA}"""

# ---- parallel_agent 模式：轻量 Agent 分析 Prompt ----

ATTRACTION_ANALYSIS_PROMPT = """根据用户偏好从以下景点数据中筛选推荐，输出纯文本摘要（不要 JSON）。
要求：
1. 按与用户偏好的相关度排序，标注推荐理由
2. 提取景点所在商圈/区域信息，方便后续酒店选址参考
3. 保留每个景点的名称、地址、经纬度坐标

用户偏好: {preferences}
城市: {city}

原始景点数据:
{raw_data}"""

WEATHER_ANALYSIS_PROMPT = """分析以下天气预报数据，输出每日天气摘要和活动建议（纯文本）。
要求：
1. 每天一行：日期 | 天气 | 气温 | 活动建议
2. 雨天标注"建议安排室内景点（博物馆、商场、展览馆等）"
3. 晴天标注"适合户外活动"

城市: {city}

原始天气数据:
{raw_data}"""

HOTEL_ANALYSIS_PROMPT = """根据景点位置信息推荐合适的酒店，输出纯文本摘要（不要 JSON）。
要求：
1. 优先推荐距离主要景点/商圈近的酒店
2. 标注每个酒店与景区的大致距离关系
3. 保留酒店名称、地址、经纬度坐标

住宿偏好: {accommodation}
景点位置参考（来自景点Agent的分析）:
{attractions_context}

原始酒店数据:
{raw_data}"""

# 上下文字数上限（截断前几个 agent 返回的信息，减少 input token）
_CONTEXT_MAX_CHARS = 1500
# 节点级重试次数（降低以减少超时浪费）
_NODE_MAX_RETRIES = 1
# 节点超时（秒）
_NODE_TIMEOUT = 90


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


async def _llm_analyze(prompt: str, node_name: str) -> str:
    """轻量 LLM 分析调用（单次，非 ReAct），带超时和降级"""
    try:
        llm_inst = get_llm()
        response = await asyncio.wait_for(
            llm_inst.ainvoke([HumanMessage(content=prompt)]),
            timeout=60,
        )
        result = response.content if isinstance(response.content, str) else str(response.content)
        logger.debug("[%s] LLM分析完成(%d字)", node_name, len(result))
        return result
    except Exception as e:
        logger.warning("[%s] LLM分析失败: %s，降级为原始数据", node_name, e)
        return ""


def _build_langgraph(tools: List, progress_emitter: Optional[ProgressEmitter] = None, mode: str = "parallel_agent") -> Any:
    """构建并编译 LangGraph，支持 sequential / parallel_react / parallel_direct / parallel_agent 四种模式"""
    llm = get_llm()

    # ---- sequential 模式：单 Agent 完成所有工具调用 + 规划 ----
    if mode == "sequential":
        single_agent = create_react_agent(llm, tools, prompt=SINGLE_AGENT_PROMPT)

        async def single_agent_node(state: TripPlannerState) -> Dict:
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.SEARCH_ATTRACTIONS)
            req = state["request"]
            keyword = req.preferences[0] if req.preferences else "景点"
            query = (
                f"请为 {req.city} 规划 {req.travel_days} 天旅行计划。\n"
                f"日期={req.start_date}~{req.end_date} 交通={req.transportation} 住宿={req.accommodation}\n"
                f"偏好={', '.join(req.preferences) if req.preferences else '无'}\n"
                f"{'额外要求: ' + req.free_text_input if req.free_text_input else ''}\n"
                f"请先搜索 {req.city} 的 {keyword} 景点，再查询天气，再搜索酒店，最后生成行程 JSON。"
            )
            logger.info("[单Agent模式] %s", query[:200])
            try:
                result = await asyncio.wait_for(
                    single_agent.ainvoke({"messages": [HumanMessage(content=query)]}),
                    timeout=300,
                )
                plan_text = _extract_last_message(result)
                if progress_emitter:
                    await progress_emitter.emit(ProgressStep.PARSE_RESULT)
                trip_plan = _parse_plan(plan_text, req)
            except Exception as e:
                logger.warning("[单Agent模式] 失败: %s，使用备用方案", e)
                trip_plan = _fallback_plan(req)
            return {"trip_plan": trip_plan}

        workflow = StateGraph(TripPlannerState)
        workflow.add_node("single_agent", single_agent_node)
        workflow.set_entry_point("single_agent")
        workflow.add_edge("single_agent", END)
        return workflow.compile()

    # ---- parallel_react 模式：每个数据节点独立 ReAct Agent ----
    if mode == "parallel_react":
        attraction_agent = create_react_agent(llm, tools, prompt=ATTRACTION_AGENT_PROMPT)
        weather_agent = create_react_agent(llm, tools, prompt=WEATHER_AGENT_PROMPT)
        hotel_agent = create_react_agent(llm, tools, prompt=HOTEL_AGENT_PROMPT)

        async def react_search_attractions(state: TripPlannerState) -> Dict:
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.SEARCH_ATTRACTIONS)
            req = state["request"]
            keyword = req.preferences[0] if req.preferences else "景点"
            query = f"请搜索 {req.city} 的 {keyword} 景点，至少找到 5 个。"
            logger.info("[ReAct景点Agent] %s", query)
            try:
                result = await asyncio.wait_for(
                    attraction_agent.ainvoke({"messages": [HumanMessage(content=query)]}),
                    timeout=_NODE_TIMEOUT,
                )
                info = _extract_last_message(result)
            except Exception as e:
                logger.warning("[ReAct景点Agent] 失败: %s", e)
                info = f"{req.city}的热门景点信息（Agent调用失败，请根据已有知识推荐）"
            return {"attractions_info": info}

        async def react_query_weather(state: TripPlannerState) -> Dict:
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.QUERY_WEATHER)
            req = state["request"]
            query = f"请查询 {req.city} 的天气预报。"
            logger.info("[ReAct天气Agent] %s", query)
            try:
                result = await asyncio.wait_for(
                    weather_agent.ainvoke({"messages": [HumanMessage(content=query)]}),
                    timeout=_NODE_TIMEOUT,
                )
                info = _extract_last_message(result)
            except Exception as e:
                logger.warning("[ReAct天气Agent] 失败: %s", e)
                info = f"{req.city}近期天气预报（Agent调用失败）"
            return {"weather_info": info}

        async def react_search_hotels(state: TripPlannerState) -> Dict:
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.SEARCH_HOTELS)
            req = state["request"]
            query = f"请搜索 {req.city} 的{req.accommodation or '酒店'}。"
            logger.info("[ReAct酒店Agent] %s", query)
            try:
                result = await asyncio.wait_for(
                    hotel_agent.ainvoke({"messages": [HumanMessage(content=query)]}),
                    timeout=_NODE_TIMEOUT,
                )
                info = _extract_last_message(result)
            except Exception as e:
                logger.warning("[ReAct酒店Agent] 失败: %s", e)
                info = f"{req.city}的{req.accommodation}推荐（Agent调用失败）"
            return {"hotels_info": info}

        async def react_weather_and_hotels(state: TripPlannerState) -> Dict:
            weather_task = asyncio.create_task(react_query_weather(state))
            hotel_task = asyncio.create_task(react_search_hotels(state))
            weather_result, hotel_result = await asyncio.gather(weather_task, hotel_task)
            return {**weather_result, **hotel_result}

        # generate_plan 定义在下方共享代码中，这里先构建图后补充
        # 为避免代码重复，parallel_react 复用下方的 generate_plan
        pass  # generate_plan 在下方统一定义

    # ---- 共享的 MCP 工具索引和 direct_tool_invoke（parallel_direct / parallel_agent 共用）----
    _tools_by_name = {t.name: t for t in tools}

    async def _direct_tool_invoke(tool_name: str, tool_input: dict, node_name: str) -> str:
        """直接调用 MCP tool（无 LLM 推理），带超时和降级"""
        tool = _tools_by_name.get(tool_name)
        if not tool:
            logger.warning("[%s] 工具 %s 不存在，降级", node_name, tool_name)
            return ""
        try:
            async with _mcp_semaphore:
                result = await asyncio.wait_for(
                    tool.ainvoke(tool_input),
                    timeout=_NODE_TIMEOUT,
                )
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except asyncio.TimeoutError:
            logger.warning("[%s] 工具 %s 调用超时(%ds)", node_name, tool_name, _NODE_TIMEOUT)
        except Exception as e:
            logger.warning("[%s] 工具 %s 调用失败: %s", node_name, tool_name, e)
        return ""

    async def search_attractions(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.SEARCH_ATTRACTIONS)
        req = state["request"]
        keyword = req.preferences[0] if req.preferences else "景点"
        logger.info("[搜索景点] city=%s keyword=%s", req.city, keyword)
        info = await _direct_tool_invoke(
            "search_poi", {"keywords": keyword, "city": req.city}, "搜索景点")
        if not info:
            info = f"{req.city}的热门景点信息（工具调用失败，请根据已有知识推荐）"
            logger.warning("[搜索景点] 降级")
        logger.debug("[搜索景点] 结果(前300字): %s", info[:300])
        return {"attractions_info": info}

    async def query_weather(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.QUERY_WEATHER)
        req = state["request"]
        logger.info("[查询天气] city=%s", req.city)
        info = await _direct_tool_invoke(
            "get_weather", {"city": req.city}, "查询天气")
        if not info:
            info = f"{req.city}近期天气预报（工具调用失败，请根据季节常识推荐）"
            logger.warning("[查询天气] 降级")
        logger.debug("[查询天气] 结果(前300字): %s", info[:300])
        return {"weather_info": info}

    async def search_hotels(state: TripPlannerState) -> Dict:
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.SEARCH_HOTELS)
        req = state["request"]
        logger.info("[搜索酒店] city=%s accommodation=%s", req.city, req.accommodation)
        info = await _direct_tool_invoke(
            "search_poi", {"keywords": req.accommodation or "酒店", "city": req.city}, "搜索酒店")
        if not info:
            info = f"{req.city}的{req.accommodation}推荐（工具调用失败，请根据已有知识推荐）"
            logger.warning("[搜索酒店] 降级")
        logger.debug("[搜索酒店] 结果(前300字): %s", info[:300])
        return {"hotels_info": info}

    async def weather_and_hotels(state: TripPlannerState) -> Dict:
        """并行执行天气查询和酒店搜索，合并结果"""
        weather_task = asyncio.create_task(query_weather(state))
        hotel_task = asyncio.create_task(search_hotels(state))
        weather_result, hotel_result = await asyncio.gather(weather_task, hotel_task)
        return {**weather_result, **hotel_result}

    # ---- parallel_agent 模式：工具调用 + LLM 分析 + Agent 间协作 ----

    async def agent_search_attractions(state: TripPlannerState) -> Dict:
        """景点Agent：确定性工具调用 + LLM 按偏好筛选分析"""
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.SEARCH_ATTRACTIONS)
        req = state["request"]
        keyword = req.preferences[0] if req.preferences else "景点"
        logger.info("[Agent景点] city=%s keyword=%s", req.city, keyword)
        raw = await _direct_tool_invoke("search_poi", {"keywords": keyword, "city": req.city}, "Agent景点")
        if not raw:
            raw = f"{req.city}的热门景点信息（工具调用失败）"
        # LLM 分析：按偏好筛选排名 + 提取商圈位置
        analysis_prompt = ATTRACTION_ANALYSIS_PROMPT.format(
            preferences=", ".join(req.preferences) if req.preferences else "无",
            city=req.city,
            raw_data=raw[:_CONTEXT_MAX_CHARS],
        )
        analysis = await _llm_analyze(analysis_prompt, "Agent景点")
        # 分析结果 + 原始数据拼接（分析失败时降级为原始数据）
        info = analysis if analysis else raw
        logger.debug("[Agent景点] 分析结果(前300字): %s", info[:300])
        return {"attractions_info": info}

    async def agent_query_weather(state: TripPlannerState) -> Dict:
        """天气Agent：确定性工具调用 + LLM 每日活动建议"""
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.QUERY_WEATHER)
        req = state["request"]
        logger.info("[Agent天气] city=%s", req.city)
        raw = await _direct_tool_invoke("get_weather", {"city": req.city}, "Agent天气")
        if not raw:
            raw = f"{req.city}近期天气预报（工具调用失败）"
        analysis_prompt = WEATHER_ANALYSIS_PROMPT.format(city=req.city, raw_data=raw[:_CONTEXT_MAX_CHARS])
        analysis = await _llm_analyze(analysis_prompt, "Agent天气")
        info = analysis if analysis else raw
        logger.debug("[Agent天气] 分析结果(前300字): %s", info[:300])
        return {"weather_info": info}

    async def agent_search_hotels(state: TripPlannerState) -> Dict:
        """酒店Agent：确定性工具调用 + LLM 参考景点位置推荐（Agent间协作）"""
        if progress_emitter:
            await progress_emitter.emit(ProgressStep.SEARCH_HOTELS)
        req = state["request"]
        logger.info("[Agent酒店] city=%s accommodation=%s", req.city, req.accommodation)
        raw = await _direct_tool_invoke(
            "search_poi", {"keywords": req.accommodation or "酒店", "city": req.city}, "Agent酒店")
        if not raw:
            raw = f"{req.city}的{req.accommodation}推荐（工具调用失败）"
        # Agent 间协作：读取景点 Agent 的分析结果作为上下文
        attractions_ctx = state.get("attractions_info", "")[:800]
        analysis_prompt = HOTEL_ANALYSIS_PROMPT.format(
            accommodation=req.accommodation,
            attractions_context=attractions_ctx,
            raw_data=raw[:_CONTEXT_MAX_CHARS],
        )
        analysis = await _llm_analyze(analysis_prompt, "Agent酒店")
        info = analysis if analysis else raw
        logger.debug("[Agent酒店] 分析结果(前300字): %s", info[:300])
        return {"hotels_info": info}

    async def agent_weather_and_hotels(state: TripPlannerState) -> Dict:
        """并行执行天气Agent和酒店Agent的LLM分析"""
        weather_task = asyncio.create_task(agent_query_weather(state))
        hotel_task = asyncio.create_task(agent_search_hotels(state))
        weather_result, hotel_result = await asyncio.gather(weather_task, hotel_task)
        return {**weather_result, **hotel_result}

    # ---- 共享的 generate_plan 节点（所有 parallel 模式复用）----

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
            # 强制 JSON 输出模式，避免 LLM 返回非 JSON 内容
            llm_json = llm_inst.bind(response_format={"type": "json_object"})
            response = await asyncio.wait_for(
                llm_json.ainvoke([
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

    # ---- 根据模式选择节点函数构建图 ----
    workflow = StateGraph(TripPlannerState)

    if mode == "parallel_react":
        workflow.add_node("search_attractions", react_search_attractions)
        workflow.add_node("weather_and_hotels", react_weather_and_hotels)
    elif mode == "parallel_agent":
        workflow.add_node("search_attractions", agent_search_attractions)
        workflow.add_node("weather_and_hotels", agent_weather_and_hotels)
    else:  # parallel_direct
        workflow.add_node("search_attractions", search_attractions)
        workflow.add_node("weather_and_hotels", weather_and_hotels)

    workflow.add_node("generate_plan", generate_plan)

    workflow.set_entry_point("search_attractions")
    workflow.add_edge("search_attractions", "weather_and_hotels")
    workflow.add_edge("weather_and_hotels", "generate_plan")
    workflow.add_edge("generate_plan",      END)

    return workflow.compile()


# ============================================================
# Helpers
# ============================================================

def _parse_plan(response: str, request: TripRequest) -> TripPlan:
    """从 LLM 响应中解析 TripPlan，解析失败时尝试 LLM 修复一次"""
    def _extract_json(text: str) -> dict:
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end   = text.find("```", json_start)
            json_str   = text[json_start:json_end].strip()
        elif "```" in text:
            json_start = text.find("```") + 3
            json_end   = text.find("```", json_start)
            json_str   = text[json_start:json_end].strip()
        elif "{" in text and "}" in text:
            json_start = text.find("{")
            json_end   = text.rfind("}") + 1
            json_str   = text[json_start:json_end]
        else:
            raise ValueError("响应中未找到 JSON 数据")
        return json.loads(json_str)

    # 第一次尝试解析
    try:
        data = _extract_json(response)
        return TripPlan(**data)
    except Exception as first_err:
        logger.warning("解析行程失败: %s，尝试 LLM 修复", first_err)

    # LLM 修复重试（最多一次）
    try:
        llm_inst = get_llm()
        llm_json = llm_inst.bind(response_format={"type": "json_object"})
        import asyncio as _aio
        loop = _aio.get_event_loop()
        fix_response = loop.run_until_complete(
            llm_json.ainvoke([
                SystemMessage(content="修复下面的 JSON，使其符合 TripPlan schema。只返回修复后的 JSON。"),
                HumanMessage(content=f"原始输出:\n{response[:3000]}\n\n错误信息: {first_err}\n\n目标 schema:\n{PLANNER_OUTPUT_SCHEMA}"),
            ])
        ) if not loop.is_running() else None
        if fix_response:
            data = _extract_json(fix_response.content)
            plan = TripPlan(**data)
            logger.info("LLM 修复解析成功")
            return plan
    except Exception as fix_err:
        logger.warning("LLM 修复失败: %s", fix_err)

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

    async def plan_trip(self, request: TripRequest, progress_emitter: Optional[ProgressEmitter] = None, mode: str = "parallel_agent") -> TripPlan:
        """异步执行多智能体协作，生成旅行计划，支持 SSE 进度回调"""
        logger.info("LangGraph 旅行规划启动 | 目的地=%s | 日期=%s~%s | 模式=%s",
                    request.city, request.start_date, request.end_date, mode)

        if progress_emitter:
            await progress_emitter.emit(ProgressStep.INIT)

        try:
            if progress_emitter:
                await progress_emitter.emit(ProgressStep.MCP_CONNECT)

            # 使用 MCP 单例，避免每次请求创建/销毁子进程
            try:
                tools = await _get_mcp_tools()
            except Exception as e:
                logger.warning("MCP 工具加载失败，尝试重置后重试: %s", e)
                await _reset_mcp_client()
                tools = await _get_mcp_tools()

            logger.info("MCP 工具加载完成，共 %d 个: %s", len(tools), [t.name for t in tools])

            graph = _build_langgraph(tools, progress_emitter, mode=mode)
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
