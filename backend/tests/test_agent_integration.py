"""
test_agent_integration.py — Agent 集成测试（Mock MCP tools + LLM）
覆盖：parallel_direct 模式完整流程，验证 plan_trip 返回合法 TripPlan
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import TripRequest, TripPlan


# ── 测试用 TripPlan JSON（LLM 返回格式）──────────────────────

_MOCK_PLAN_JSON = {
    "city": "北京",
    "start_date": "2026-03-15",
    "end_date": "2026-03-17",
    "days": [
        {
            "date": "2026-03-15",
            "day_index": 0,
            "description": "第一天游览天安门广场和故宫",
            "transportation": "地铁",
            "accommodation": "如家酒店",
            "hotel": {
                "name": "如家快捷酒店",
                "address": "北京市东城区",
                "location": {"longitude": 116.3974, "latitude": 39.9163},
                "price_range": "200-300元/晚",
                "rating": "4.2",
                "distance": "0.5km",
                "type": "经济型酒店",
                "estimated_cost": 250,
            },
            "attractions": [
                {
                    "name": "故宫博物院",
                    "address": "北京市东城区景山前街4号",
                    "location": {"longitude": 116.3974, "latitude": 39.9163},
                    "visit_duration": 180,
                    "description": "中国最大的古代宫廷建筑群",
                    "category": "历史文化",
                    "ticket_price": 60,
                }
            ],
            "meals": [
                {"type": "breakfast", "name": "豆浆油条", "estimated_cost": 15},
                {"type": "lunch", "name": "宫廷菜", "estimated_cost": 80},
                {"type": "dinner", "name": "北京烤鸭", "estimated_cost": 120},
            ],
        }
    ],
    "weather_info": [
        {
            "date": "2026-03-15",
            "day_weather": "晴",
            "night_weather": "多云",
            "day_temp": 15,
            "night_temp": 5,
            "wind_direction": "南风",
            "wind_power": "3级",
        }
    ],
    "overall_suggestions": "建议提前购票，避开高峰期参观。",
    "budget": {
        "total_attractions": 60,
        "total_hotels": 250,
        "total_meals": 215,
        "total_transportation": 30,
        "total": 555,
    },
}


def _make_mock_tool(name: str, return_value: str = "{}"):
    """创建 Mock MCP 工具"""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def _make_mock_llm_response(content: str):
    """创建 Mock LLM 响应"""
    response = MagicMock()
    response.content = content
    return response


# ── parallel_direct 模式集成测试 ─────────────────────────────

class TestParallelDirectIntegration:
    @pytest.fixture
    def trip_request(self):
        return TripRequest(
            city="北京",
            start_date="2026-03-15",
            end_date="2026-03-15",
            travel_days=1,
            transportation="地铁",
            accommodation="经济型酒店",
            preferences=["历史文化"],
        )

    @pytest.fixture
    def mock_tools(self):
        poi_result = json.dumps({
            "pois": [{"id": "1", "name": "故宫", "type": "景点", "address": "东城区", "lon": "116.39", "lat": "39.91", "tel": ""}],
            "count": 1,
        })
        weather_result = json.dumps({
            "forecasts": [{"date": "2026-03-15", "day_weather": "晴", "night_weather": "多云",
                           "day_temp": "15", "night_temp": "5", "wind_dir": "南风", "wind_power": "3级"}]
        })
        return [
            _make_mock_tool("search_poi", poi_result),
            _make_mock_tool("get_weather", weather_result),
            _make_mock_tool("geocode", '{"longitude": 116.3974, "latitude": 39.9163}'),
        ]

    @pytest.mark.asyncio
    async def test_plan_trip_returns_tripplan(self, trip_request, mock_tools):
        """parallel_direct 模式：完整流程返回 TripPlan 对象"""
        mock_llm_response = _make_mock_llm_response(json.dumps(_MOCK_PLAN_JSON))
        mock_llm = MagicMock()
        mock_llm_bound = MagicMock()
        mock_llm_bound.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_llm.bind = MagicMock(return_value=mock_llm_bound)

        with patch("app.agents.trip_planner_agent._get_mcp_tools", AsyncMock(return_value=mock_tools)), \
             patch("app.agents.trip_planner_agent.get_llm", return_value=mock_llm):
            from app.agents.trip_planner_agent import LangGraphTripPlanner
            planner = LangGraphTripPlanner()
            result = await planner.plan_trip(trip_request, mode="parallel_direct")

        assert isinstance(result, TripPlan)
        assert result.city == "北京"

    @pytest.mark.asyncio
    async def test_plan_trip_falls_back_on_llm_failure(self, trip_request, mock_tools):
        """LLM 调用失败时，plan_trip 不抛出异常，返回降级 TripPlan"""
        mock_llm = MagicMock()
        mock_llm_bound = MagicMock()
        mock_llm_bound.ainvoke = AsyncMock(side_effect=Exception("LLM 连接失败"))
        mock_llm.bind = MagicMock(return_value=mock_llm_bound)

        with patch("app.agents.trip_planner_agent._get_mcp_tools", AsyncMock(return_value=mock_tools)), \
             patch("app.agents.trip_planner_agent.get_llm", return_value=mock_llm):
            from app.agents.trip_planner_agent import LangGraphTripPlanner
            planner = LangGraphTripPlanner()
            result = await planner.plan_trip(trip_request, mode="parallel_direct")

        # 降级方案依然返回有效 TripPlan
        assert isinstance(result, TripPlan)
        assert result.city == "北京"
        assert len(result.days) == trip_request.travel_days

    @pytest.mark.asyncio
    async def test_plan_trip_falls_back_on_mcp_failure(self, trip_request):
        """MCP 工具加载失败时，plan_trip 不抛出异常，返回降级计划"""
        with patch("app.agents.trip_planner_agent._get_mcp_tools",
                   AsyncMock(side_effect=Exception("MCP 子进程启动失败"))), \
             patch("app.agents.trip_planner_agent._reset_mcp_client", AsyncMock()):
            from app.agents.trip_planner_agent import LangGraphTripPlanner
            planner = LangGraphTripPlanner()
            result = await planner.plan_trip(trip_request, mode="parallel_direct")

        assert isinstance(result, TripPlan)

    @pytest.mark.asyncio
    async def test_mcp_tools_called_with_correct_args(self, trip_request, mock_tools):
        """验证 MCP 工具被正确参数调用"""
        mock_llm_response = _make_mock_llm_response(json.dumps(_MOCK_PLAN_JSON))
        mock_llm = MagicMock()
        mock_llm_bound = MagicMock()
        mock_llm_bound.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_llm.bind = MagicMock(return_value=mock_llm_bound)

        with patch("app.agents.trip_planner_agent._get_mcp_tools", AsyncMock(return_value=mock_tools)), \
             patch("app.agents.trip_planner_agent.get_llm", return_value=mock_llm):
            from app.agents.trip_planner_agent import LangGraphTripPlanner
            planner = LangGraphTripPlanner()
            await planner.plan_trip(trip_request, mode="parallel_direct")

        # search_poi 和 get_weather 都应被调用
        poi_tool = next(t for t in mock_tools if t.name == "search_poi")
        weather_tool = next(t for t in mock_tools if t.name == "get_weather")
        assert poi_tool.ainvoke.called
        assert weather_tool.ainvoke.called

        # get_weather 应传入城市名
        weather_call_args = weather_tool.ainvoke.call_args
        assert "city" in weather_call_args[0][0]
        assert weather_call_args[0][0]["city"] == "北京"
