"""
Amap MCP Server
===============
独立的高德地图 MCP 工具服务，通过 stdio 协议对外暴露工具。
Agent 以子进程方式启动本服务，通过 MCP 协议调用工具，
实现 Amap API 与 LangGraph Agent 的彻底解耦。

工具列表:
  - search_poi      : 关键词搜索 POI
  - get_weather     : 城市天气预报
  - geocode         : 地址转经纬度
  - plan_route      : 路线规划 (walking / driving / transit)
  - get_poi_detail  : POI 详情

环境变量:
  AMAP_API_KEY      : 高德地图 REST API Key (必填)
"""

import os
import json
import httpx
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------------
# 加载本目录的 .env
# ------------------------------------------------------------------
# 优先加载项目根目录 .env，再兜底本目录（独立运行时）
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
load_dotenv(Path(__file__).parent / ".env", override=False)

AMAP_API_KEY = os.environ.get("AMAP_API_KEY", "")
AMAP_BASE_URL = "https://restapi.amap.com"

if not AMAP_API_KEY:
    raise EnvironmentError("AMAP_API_KEY 未配置，请在 mcp-server/.env 中设置")

# ------------------------------------------------------------------
# FastMCP 实例
# ------------------------------------------------------------------
mcp = FastMCP("amap-tools")


# ------------------------------------------------------------------
# 内部 HTTP 辅助
# ------------------------------------------------------------------
async def _get(path: str, params: dict) -> dict:
    params["key"] = AMAP_API_KEY
    params["output"] = "JSON"
    async with httpx.AsyncClient(base_url=AMAP_BASE_URL, timeout=20.0) as client:
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------
# Tool: search_poi
# ------------------------------------------------------------------
@mcp.tool()
async def search_poi(keywords: str, city: str, citylimit: bool = True) -> str:
    """
    在指定城市内搜索 POI（兴趣点），例如景点、餐厅、酒店等。

    Args:
        keywords: 搜索关键词，例如 "故宫"、"酒店"、"美食"
        city:     城市名，例如 "北京"
        citylimit: 是否限定在城市内，默认 True

    Returns:
        JSON 字符串，包含 POI 列表（名称、地址、坐标、电话等）
    """
    data = await _get(
        "/v3/place/text",
        {
            "keywords": keywords,
            "city": city,
            "citylimit": "true" if citylimit else "false",
            "offset": 10,
        },
    )

    if data.get("status") != "1":
        return json.dumps({"error": data.get("info", "搜索失败"), "pois": []}, ensure_ascii=False)

    pois = []
    for p in data.get("pois", []):
        loc = p.get("location", "0,0").split(",")
        pois.append({
            "id": p.get("id", ""),
            "name": p.get("name", ""),
            "type": p.get("type", ""),
            "address": p.get("address", ""),
            "longitude": float(loc[0]) if len(loc) == 2 else 0.0,
            "latitude": float(loc[1]) if len(loc) == 2 else 0.0,
            "tel": p.get("tel") or "",
        })

    return json.dumps({"pois": pois, "count": len(pois)}, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool: get_weather
# ------------------------------------------------------------------
@mcp.tool()
async def get_weather(city: str) -> str:
    """
    查询指定城市的未来几天天气预报。

    Args:
        city: 城市名，例如 "北京"、"上海"

    Returns:
        JSON 字符串，包含逐日天气预报（日期、天气、气温、风向、风力）
    """
    data = await _get(
        "/v3/weather/weatherInfo",
        {"city": city, "extensions": "all"},
    )

    if data.get("status") != "1":
        return json.dumps({"error": data.get("info", "查询失败"), "forecasts": []}, ensure_ascii=False)

    result = []
    for forecast in data.get("forecasts", []):
        for cast in forecast.get("casts", []):
            result.append({
                "date": cast.get("date", ""),
                "day_weather": cast.get("dayweather", ""),
                "night_weather": cast.get("nightweather", ""),
                "day_temp": cast.get("daytemp", 0),
                "night_temp": cast.get("nighttemp", 0),
                "wind_direction": cast.get("daywind", ""),
                "wind_power": cast.get("daypower", ""),
            })

    return json.dumps({"forecasts": result}, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool: geocode
# ------------------------------------------------------------------
@mcp.tool()
async def geocode(address: str, city: Optional[str] = None) -> str:
    """
    将地址字符串转换为经纬度坐标（地理编码）。

    Args:
        address: 详细地址，例如 "北京市天安门广场"
        city:    可选，城市名，用于提高精度

    Returns:
        JSON 字符串，包含 longitude / latitude；失败时返回空对象
    """
    params: dict = {"address": address}
    if city:
        params["city"] = city

    data = await _get("/v3/geocode/geo", params)

    if data.get("status") != "1" or not data.get("geocodes"):
        return json.dumps({"error": "地理编码失败", "longitude": None, "latitude": None}, ensure_ascii=False)

    loc_str = data["geocodes"][0].get("location", "")
    if not loc_str:
        return json.dumps({"error": "未获取到坐标", "longitude": None, "latitude": None}, ensure_ascii=False)

    lng, lat = (float(x) for x in loc_str.split(","))
    return json.dumps({"longitude": lng, "latitude": lat}, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool: plan_route
# ------------------------------------------------------------------
@mcp.tool()
async def plan_route(
    origin_address: str,
    destination_address: str,
    route_type: str = "walking",
    city: Optional[str] = None,
) -> str:
    """
    规划两地之间的出行路线。

    Args:
        origin_address:      出发地地址
        destination_address: 目的地地址
        route_type:          路线类型：walking（步行）/ driving（驾车）/ transit（公交）
        city:                城市（公交模式必填）

    Returns:
        JSON 字符串，包含距离（米）、耗时（秒）、路线描述等
    """
    # 先地理编码
    origin_data = json.loads(await geocode(origin_address, city))
    dest_data = json.loads(await geocode(destination_address, city))

    if origin_data.get("error") or dest_data.get("error"):
        return json.dumps({"error": "地理编码失败，无法规划路线"}, ensure_ascii=False)

    origin_str = f"{origin_data['longitude']},{origin_data['latitude']}"
    dest_str = f"{dest_data['longitude']},{dest_data['latitude']}"

    endpoint_map = {
        "walking": "/v3/direction/walking",
        "driving": "/v3/direction/driving",
        "transit": "/v5/direction/transit/integrated",
    }
    endpoint = endpoint_map.get(route_type, "/v3/direction/walking")

    params: dict = {"origin": origin_str, "destination": dest_str}
    if route_type == "transit":
        params["city1"] = city or ""
        params["city2"] = city or ""

    data = await _get(endpoint, params)

    if data.get("status") != "1":
        return json.dumps({"error": data.get("info", "路线规划失败")}, ensure_ascii=False)

    route = data.get("route", {})
    paths = route.get("paths", []) or route.get("transits", [])
    if not paths:
        return json.dumps({"error": "未找到路线"}, ensure_ascii=False)

    first = paths[0]
    return json.dumps({
        "distance": float(first.get("distance", 0)),
        "duration": int(first.get("duration", 0)),
        "route_type": route_type,
        "origin": origin_address,
        "destination": destination_address,
        "description": (
            f"从「{origin_address}」到「{destination_address}」"
            f"({route_type})：约 {int(first.get('distance', 0))/1000:.1f} km，"
            f"耗时约 {int(first.get('duration', 0))//60} 分钟"
        ),
    }, ensure_ascii=False)


# ------------------------------------------------------------------
# Tool: get_poi_detail
# ------------------------------------------------------------------
@mcp.tool()
async def get_poi_detail(poi_id: str) -> str:
    """
    根据 POI ID 获取景点/地点的详细信息。

    Args:
        poi_id: 高德 POI ID，例如通过 search_poi 返回的 id 字段

    Returns:
        JSON 字符串，包含 POI 详情
    """
    data = await _get("/v3/place/detail", {"id": poi_id})

    if data.get("status") != "1":
        return json.dumps({"error": data.get("info", "获取详情失败")}, ensure_ascii=False)

    pois = data.get("pois", [])
    if not pois:
        return json.dumps({"error": "未找到 POI"}, ensure_ascii=False)

    p = pois[0]
    loc = p.get("location", "0,0").split(",")
    return json.dumps({
        "id": p.get("id", ""),
        "name": p.get("name", ""),
        "type": p.get("type", ""),
        "address": p.get("address", ""),
        "longitude": float(loc[0]) if len(loc) == 2 else 0.0,
        "latitude": float(loc[1]) if len(loc) == 2 else 0.0,
        "tel": p.get("tel") or "",
        "photos": [ph.get("url", "") for ph in p.get("photos", [])],
    }, ensure_ascii=False)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()  # 默认 stdio transport
