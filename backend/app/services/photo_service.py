"""Unsplash 图片搜索服务"""

import asyncio
import logging
import httpx
from typing import Optional
from ..config import get_settings

logger = logging.getLogger(__name__)

UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

# 复用同一个 httpx 客户端，减少连接开销
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=15)
    return _http_client


async def search_photo(query: str, city: str = "", page: int = 1) -> Optional[str]:
    """
    通过 Unsplash API 搜索景点图片，返回第一张图片的 small URL。

    搜索策略：先用景点名单独搜索（更精准）；
    如果无结果，再追加城市名重试。

    Args:
        query: 搜索关键词（景点名称）
        city: 城市名，仅在单独搜索无结果时追加
        page: 结果页码，用于避免所有查询返回同一张图

    Returns:
        图片URL 或 None
    """
    settings = get_settings()
    access_key = settings.unsplash_access_key
    if not access_key:
        return None

    headers = {"Authorization": f"Client-ID {access_key}"}
    client = _get_http_client()

    # 第一轮：只用景点名搜索（更精确，不会被城市名主导）
    for search_query in [query, f"{query} {city}"] if city else [query]:
        try:
            resp = await client.get(
                UNSPLASH_SEARCH_URL,
                params={
                    "query": search_query,
                    "per_page": 3,
                    "page": page,
                    "orientation": "landscape",
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                # 从前3张中选取（基于page偏移），尽量保证不同景点得到不同图片
                idx = (page - 1) % len(results)
                return results[idx]["urls"].get("small")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Unsplash 速率限制，跳过 [%s]", query)
                return None
            logger.warning("Unsplash HTTP 错误 [%s]: %d", query, e.response.status_code)
        except Exception as e:
            logger.warning("Unsplash 搜索失败 [%s]: %s", query, e)

    return None


async def batch_search_photos(names: list[str], city: str = "") -> dict[str, str]:
    """
    批量搜索多个景点的图片。

    使用串行 + 短延迟策略避免触发 Unsplash 速率限制，
    并为每个景点分配不同的 page 参数以获取不同图片。

    Args:
        names: 景点名称列表
        city: 城市名

    Returns:
        {景点名: 图片URL} 字典（仅包含有结果的）
    """
    results: dict[str, str] = {}
    unique_names = list(dict.fromkeys(names))  # 去重且保持顺序

    for i, name in enumerate(unique_names):
        url = await search_photo(name, city, page=i + 1)
        if url:
            results[name] = url
        # 每次请求间隔 200ms，避免并发触发速率限制
        if i < len(unique_names) - 1:
            await asyncio.sleep(0.2)

    logger.info("Unsplash 图片加载完成: %d/%d 张", len(results), len(unique_names))
    return results
