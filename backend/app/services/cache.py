"""
Redis 行程缓存工具
==================
以 hash(city + dates + preferences) 为 key，TTL 默认 24h。

- REDIS_URL 为空时自动禁用，所有操作静默降级，不影响主流程。
- 连接失败时同样静默降级，不抛出异常。
"""

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_redis_client = None
_redis_unavailable = False  # 标记本次进程 Redis 是否可用，避免反复重连


async def _get_redis():
    """懒加载 Redis 连接，失败时标记不可用并静默降级"""
    global _redis_client, _redis_unavailable

    if _redis_unavailable:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        from ..config import get_settings
        s = get_settings()
        if not s.redis_url:
            logger.debug("[缓存] REDIS_URL 未配置，缓存已禁用")
            _redis_unavailable = True
            return None

        import redis.asyncio as aioredis
        client = aioredis.from_url(s.redis_url, decode_responses=True, socket_connect_timeout=2)
        await client.ping()  # 验证连接
        _redis_client = client
        logger.info("[缓存] Redis 连接成功: %s", s.redis_url)
        return _redis_client

    except Exception as e:
        logger.warning("[缓存] Redis 不可用，降级为无缓存模式: %s", e)
        _redis_unavailable = True
        return None


def make_cache_key(req) -> str:
    """根据请求关键字段生成确定性缓存 key（sha256 前16位）"""
    prefs = ",".join(sorted(req.preferences or []))
    raw = f"{req.city}|{req.start_date}|{req.end_date}|{req.transportation}|{req.accommodation}|{prefs}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"trip_plan:{digest}"


async def get_cached_plan(key: str) -> Optional[dict]:
    """查找缓存，命中返回 dict，未命中或异常返回 None"""
    try:
        r = await _get_redis()
        if r is None:
            return None
        raw = await r.get(key)
        if raw:
            logger.info("[缓存] 命中 key=%s", key)
            return json.loads(raw)
        logger.debug("[缓存] 未命中 key=%s", key)
    except Exception as e:
        logger.warning("[缓存] 读取异常，降级: %s", e)
    return None


async def set_cached_plan(key: str, plan_dict: dict, ttl: int = 86400):
    """写入缓存，失败时静默忽略"""
    try:
        r = await _get_redis()
        if r is None:
            return
        await r.setex(key, ttl, json.dumps(plan_dict, ensure_ascii=False))
        logger.info("[缓存] 写入成功 key=%s ttl=%ds", key, ttl)
    except Exception as e:
        logger.warning("[缓存] 写入异常，忽略: %s", e)
