"""
test_cache.py — 缓存工具单元测试
覆盖：make_cache_key 幂等性、preferences 顺序无关性、Redis 不可用时静默降级
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from types import SimpleNamespace


def _make_req(**kwargs):
    """构造最小 TripRequest-like 对象"""
    defaults = dict(
        city="北京",
        start_date="2026-03-15",
        end_date="2026-03-17",
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化", "美食"],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── make_cache_key ──────────────────────────────────────────

class TestMakeCacheKey:
    def _key(self, **kwargs):
        from app.services.cache import make_cache_key
        return make_cache_key(_make_req(**kwargs))

    def test_returns_string_with_prefix(self):
        from app.services.cache import make_cache_key
        key = make_cache_key(_make_req())
        assert key.startswith("trip_plan:")

    def test_idempotent(self):
        """相同请求两次调用结果一致"""
        from app.services.cache import make_cache_key
        req = _make_req()
        assert make_cache_key(req) == make_cache_key(req)

    def test_preferences_order_independent(self):
        """preferences 顺序不影响 key"""
        from app.services.cache import make_cache_key
        req_a = _make_req(preferences=["历史文化", "美食"])
        req_b = _make_req(preferences=["美食", "历史文化"])
        assert make_cache_key(req_a) == make_cache_key(req_b)

    def test_different_city_different_key(self):
        from app.services.cache import make_cache_key
        key_bj = make_cache_key(_make_req(city="北京"))
        key_sh = make_cache_key(_make_req(city="上海"))
        assert key_bj != key_sh

    def test_different_dates_different_key(self):
        from app.services.cache import make_cache_key
        key_a = make_cache_key(_make_req(start_date="2026-03-15"))
        key_b = make_cache_key(_make_req(start_date="2026-04-01"))
        assert key_a != key_b

    def test_different_prefs_different_key(self):
        from app.services.cache import make_cache_key
        key_a = make_cache_key(_make_req(preferences=["历史文化"]))
        key_b = make_cache_key(_make_req(preferences=["自然风光"]))
        assert key_a != key_b

    def test_empty_preferences(self):
        """空 preferences 不应抛出异常"""
        from app.services.cache import make_cache_key
        key = make_cache_key(_make_req(preferences=[]))
        assert key.startswith("trip_plan:")

    def test_key_length(self):
        """key 格式：trip_plan: + 16位 hex"""
        from app.services.cache import make_cache_key
        key = make_cache_key(_make_req())
        digest_part = key[len("trip_plan:"):]
        assert len(digest_part) == 16
        assert all(c in "0123456789abcdef" for c in digest_part)


# ── get_cached_plan / set_cached_plan 降级 ─────────────────

class TestCacheFallback:
    @pytest.mark.asyncio
    async def test_get_returns_none_when_redis_unavailable(self):
        """Redis 不可用时 get_cached_plan 静默返回 None"""
        import app.services.cache as cache_mod
        # 重置模块状态，模拟 Redis 不可用
        original_unavailable = cache_mod._redis_unavailable
        original_client = cache_mod._redis_client
        try:
            cache_mod._redis_unavailable = True
            cache_mod._redis_client = None
            result = await cache_mod.get_cached_plan("trip_plan:abc123")
            assert result is None
        finally:
            cache_mod._redis_unavailable = original_unavailable
            cache_mod._redis_client = original_client

    @pytest.mark.asyncio
    async def test_set_does_not_raise_when_redis_unavailable(self):
        """Redis 不可用时 set_cached_plan 不抛出异常"""
        import app.services.cache as cache_mod
        original_unavailable = cache_mod._redis_unavailable
        original_client = cache_mod._redis_client
        try:
            cache_mod._redis_unavailable = True
            cache_mod._redis_client = None
            # 不应抛出任何异常
            await cache_mod.set_cached_plan("trip_plan:abc123", {"city": "北京"})
        finally:
            cache_mod._redis_unavailable = original_unavailable
            cache_mod._redis_client = original_client

    @pytest.mark.asyncio
    async def test_get_returns_none_when_redis_url_empty(self):
        """REDIS_URL 未配置时 get_cached_plan 返回 None"""
        import app.services.cache as cache_mod
        original_unavailable = cache_mod._redis_unavailable
        original_client = cache_mod._redis_client
        try:
            cache_mod._redis_unavailable = False
            cache_mod._redis_client = None

            mock_settings = MagicMock()
            mock_settings.redis_url = ""  # 空 URL

            # get_settings 在 _get_redis 内部通过 from ..config import get_settings 导入
            # 需要 patch 源模块路径
            with patch("app.config.get_settings", return_value=mock_settings):
                result = await cache_mod.get_cached_plan("trip_plan:test")
                assert result is None
        finally:
            cache_mod._redis_unavailable = original_unavailable
            cache_mod._redis_client = original_client

    @pytest.mark.asyncio
    async def test_get_cached_plan_returns_data_on_hit(self):
        """Redis 命中时正确返回反序列化的数据"""
        import app.services.cache as cache_mod
        import json

        plan_data = {"city": "北京", "days": []}
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(plan_data))

        original_unavailable = cache_mod._redis_unavailable
        original_client = cache_mod._redis_client
        try:
            cache_mod._redis_unavailable = False
            cache_mod._redis_client = mock_redis
            result = await cache_mod.get_cached_plan("trip_plan:abc")
            assert result == plan_data
        finally:
            cache_mod._redis_unavailable = original_unavailable
            cache_mod._redis_client = original_client
