"""
test_schemas.py — Pydantic 数据模型校验单元测试
覆盖：TripRequest 字段校验边界、非法输入拒绝、free_text 净化
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Location


def _valid_request(**overrides):
    """构造合法的最小请求数据"""
    data = dict(
        city="北京",
        start_date="2026-03-15",
        end_date="2026-03-17",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
    )
    data.update(overrides)
    return data


# ── city 校验 ────────────────────────────────────────────────

class TestCityValidation:
    def test_valid_city(self):
        req = TripRequest(**_valid_request(city="北京"))
        assert req.city == "北京"

    def test_city_trimmed(self):
        req = TripRequest(**_valid_request(city="  上海  "))
        assert req.city == "上海"

    def test_city_too_long_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(city="x" * 51))

    def test_city_empty_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(city=""))

    def test_city_with_html_tag_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(city="<script>北京"))

    def test_city_with_slash_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(city="北京/上海"))

    def test_city_with_semicolon_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(city="北京;rm -rf"))


# ── 日期格式校验 ──────────────────────────────────────────────

class TestDateValidation:
    def test_valid_date(self):
        req = TripRequest(**_valid_request())
        assert req.start_date == "2026-03-15"

    def test_invalid_date_format_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(start_date="2026/03/15"))

    def test_invalid_date_letters_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(end_date="not-a-date"))

    def test_invalid_date_missing_day_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(start_date="2026-03"))


# ── travel_days 边界 ─────────────────────────────────────────

class TestTravelDaysValidation:
    def test_min_days(self):
        req = TripRequest(**_valid_request(travel_days=1))
        assert req.travel_days == 1

    def test_max_days(self):
        req = TripRequest(**_valid_request(travel_days=30))
        assert req.travel_days == 30

    def test_zero_days_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(travel_days=0))

    def test_exceeds_max_raises(self):
        with pytest.raises(ValidationError):
            TripRequest(**_valid_request(travel_days=31))


# ── free_text_input 净化 ─────────────────────────────────────

class TestFreeTextSanitization:
    def test_html_tags_removed(self):
        req = TripRequest(**_valid_request(free_text_input="<b>请安排博物馆</b>"))
        assert "<b>" not in req.free_text_input
        assert "请安排博物馆" in req.free_text_input

    def test_truncated_to_500(self):
        req = TripRequest(**_valid_request(free_text_input="x" * 600))
        assert len(req.free_text_input) == 500

    def test_none_free_text_ok(self):
        req = TripRequest(**_valid_request(free_text_input=None))
        assert req.free_text_input is None

    def test_empty_free_text_ok(self):
        req = TripRequest(**_valid_request(free_text_input=""))
        assert req.free_text_input == ""


# ── preferences 字段 ─────────────────────────────────────────

class TestPreferencesField:
    def test_default_empty_list(self):
        req = TripRequest(**_valid_request())
        assert req.preferences == []

    def test_multiple_preferences(self):
        req = TripRequest(**_valid_request(preferences=["历史文化", "美食", "自然"]))
        assert len(req.preferences) == 3


# ── execution_mode 默认值 ────────────────────────────────────

class TestExecutionMode:
    def test_default_mode(self):
        req = TripRequest(**_valid_request())
        assert req.execution_mode == "parallel_agent"

    def test_custom_mode(self):
        req = TripRequest(**_valid_request(execution_mode="sequential"))
        assert req.execution_mode == "sequential"
