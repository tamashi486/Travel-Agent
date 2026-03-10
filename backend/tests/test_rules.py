"""
test_rules.py — Benchmark 规则评测函数单元测试
覆盖：evaluate_rules 的 13 项规则，_haversine, _eval_hotel_near_attractions, _eval_weather_activity_match
"""
import sys
import os
import math
import pytest

# benchmark 目录在 backend 同级，需要手动加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../benchmark"))
from benchmark_trip import (
    evaluate_rules,
    _haversine,
    _eval_hotel_near_attractions,
    _eval_weather_activity_match,
)


# ── 辅助构造函数 ─────────────────────────────────────────────

def _attraction(name="故宫", lon=116.3974, lat=39.9163, category="历史文化"):
    return {
        "name": name,
        "address": "北京市东城区",
        "location": {"longitude": lon, "latitude": lat},
        "description": f"{name}是著名景点",
        "category": category,
        "visit_duration": 120,
    }


def _hotel(name="如家酒店", lon=116.3970, lat=39.9160):
    return {
        "name": name,
        "address": "北京市东城区",
        "location": {"longitude": lon, "latitude": lat},
    }


def _day(date="2026-03-15", hotel=None, attractions=None, weather=None):
    return {
        "date": date,
        "day_index": 0,
        "description": "第一天行程",
        "transportation": "地铁",
        "accommodation": "酒店",
        "hotel": hotel if hotel is not None else _hotel(),
        "attractions": attractions if attractions is not None else [_attraction()],
        "meals": [],
    }


def _request(city="北京", travel_days=1, preferences=None, start_date="2026-03-15"):
    return {
        "city": city,
        "start_date": start_date,
        "travel_days": travel_days,
        "preferences": preferences or [],
    }


_SENTINEL = object()

def _full_data(city="北京", days=None, weather_info=None, suggestions="注意防晒", budget=_SENTINEL):
    return {
        "city": city,
        "start_date": "2026-03-15",
        "end_date": "2026-03-15",
        "days": days if days is not None else [_day()],
        "weather_info": weather_info if weather_info is not None else [],
        "overall_suggestions": suggestions,
        "budget": {"total": 500} if budget is _SENTINEL else budget,
    }


# ── _haversine ───────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine(116.0, 39.0, 116.0, 39.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # 北京(116.3974, 39.9163) 到 上海(121.4737, 31.2304) ≈ 1067 km
        dist = _haversine(116.3974, 39.9163, 121.4737, 31.2304)
        assert 1050 < dist < 1090

    def test_symmetry(self):
        d1 = _haversine(116.0, 39.0, 121.0, 31.0)
        d2 = _haversine(121.0, 31.0, 116.0, 39.0)
        assert d1 == pytest.approx(d2, rel=1e-9)


# ── _eval_hotel_near_attractions ─────────────────────────────

class TestHotelNearAttractions:
    def test_hotel_very_close_to_attraction_passes(self):
        # 酒店和景点坐标几乎相同，距离 < 1km
        days = [_day(hotel=_hotel(lon=116.3974, lat=39.9163),
                     attractions=[_attraction(lon=116.3975, lat=39.9164)])]
        assert _eval_hotel_near_attractions(days) is True

    def test_hotel_far_from_attraction_fails(self):
        # 酒店在北京(116.39), 景点在上海(121.47) — 超过 10km
        days = [_day(hotel=_hotel(lon=116.3974, lat=39.9163),
                     attractions=[_attraction(lon=121.4737, lat=31.2304)])]
        assert _eval_hotel_near_attractions(days) is False

    def test_no_location_data_skips(self):
        # 酒店坐标为 0,0 时跳过，返回 True
        hotel = {"name": "酒店", "location": {"longitude": 0, "latitude": 0}}
        days = [_day(hotel=hotel)]
        assert _eval_hotel_near_attractions(days) is True

    def test_empty_days_returns_true(self):
        assert _eval_hotel_near_attractions([]) is True

    def test_multiple_days_averaged(self):
        # 第一天很近，第二天也很近 → pass
        d1 = _day(date="2026-03-15", hotel=_hotel(lon=116.40, lat=39.92),
                  attractions=[_attraction(lon=116.40, lat=39.92)])
        d2 = _day(date="2026-03-16", hotel=_hotel(lon=116.41, lat=39.93),
                  attractions=[_attraction(lon=116.41, lat=39.93)])
        assert _eval_hotel_near_attractions([d1, d2]) is True


# ── _eval_weather_activity_match ─────────────────────────────

class TestWeatherActivityMatch:
    def _weather(self, date, day_weather):
        return {"date": date, "day_weather": day_weather, "night_weather": "晴"}

    def test_no_weather_passes(self):
        data = _full_data(weather_info=[])
        assert _eval_weather_activity_match(data) is True

    def test_sunny_weather_passes(self):
        data = _full_data(
            weather_info=[self._weather("2026-03-15", "晴")],
            days=[_day(date="2026-03-15", attractions=[_attraction("故宫")])]
        )
        assert _eval_weather_activity_match(data) is True

    def test_rainy_with_indoor_attr_passes(self):
        data = _full_data(
            weather_info=[self._weather("2026-03-15", "小雨")],
            days=[_day(date="2026-03-15",
                       attractions=[_attraction("国家博物馆", category="博物馆")])]
        )
        assert _eval_weather_activity_match(data) is True

    def test_rainy_all_outdoor_fails(self):
        data = _full_data(
            weather_info=[self._weather("2026-03-15", "大雨")],
            days=[_day(date="2026-03-15",
                       attractions=[_attraction("长城", category="自然风光")])]
        )
        assert _eval_weather_activity_match(data) is False

    def test_half_rainy_days_with_indoor_passes(self):
        """2天雨天，只要≥50%有室内景点即通过"""
        data = _full_data(
            weather_info=[
                self._weather("2026-03-15", "小雨"),
                self._weather("2026-03-16", "暴雨"),
            ],
            days=[
                _day(date="2026-03-15", attractions=[_attraction("国家博物馆", category="博物馆")]),
                _day(date="2026-03-16", attractions=[_attraction("长城", category="自然风光")]),
            ]
        )
        assert _eval_weather_activity_match(data) is True


# ── evaluate_rules 整体 ──────────────────────────────────────

class TestEvaluateRules:
    def test_none_data_all_false(self):
        rules = evaluate_rules(_request(), None)
        assert rules["json_parseable"] is False
        assert all(v is False for v in rules.values())

    def test_complete_data_passes_basic_rules(self):
        data = _full_data()
        rules = evaluate_rules(_request(), data)
        assert rules["json_parseable"] is True
        assert rules["has_daily_plan"] is True
        assert rules["has_tips"] is True
        assert rules["city_match"] is True
        assert rules["date_match"] is True
        assert rules["has_budget"] is True

    def test_trip_days_match(self):
        data = _full_data(days=[_day(), _day(date="2026-03-16")])
        rules = evaluate_rules(_request(travel_days=2), data)
        assert rules["trip_days_match"] is True

    def test_trip_days_mismatch(self):
        data = _full_data(days=[_day()])
        rules = evaluate_rules(_request(travel_days=3), data)
        assert rules["trip_days_match"] is False

    def test_missing_hotel_fails(self):
        day_no_hotel = _day()
        day_no_hotel["hotel"] = None
        data = _full_data(days=[day_no_hotel])
        rules = evaluate_rules(_request(), data)
        assert rules["has_hotel"] is False

    def test_missing_attractions_fails(self):
        day_no_attr = _day(attractions=[])
        data = _full_data(days=[day_no_attr])
        rules = evaluate_rules(_request(), data)
        assert rules["has_attractions"] is False

    def test_city_mismatch(self):
        data = _full_data(city="上海")
        rules = evaluate_rules(_request(city="北京"), data)
        assert rules["city_match"] is False

    def test_preference_reflect_no_prefs_passes(self):
        data = _full_data()
        rules = evaluate_rules(_request(preferences=[]), data)
        assert rules["preference_reflect"] is True

    def test_preference_reflect_matched(self):
        # 景点名称包含偏好关键词
        attr = _attraction(name="历史文化博物馆", category="历史文化")
        day = _day(attractions=[attr, attr, attr, attr])  # 4 个景点全部匹配
        data = _full_data(days=[day])
        rules = evaluate_rules(_request(preferences=["历史文化"]), data)
        assert rules["preference_reflect"] is True

    def test_preference_reflect_not_matched(self):
        # 景点完全不包含偏好关键词
        attr = _attraction(name="长城", category="自然风光")
        day = _day(attractions=[attr] * 5)
        data = _full_data(days=[day])
        rules = evaluate_rules(_request(preferences=["科技"]), data)
        assert rules["preference_reflect"] is False

    def test_has_budget_false_when_missing(self):
        data = _full_data(budget=None)
        rules = evaluate_rules(_request(), data)
        assert rules["has_budget"] is False

    def test_has_budget_false_when_no_total(self):
        data = _full_data(budget={"total_hotels": 200})
        rules = evaluate_rules(_request(), data)
        assert rules["has_budget"] is False

    def test_returns_all_13_rules(self):
        data = _full_data()
        rules = evaluate_rules(_request(), data)
        expected_keys = {
            "json_parseable", "has_daily_plan", "has_weather", "has_hotel",
            "has_attractions", "has_tips", "trip_days_match", "has_budget",
            "preference_reflect", "city_match", "date_match",
            "hotel_near_attractions", "weather_activity_match",
        }
        assert set(rules.keys()) == expected_keys
