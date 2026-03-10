"""旅行规划API路由"""

import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from ...models.schemas import (
    TripRequest,
    TripPlanResponse,
    ErrorResponse
)
from ...agents.trip_planner_agent import get_trip_planner_agent
from ...services.photo_service import batch_search_photos
from ...services.cache import make_cache_key, get_cached_plan, set_cached_plan
from ...config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划
    """
    try:
        mode = request.execution_mode or "parallel_cache"
        logger.info("收到旅行规划请求 | 城市=%s 日期=%s~%s 天数=%d 模式=%s",
                    request.city, request.start_date, request.end_date, request.travel_days, mode)

        # 仅 parallel_cache 模式使用缓存
        cache_hit = False
        if mode == "parallel_cache":
            cache_key = make_cache_key(request)
            cached = await get_cached_plan(cache_key)
            if cached:
                from ...models.schemas import TripPlan
                try:
                    return TripPlanResponse(
                        success=True,
                        message="旅行计划生成成功（缓存）",
                        data=TripPlan(**cached),
                        cache_hit=True,
                        execution_mode=mode,
                        fallback_used=False,
                    )
                except Exception:
                    pass

        # 确定 agent 模式
        mode_to_agent = {
            "sequential": "sequential",
            "parallel_react": "parallel_react",
            "parallel_direct": "parallel_direct",
            "parallel_agent": "parallel_agent",
            "parallel_cache": "parallel_agent",
            # 兼容旧名
            "parallel_no_cache": "parallel_direct",
        }
        agent_mode = mode_to_agent.get(mode, "parallel_agent")

        agent = get_trip_planner_agent()
        trip_plan = await agent.plan_trip(request, mode=agent_mode)

        # 检测是否使用了 fallback（fallback 生成的景点名含固定模式）
        fallback_used = False
        if trip_plan.days and trip_plan.days[0].attractions:
            first_attr = trip_plan.days[0].attractions[0].name
            if "景点1" in first_attr or "景点2" in first_attr:
                fallback_used = True

        # 仅 parallel_cache 模式写缓存
        if mode == "parallel_cache":
            await set_cached_plan(cache_key, trip_plan.model_dump(), get_settings().trip_cache_ttl)

        logger.info("旅行计划生成成功 | 城市=%s 模式=%s", request.city, mode)

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            data=trip_plan,
            cache_hit=cache_hit,
            execution_mode=mode,
            fallback_used=fallback_used,
        )

    except Exception as e:
        logger.error("生成旅行计划失败: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
        )


@router.post(
    "/plan/stream",
    summary="生成旅行计划 (SSE 流式)",
    description="通过 Server-Sent Events 实时推送 Agent 执行进度，最终返回完整旅行计划",
)
async def plan_trip_stream(request: TripRequest):
    """
    SSE 流式生成旅行计划

    前端通过 fetch + ReadableStream 接收实时进度事件，
    每个事件格式: {step, percent, message[, data][, error]}
    """
    from ...services.progress import ProgressEmitter

    logger.info("收到 SSE 旅行规划请求 | 城市=%s 日期=%s~%s",
                request.city, request.start_date, request.end_date)

    async def event_generator():
        emitter = ProgressEmitter()

        async def run_agent():
            try:
                # 缓存查找
                cache_key = make_cache_key(request)
                cached = await get_cached_plan(cache_key)
                if cached:
                    from ...models.schemas import TripPlan
                    try:
                        trip_plan = TripPlan(**cached)
                        await emitter.emit_done(trip_plan.model_dump())
                        return
                    except Exception:
                        pass

                agent = get_trip_planner_agent()
                trip_plan = await agent.plan_trip(request, progress_emitter=emitter)

                # 写缓存
                await set_cached_plan(cache_key, trip_plan.model_dump(), get_settings().trip_cache_ttl)

                await emitter.emit_done(trip_plan.model_dump())
            except Exception as e:
                logger.error("SSE Agent 执行失败: %s", e, exc_info=True)
                await emitter.emit_error(str(e))

        task = asyncio.create_task(run_agent())

        try:
            while True:
                try:
                    event = await asyncio.wait_for(emitter.queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    timeout_evt = {"step": "error", "percent": -1,
                                   "message": "请求超时(5分钟)", "error": "请求超时"}
                    yield f"data: {json.dumps(timeout_evt, ensure_ascii=False)}\n\n"
                    break

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                if event.get("step") in ("done", "error"):
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        # 检查Agent是否可用
        agent = get_trip_planner_agent()

        return {
            "status": "healthy",
            "service": "trip-planner",
            "framework": "LangGraph"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )


# ---------- Unsplash 图片批量查询 ----------

class PhotoRequest(BaseModel):
    names: List[str]
    city: str = ""


@router.post(
    "/photos",
    summary="批量获取景点图片",
    description="通过 Unsplash 批量搜索景点图片",
)
async def get_attraction_photos(req: PhotoRequest):
    """
    返回 {name: url} 映射
    """
    try:
        photos = await batch_search_photos(req.names, req.city)
        return {"success": True, "data": photos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

