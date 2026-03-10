"""FastAPI主应用"""

import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import get_settings, validate_config, print_config, setup_logging
from .routes import trip
from ..agents.trip_planner_agent import shutdown_mcp_client

logger = logging.getLogger(__name__)

# 获取配置
settings = get_settings()

# 初始化日志
setup_logging()


# ============================================================
# Lifespan（替代已废弃的 on_event）
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("%s v%s 启动中...", settings.app_name, settings.app_version)
    print_config()
    try:
        validate_config()
        logger.info("配置验证通过")
    except ValueError as e:
        logger.error("配置验证失败: %s", e)
        raise
    logger.info("API文档: http://localhost:%d/docs", settings.port)
    yield
    # 应用关闭时清理 MCP 子进程
    await shutdown_mcp_client()
    logger.info("应用关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于 LangGraph + Amap MCP Server 的智能旅行规划助手 API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ============================================================
# 中间件
# ============================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求注入唯一 request_id，记录请求耗时"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.0fms) [rid=%s]",
            request.method, request.url.path, response.status_code, elapsed_ms, request_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """简易内存限流：同一 IP 每分钟最多 N 次规划请求"""
    RATE_LIMIT = settings.rate_limit_per_minute
    WINDOW = 60  # 秒

    def __init__(self, app):
        super().__init__(app)
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/trip/plan"):
            client_ip = request.client.host if request.client else "unknown"
            now = time.monotonic()
            window = [t for t in self._requests[client_ip] if now - t < self.WINDOW]
            if len(window) >= self.RATE_LIMIT:
                logger.warning("限流触发: IP=%s, path=%s", client_ip, request.url.path)
                return Response(
                    content='{"detail":"请求过于频繁，请稍后再试"}',
                    status_code=429,
                    media_type="application/json",
                )
            window.append(now)
            self._requests[client_ip] = window
        return await call_next(request)


app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware)

# 注册路由
app.include_router(trip.router, prefix="/api")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }

