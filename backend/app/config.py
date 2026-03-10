"""配置管理模块"""

import logging
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 项目根目录 = trip-planner/
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# 加载根目录 .env（不覆盖已经存在的环境变量，适配 CI/CD 注入场景）
load_dotenv(_ENV_FILE, override=False)


class Settings(BaseSettings):
    """
    应用配置——字段名即环境变量名，零中间转换。

    监控 .env 中的字段名：
      LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID / LLM_TIMEOUT
      AMAP_API_KEY
      HOST / PORT / CORS_ORIGINS / LOG_LEVEL
    """

    # 应用基本信息
    app_name: str = "LangGraph智能旅行助手"
    app_version: str = "2.1.0"
    debug: bool = False

    # 服务器
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # 日志
    log_level: str = "INFO"

    # 高德地图（由 backend 传递给 Amap MCP Server 子进程）
    amap_api_key: str = ""

    # Amap MCP Server 路径（空则自动定位）
    mcp_server_path: str = ""

    # Unsplash
    unsplash_access_key: str = ""

    # Redis 缓存（REDIS_URL 为空则禁用缓存，不影响主流程）
    redis_url: str = "redis://localhost:6379"          # e.g. redis://localhost:6379/0
    trip_cache_ttl: int = 86400  # 缓存有效期（秒），默认 24 小时

    # LLM（字段名与 .env 中一致）
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model_id: str = "gpt-4o"
    llm_timeout: int = 300

    # 限流（每 IP 每分钟规划请求数，可通过环境变量 RATE_LIMIT_PER_MINUTE 调整）
    rate_limit_per_minute: int = 10

    class Config:
        env_file = str(_ENV_FILE)
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


# 全局单例
settings = Settings()


def get_settings() -> Settings:
    return settings


def get_mcp_server_path() -> str:
    """MCP Server 脚本绝对路径，静态计算无需配置运行时解析"""
    configured = settings.mcp_server_path
    if configured:
        return configured
    return str(_PROJECT_ROOT / "mcp-server" / "server.py")


def validate_config():
    """Warn about missing optional keys, raise on fatal ones."""
    warnings = []
    if not settings.amap_api_key:
        warnings.append("AMAP_API_KEY 未配置，Amap MCP 工具将无法调用")
    if not settings.llm_api_key:
        warnings.append("LLM_API_KEY 未配置，LLM 功能将无法使用")
    mcp_path = get_mcp_server_path()
    if not Path(mcp_path).exists():
        warnings.append(f"MCP Server 未找到: {mcp_path}")
    if warnings:
        for w in warnings:
            logger.warning(w)
    return True


def print_config():
    logger.info(f"应用: {settings.app_name} v{settings.app_version}")


def setup_logging():
    """配置全局日志格式与级别"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"LLM API Key: {'已配置' if settings.llm_api_key else '未配置'}")
    print(f"LLM Base URL: {settings.llm_base_url or '(默认 OpenAI)'}")
    print(f"LLM Model: {settings.llm_model_id}")
    print(f"Amap API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"Amap MCP Server: {get_mcp_server_path()}")
    print(f"日志级别: {settings.log_level}")
