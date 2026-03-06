"""LLM服务模块 - 基于 LangChain / langchain-openai"""

import logging
from langchain_openai import ChatOpenAI
from ..config import get_settings

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm() -> ChatOpenAI:
    """
    获取 ChatOpenAI 实例（单例）。

    直接读取 .env 中的原始字段，不做 remap：
      LLM_API_KEY   - API 密钥
      LLM_BASE_URL  - 接口地址（空则使用 OpenAI 默认）
      LLM_MODEL_ID  - 模型名称
      LLM_TIMEOUT   - 请求超时秒数
    """
    global _llm_instance

    if _llm_instance is None:
        s = get_settings()

        if not s.llm_api_key:
            raise ValueError(
                "LLM_API_KEY 未配置，请在项目根目录 .env 中设置 LLM_API_KEY"
            )

        _llm_instance = ChatOpenAI(
            api_key=s.llm_api_key,
            base_url=s.llm_base_url or None,
            model=s.llm_model_id,
            temperature=0.3,
            timeout=s.llm_timeout,
        )

        logger.info("LLM 初始化成功 | 模型=%s | base_url=%s", s.llm_model_id, s.llm_base_url or "(默认 OpenAI)")
    return _llm_instance


def reset_llm() -> None:
    """  重置实例（用于测试）"""
    global _llm_instance
    _llm_instance = None

