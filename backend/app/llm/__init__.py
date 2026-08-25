from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.groq_provider import GroqProvider
from app.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


def get_llm_provider() -> LLMProvider:
    logger.info("llm provider selected", extra={"provider": settings.llm_provider, "model": settings.llm_model if settings.llm_provider == "groq" else settings.openai_model, "api_key_configured": bool(settings.groq_api_key) if settings.llm_provider == "groq" else bool(settings.openai_api_key)})
    return GroqProvider() if settings.llm_provider == "groq" else OpenAIProvider()
import logging
