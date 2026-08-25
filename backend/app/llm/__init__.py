from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.groq_provider import GroqProvider
from app.llm.openai_provider import OpenAIProvider


def get_llm_provider() -> LLMProvider:
    return GroqProvider() if settings.llm_provider == "groq" else OpenAIProvider()
