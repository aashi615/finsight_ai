import asyncio
import json
import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from groq import Groq

from app.core.config import settings
from app.llm.base import LLMProviderError
from app.llm.local_embeddings import LocalEmbeddingProvider

logger = logging.getLogger(__name__)


class GroqProvider:
    """Groq chat provider with a process-wide serialized request budget."""

    _request_limiter = threading.BoundedSemaphore(1)
    _max_attempts = 3

    def __init__(self, api_key: str | None = None, *, client: Any | None = None, sleep: Callable[[float], None] = time.sleep, random_value: Callable[[], float] = random.random):
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.client = client or (Groq(api_key=self.api_key, max_retries=0, timeout=30.0) if self.api_key else None)
        self._sleep, self._random_value = sleep, random_value
        self._embeddings = LocalEmbeddingProvider()

    async def complete_json(self, prompt: str, payload: dict, *, agent: str, max_output_tokens: int) -> dict:
        return await asyncio.to_thread(self._complete_json, prompt, payload, agent, max_output_tokens)

    def _complete_json(self, prompt: str, payload: dict, agent: str, max_output_tokens: int) -> dict:
        if not self.client:
            raise LLMProviderError("Groq provider is not configured.", category="llm_provider_error")
        with self._request_limiter:
            for attempt in range(self._max_attempts):
                try:
                    response = self.client.chat.completions.create(model=settings.llm_model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, default=str)}], response_format={"type": "json_object"}, max_tokens=max_output_tokens, temperature=0.2)
                    content = response.choices[0].message.content
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise ValueError
                    usage = getattr(response, "usage", None)
                    logger.info("llm_usage", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None), "total_tokens": getattr(usage, "total_tokens", None), "request_id": getattr(response, "_request_id", None)})
                    return parsed
                except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                    raise LLMProviderError("Groq returned malformed structured output.", category="llm_invalid_response") from exc
                except Exception as exc:
                    category, retryable, retry_after = self._classify_error(exc)
                    self._log_error(agent, attempt + 1, category, exc)
                    if not retryable or attempt == self._max_attempts - 1:
                        raise LLMProviderError("Groq quota is exhausted." if category == "llm_quota_exhausted" else "Groq provider is temporarily unavailable.", category=category) from exc
                    delay = self._retry_delay(attempt, retry_after)
                    logger.warning("llm_retry", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "attempt": attempt + 1, "retry_delay_seconds": round(delay, 3), "error_category": category})
                    self._sleep(delay)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed(texts)

    def _classify_error(self, exc: Exception) -> tuple[str, bool, str | None]:
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        error_text = str(body or exc).lower()
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry_after = headers.get("retry-after")
        if "insufficient_quota" in error_text or "quota" in error_text or "billing" in error_text:
            return "llm_quota_exhausted", False, retry_after
        if status == 429:
            return "llm_rate_limit", True, retry_after
        if isinstance(status, int) and 500 <= status <= 599:
            return "llm_provider_error", True, retry_after
        if exc.__class__.__name__ in {"APITimeoutError", "APIConnectionError"}:
            return "llm_timeout", True, retry_after
        return "llm_provider_error", False, retry_after

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        try:
            if retry_after is not None and float(retry_after) >= 0:
                return float(retry_after)
        except ValueError:
            pass
        return (2**attempt) + self._random_value()

    def _log_error(self, agent: str, attempt: int, category: str, exc: Exception) -> None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        logger.warning("llm_request_failed", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "attempt": attempt, "status": getattr(exc, "status_code", None), "request_id": headers.get("x-request-id"), "groq_error_type": error.get("type") if isinstance(error, dict) else type(exc).__name__, "groq_error_code": error.get("code") if isinstance(error, dict) else None, "groq_error_message": error.get("message") if isinstance(error, dict) else str(exc), "error_category": category})
