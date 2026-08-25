import asyncio
import json
import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.core.config import settings
from app.llm.base import LLMProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """The sole OpenAI transport, with process-wide request control.

    Research jobs execute in FastAPI background threads and each starts its own
    event loop. A process-wide threading semaphore is therefore the safe global
    limiter; the async agent API waits for it without blocking its event loop.
    """

    _request_limiter = threading.BoundedSemaphore(1)
    _max_attempts = 3
    _base_backoff_seconds = 1.0

    def __init__(
        self,
        api_key: str | None = None,
        *,
        post: Callable[..., httpx.Response] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ):
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self._post_request = post or httpx.post
        self._sleep = sleep
        self._random_value = random_value

    async def complete_json(self, prompt: str, payload: dict, *, agent: str, max_output_tokens: int) -> dict:
        return await asyncio.to_thread(self._complete_json, prompt, payload, agent, max_output_tokens)

    def _complete_json(self, prompt: str, payload: dict, agent: str, max_output_tokens: int) -> dict:
        data = self._post_with_control(
            "/chat/completions",
            {
                "model": settings.openai_model,
                "response_format": {"type": "json_object"},
                "max_tokens": max_output_tokens,
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, default=str)}],
            },
            agent,
            settings.openai_model,
        )
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError
            return parsed
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMProviderError("LLM provider returned malformed structured output.") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post_with_control(
            "/embeddings",
            {"model": settings.openai_embedding_model, "input": texts},
            "rag_embedding",
            settings.openai_embedding_model,
        )
        try:
            vectors = [item["embedding"] for item in data["data"]]
            if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
                raise ValueError
            return vectors
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError("LLM provider returned malformed embeddings.") from exc

    def _post_with_control(self, path: str, payload: dict[str, Any], agent: str, model: str) -> dict:
        if not self.api_key:
            raise LLMProviderError("LLM provider is not configured.")
        # Retried calls remain inside the lock, so another agent cannot form a new
        # burst while this request is observing a server-provided Retry-After.
        with self._request_limiter:
            for attempt in range(self._max_attempts):
                try:
                    response = self._post_request(
                        f"https://api.openai.com/v1{path}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        raise ValueError("OpenAI response was not an object.")
                    self._log_usage(agent, model, data, response)
                    return data
                except httpx.HTTPStatusError as exc:
                    response = exc.response
                    if response.status_code == 429:
                        self._log_rate_limit(agent, model, response)
                    if not self._is_retryable_status(response.status_code) or attempt == self._max_attempts - 1:
                        raise LLMProviderError("LLM provider is unavailable or rejected the request.") from exc
                    self._wait_before_retry(agent, model, attempt, response.headers.get("retry-after"), response.status_code)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt == self._max_attempts - 1:
                        raise LLMProviderError("LLM provider is unavailable or rejected the request.") from exc
                    self._wait_before_retry(agent, model, attempt, None, None)
                except (ValueError, TypeError) as exc:
                    raise LLMProviderError("LLM provider is unavailable or returned malformed data.") from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code <= 599

    def _wait_before_retry(self, agent: str, model: str, attempt: int, retry_after: str | None, status_code: int | None) -> None:
        delay = self._retry_delay(attempt, retry_after)
        logger.warning(
            "openai_retry",
            extra={"agent": agent, "model": model, "attempt": attempt + 1, "retry_delay_seconds": round(delay, 3), "status": status_code, "error_category": "openai_transient_failure"},
        )
        self._sleep(delay)

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        try:
            if retry_after is not None and float(retry_after) >= 0:
                return float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, IndexError):
                pass
        return self._base_backoff_seconds * (2**attempt) + self._random_value()

    def _log_usage(self, agent: str, model: str, data: dict, response: httpx.Response) -> None:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        logger.info(
            "openai_usage",
            extra={"agent": agent, "model": model, "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"), "request_id": response.headers.get("x-request-id"), "rate_limit_remaining_tokens": response.headers.get("x-ratelimit-remaining-tokens")},
        )

    def _log_rate_limit(self, agent: str, model: str, response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) else {}
        headers = response.headers
        logger.warning(
            "openai_rate_limited",
            extra={
                "agent": agent, "model": model, "openai_error_type": error.get("type"), "openai_error_code": error.get("code"), "openai_error_message": error.get("message"), "request_id": headers.get("x-request-id"), "rate_limit_limit_requests": headers.get("x-ratelimit-limit-requests"), "rate_limit_remaining_requests": headers.get("x-ratelimit-remaining-requests"), "rate_limit_reset_requests": headers.get("x-ratelimit-reset-requests"), "rate_limit_limit_tokens": headers.get("x-ratelimit-limit-tokens"), "rate_limit_remaining_tokens": headers.get("x-ratelimit-remaining-tokens"), "rate_limit_reset_tokens": headers.get("x-ratelimit-reset-tokens"), "retry_after": headers.get("retry-after"), "error_category": "openai_rate_limit",
            },
        )
