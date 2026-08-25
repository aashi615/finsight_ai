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


class StructuredOutputParseError(ValueError):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


def normalize_json_response(content: str) -> str:
    """Extract the first balanced JSON object/array from a model response.

    This is deliberately structural only: JSON syntax is validated by json.loads
    in the provider immediately after normalization.
    """
    if not isinstance(content, str):
        raise StructuredOutputParseError("response_content")
    text = content.strip()
    if not text:
        raise StructuredOutputParseError("empty_response")

    for start, character in enumerate(text):
        if character not in "{[":
            continue
        end = _balanced_json_end(text, start)
        if end is not None:
            return text[start:end]
    raise StructuredOutputParseError("json_extraction")


def _balanced_json_end(text: str, start: int) -> int | None:
    """Find a JSON container end while respecting quoted strings and escapes."""
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            stack.append("}")
        elif character == "[":
            stack.append("]")
        elif character in "}]":
            if character != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def extract_json_object(text: str) -> str:
    """Backward-compatible alias for the provider JSON normalizer."""
    return normalize_json_response(text)


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
            transport_attempt = 0
            incomplete_retried = False
            active_prompt = prompt
            request_attempt = 0
            while transport_attempt < self._max_attempts:
                request_attempt += 1
                request_id = None
                response_details: dict[str, Any] = {}
                choice = message = None
                content = None
                usage = None
                try:
                    request = {
                        "model": settings.llm_model,
                        "messages": self._messages(active_prompt, payload, agent),
                        "max_tokens": max_output_tokens,
                        "temperature": 0.2,
                        # GPT-OSS supports hidden reasoning. We intentionally do not
                        # use response_format/json_object here: production proves that
                        # Groq's server-side JSON validator rejects these nested agent
                        # contracts before the application can validate them locally.
                        "reasoning_format": "hidden",
                    }
                    response = self.client.chat.completions.create(**request)
                    request_id = getattr(response, "_request_id", None)
                    choice = response.choices[0]
                    message = choice.message
                    content = getattr(message, "content", None)
                    usage = getattr(response, "usage", None)
                    response_details = self._response_details(content, choice, message, request_id, usage)
                    logger.info("groq_response_received", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, **response_details})
                    if getattr(choice, "finish_reason", None) == "length":
                        logger.warning("groq_incomplete_response", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "attempt": request_attempt, **self._response_details(content, choice, message, request_id, usage, include_preview=True)})
                        if not incomplete_retried:
                            incomplete_retried = True
                            active_prompt = f"{prompt}\n\nRetry instruction: return a shorter complete JSON object now. Omit all nonessential wording; do not truncate."
                            continue
                        raise LLMProviderError("Groq response was incomplete because it reached the output token limit.", category="llm_incomplete_response")
                    parsed = self._parse_json_response(content, agent, request_id, response_details)
                    logger.info("llm_usage", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None), "total_tokens": getattr(usage, "total_tokens", None), "request_id": getattr(response, "_request_id", None)})
                    return parsed
                except (StructuredOutputParseError, TypeError, KeyError, IndexError) as exc:
                    stage = getattr(exc, "stage", "response_content")
                    event = {"empty_response": "groq_empty_response", "json_extraction": "groq_json_extraction_failed", "json_decode": "groq_json_decode_failed"}.get(stage, "groq_json_parse_failed")
                    diagnostics = self._response_details(content, choice, message, request_id, usage, include_preview=True) if choice is not None else response_details
                    logger.warning(event, extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "request_id": request_id, "response_parsing_stage": stage, **diagnostics})
                    raise LLMProviderError(f"Groq returned malformed JSON output at {stage}.", category="llm_invalid_response") from exc
                except LLMProviderError:
                    raise
                except Exception as exc:
                    category, retryable, retry_after = self._classify_error(exc)
                    self._log_error(agent, transport_attempt + 1, category, exc)
                    if not retryable or transport_attempt == self._max_attempts - 1:
                        message = "Groq quota is exhausted." if category == "llm_quota_exhausted" else "Groq rejected the requested JSON output." if category == "llm_invalid_response" else "Groq provider is temporarily unavailable."
                        raise LLMProviderError(message, category=category) from exc
                    delay = self._retry_delay(transport_attempt, retry_after)
                    logger.warning("llm_retry", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "attempt": transport_attempt + 1, "retry_delay_seconds": round(delay, 3), "error_category": category})
                    self._sleep(delay)
                    transport_attempt += 1
        raise AssertionError("unreachable")

    @staticmethod
    def _messages(prompt: str, payload: dict, agent: str) -> list[dict[str, str]]:
        payload_json = json.dumps(payload, default=str)
        return [{"role": "user", "content": f"{prompt}\n\nInput data:\n{payload_json}"}]

    @staticmethod
    def _parse_json_response(content: Any, agent: str, request_id: str | None = None, response_details: dict[str, Any] | None = None) -> dict:
        extracted = normalize_json_response(content)
        logger.info("groq_json_parse_attempt", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "request_id": request_id, "response_parsing_stage": "before_json_loads", **(response_details or {})})
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise StructuredOutputParseError("json_decode") from exc
        if not isinstance(parsed, dict):
            raise StructuredOutputParseError("json_object")
        logger.info("llm_response_parsed", extra={"provider": "groq", "agent": agent, "model": settings.llm_model, "request_id": request_id, "response_parsing_stage": "json_loaded"})
        return parsed

    @staticmethod
    def _response_details(content: Any, choice: Any, message: Any, request_id: str | None, usage: Any, *, include_preview: bool = False) -> dict[str, Any]:
        preview = content[:1000] if include_preview and isinstance(content, str) else None
        return {
            "status": 200,
            "request_id": request_id,
            "finish_reason": getattr(choice, "finish_reason", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "content_type": type(content).__name__ if content is not None else None,
            "content_length": len(content) if isinstance(content, str) else None,
            "content_preview": preview,
            "content_empty": not bool(content and content.strip()) if isinstance(content, str) else content is None,
            "reasoning_content_present": bool(getattr(message, "reasoning_content", None)),
            "content_has_code_fence": "```" in content if isinstance(content, str) else False,
            "tool_calls_present": bool(getattr(message, "tool_calls", None)),
            "response_format": "none",
            "reasoning_format": "hidden",
        }

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
        if "json_validate_failed" in error_text or "generated json does not match" in error_text:
            return "llm_invalid_response", False, retry_after
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
