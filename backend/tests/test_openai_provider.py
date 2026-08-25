import asyncio
import json
import threading
import time

import httpx
import pytest

from app.llm.base import LLMProviderError
from app.llm.openai_provider import OpenAIProvider


def response(status_code: int, payload: dict, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status_code, json=payload, headers=headers, request=request)


def completion(value: dict | None = None, headers: dict | None = None) -> httpx.Response:
    return response(200, {"choices": [{"message": {"content": json.dumps(value or {"ok": True})}}], "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}, headers)


def test_global_openai_limiter_serializes_concurrent_agent_requests():
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def post(*args, **kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return completion()

    provider = OpenAIProvider(api_key="test", post=post)

    async def run():
        return await asyncio.gather(*(provider.complete_json("system", {"n": index}, agent=f"agent-{index}", max_output_tokens=900) for index in range(3)))

    assert asyncio.run(run()) == [{"ok": True}] * 3
    assert maximum_active == 1


def test_429_retries_with_retry_after_and_logs_rate_limit_details(caplog):
    calls = 0
    delays: list[float] = []

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(429, {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "Rate limit reached."}}, {"retry-after": "2", "x-request-id": "req_429", "x-ratelimit-remaining-tokens": "0"})
        return completion()

    provider = OpenAIProvider(api_key="test", post=post, sleep=delays.append, random_value=lambda: 0.5)
    assert asyncio.run(provider.complete_json("system", {}, agent="market_analyst", max_output_tokens=900)) == {"ok": True}
    assert calls == 2
    assert delays == [2.0]
    rate_limit = next(record for record in caplog.records if record.message == "openai_rate_limited")
    assert rate_limit.agent == "market_analyst"
    assert rate_limit.request_id == "req_429"
    assert rate_limit.rate_limit_remaining_tokens == "0"


def test_retry_backoff_uses_exponential_delay_plus_jitter():
    provider = OpenAIProvider(api_key="test", post=lambda *args, **kwargs: completion(), random_value=lambda: 0.25)
    assert provider._retry_delay(0, None) == 1.25
    assert provider._retry_delay(1, None) == 2.25
    assert provider._retry_delay(0, "3") == 3.0


def test_non_retryable_openai_error_is_not_retried():
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return response(400, {"error": {"type": "invalid_request_error", "message": "Bad request."}})

    provider = OpenAIProvider(api_key="test", post=post, sleep=lambda _: None)
    with pytest.raises(LLMProviderError):
        asyncio.run(provider.complete_json("system", {}, agent="news_analyst", max_output_tokens=900))
    assert calls == 1


def test_successful_response_logs_actual_token_usage_and_request_id(caplog):
    provider = OpenAIProvider(api_key="test", post=lambda *args, **kwargs: completion(headers={"x-request-id": "req_usage", "x-ratelimit-remaining-tokens": "1234"}))
    asyncio.run(provider.complete_json("system", {}, agent="research_synthesizer", max_output_tokens=1500))
    usage = next(record for record in caplog.records if record.message == "openai_usage")
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (12, 7, 19)
    assert usage.request_id == "req_usage"
    assert usage.rate_limit_remaining_tokens == "1234"
