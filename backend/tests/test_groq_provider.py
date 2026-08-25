import asyncio
import json
from types import SimpleNamespace

import pytest

from app.llm.base import LLMProviderError
from app.llm.groq_provider import GroqProvider, TokenBudgetManager, extract_json_object, normalize_json_response


def completion(payload=None, *, finish_reason="stop", output_tokens=7, content=None, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload or {"ok": True}) if content is None else content, reasoning=reasoning), finish_reason=finish_reason)], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=output_tokens, total_tokens=18), _request_id="req_groq")


class Chat:
    def __init__(self, outcomes): self.outcomes, self.calls, self.requests = iter(outcomes), 0, []
    def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        value = next(self.outcomes)
        if isinstance(value, Exception): raise value
        return value


def client(outcomes):
    return SimpleNamespace(chat=SimpleNamespace(completions=Chat(outcomes)))


class ProviderException(Exception):
    def __init__(self, status_code, body):
        self.status_code, self.body, self.response = status_code, body, SimpleNamespace(headers={})


def test_groq_provider_uses_configured_model_and_parses_json(monkeypatch):
    monkeypatch.setattr("app.llm.groq_provider.settings.llm_model", "openai/gpt-oss-120b")
    fake = client([completion()])
    provider = GroqProvider(api_key="test", client=fake)
    assert asyncio.run(provider.complete_json("system", {"a": 1}, agent="market_analyst", max_output_tokens=900)) == {"ok": True}
    assert fake.chat.completions.calls == 1


def test_groq_complete_json_with_stop_succeeds():
    provider = GroqProvider(api_key="test", client=client([completion({"ok": True}, finish_reason="stop")]))
    assert asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=2000)) == {"ok": True}


def test_groq_incomplete_response_retries_once_with_a_compact_request():
    fake = client([completion({"partial": True}, finish_reason="length", output_tokens=2000), completion({"ok": True}, finish_reason="stop")])
    provider = GroqProvider(api_key="test", client=fake)
    assert asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=2000)) == {"ok": True}
    assert fake.chat.completions.calls == 2
    assert "smallest valid JSON" in fake.chat.completions.requests[1]["messages"][0]["content"]
    assert fake.chat.completions.requests[0]["messages"] != fake.chat.completions.requests[1]["messages"]


def test_groq_incomplete_response_twice_fails_without_infinite_retries():
    fake = client([completion({"partial": True}, finish_reason="length", output_tokens=2000), completion({"still": "partial"}, finish_reason="length", output_tokens=2000)])
    provider = GroqProvider(api_key="test", client=fake)
    with pytest.raises(LLMProviderError, match="incomplete") as error:
        asyncio.run(provider.complete_json("prompt", {}, agent="news_analyst", max_output_tokens=2000))
    assert error.value.category == "llm_incomplete_response"
    assert fake.chat.completions.calls == 2


def test_groq_market_request_uses_supported_low_reasoning_configuration():
    fake = client([completion()])
    provider = GroqProvider(api_key="test", client=fake)
    asyncio.run(provider.complete_json("Return ONLY valid JSON.", {"a": 1}, agent="market_analyst", max_output_tokens=900))
    request = fake.chat.completions.requests[0]
    assert request["reasoning_effort"] == "low"
    assert request["include_reasoning"] is False
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"] == [{"role": "user", "content": "Return ONLY valid JSON.\n\nInput data:\n{\"a\": 1}"}]


def test_groq_malformed_json_is_a_clear_structured_output_error():
    malformed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])
    provider = GroqProvider(api_key="test", client=client([malformed]))
    with pytest.raises(LLMProviderError, match="malformed JSON output at json_extraction") as error:
        asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=900))
    assert error.value.category == "llm_invalid_response"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (' {"ok": true} \n', {"ok": True}),
        ('```json\n{"ok": true}\n```', {"ok": True}),
        ('json\n{"ok": true}', {"ok": True}),
    ],
)
def test_groq_extracts_supported_json_wrappers(content, expected):
    assert json.loads(normalize_json_response(content)) == expected


def test_groq_extracts_json_surrounded_by_explanatory_text_and_braces_in_strings():
    content = 'Here is the result: ```json\n{"summary": "A {brace} in a string"}\n``` Thanks.'
    assert json.loads(extract_json_object(content)) == {"summary": "A {brace} in a string"}


@pytest.mark.parametrize("content, stage", [("", "empty_response"), (None, "response_content"), ("Explanation without JSON", "json_extraction")])
def test_groq_extraction_failures_are_typed(content, stage):
    with pytest.raises(Exception) as error:
        extract_json_object(content)
    assert getattr(error.value, "stage", None) == stage


def test_groq_malformed_json_reaches_the_controlled_json_decode_stage():
    provider = GroqProvider(api_key="test", client=client([]))
    with pytest.raises(Exception) as error:
        provider._parse_json_response('{"broken": }', "market_analyst")
    assert getattr(error.value, "stage", None) == "json_decode"


def test_groq_rejects_array_when_an_object_is_required():
    provider = GroqProvider(api_key="test", client=client([]))
    with pytest.raises(Exception) as error:
        provider._parse_json_response('[{"ok": true}]', "market_analyst")
    assert getattr(error.value, "stage", None) == "json_object"


def test_groq_sdk_gpt_oss_message_shape_parses_final_content_not_reasoning():
    sdk_response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content='{"ok": true}', reasoning="internal reasoning", tool_calls=None),
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        _request_id="req_gpt_oss",
    )
    fake = client([sdk_response])
    provider = GroqProvider(api_key="test", client=fake)
    assert asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=900)) == {"ok": True}


def test_groq_response_preview_is_reserved_for_failure_diagnostics():
    choice = SimpleNamespace(finish_reason="stop")
    message = SimpleNamespace(reasoning_content=None, tool_calls=None)
    usage = SimpleNamespace(completion_tokens=4)
    assert GroqProvider._response_details('{"safe": true}', choice, message, "req", usage)["content_preview"] is None
    assert GroqProvider._response_details('{"safe": true}', choice, message, "req", usage, include_preview=True)["content_preview"] == '{"safe": true}'


def test_groq_json_validation_failure_is_categorized_as_invalid_response():
    rejected = ProviderException(400, {"error": {"type": "invalid_request_error", "code": "json_validate_failed", "message": "Failed to validate JSON"}})
    provider = GroqProvider(api_key="test", client=client([rejected]))
    with pytest.raises(LLMProviderError, match="rejected the requested JSON output") as error:
        asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=900))
    assert error.value.category == "llm_invalid_response"


def test_groq_rate_limit_retries_but_quota_does_not():
    rate_limited = ProviderException(429, {"error": {"code": "rate_limit_exceeded"}})
    fake = client([rate_limited, completion()])
    delays = []
    provider = GroqProvider(api_key="test", client=fake, sleep=delays.append, random_value=lambda: 0)
    assert asyncio.run(provider.complete_json("system", {}, agent="news_analyst", max_output_tokens=900)) == {"ok": True}
    assert fake.chat.completions.calls == 2 and delays == [1]

    quota = ProviderException(429, {"error": {"code": "insufficient_quota", "message": "quota exhausted"}})
    fake = client([quota])
    provider = GroqProvider(api_key="test", client=fake, sleep=lambda _: (_ for _ in ()).throw(AssertionError("must not retry")))
    with pytest.raises(LLMProviderError, match="quota") as error:
        asyncio.run(provider.complete_json("system", {}, agent="news_analyst", max_output_tokens=900))
    assert error.value.category == "llm_quota_exhausted"
    assert fake.chat.completions.calls == 1


def test_groq_local_embeddings_do_not_require_external_api():
    provider = GroqProvider(api_key="test", client=client([]))
    vector = provider.embed(["NVIDIA earnings growth"])[0]
    assert len(vector) == 256
    assert any(vector)


def test_groq_enforces_agent_output_budgets():
    fake = client([completion(), completion(), completion(), completion()])
    provider = GroqProvider(api_key="test", client=fake, token_budget=TokenBudgetManager(7000))
    for agent, requested, expected in [
        ("news_analyst", 4096, 1000),
        ("market_analyst", 4913, 1000),
        ("document_rag_agent", 2000, 1000),
        ("research_synthesizer", 5000, 1000),
    ]:
        asyncio.run(provider.complete_json("prompt", {}, agent=agent, max_output_tokens=requested))
        assert fake.chat.completions.requests[-1]["max_completion_tokens"] == expected


def test_tpm_guard_delays_when_request_would_exceed_safe_limit():
    budget = TokenBudgetManager(7000)
    assert budget.reserve_or_delay(6500) == 0
    assert budget.reserve_or_delay(1200) > 0


def test_token_429_respects_retry_after_and_max_retry_count():
    limited = ProviderException(429, {"error": {"type": "tokens", "code": "rate_limit_exceeded"}})
    limited.response.headers = {"retry-after": "3"}
    fake = client([limited, completion()])
    delays = []
    provider = GroqProvider(api_key="test", client=fake, sleep=delays.append, random_value=lambda: 0, token_budget=TokenBudgetManager(7000))
    assert asyncio.run(provider.complete_json("prompt", {}, agent="news_analyst", max_output_tokens=700)) == {"ok": True}
    assert delays == [3] and fake.chat.completions.calls == 2


def test_empty_gpt_oss_length_response_uses_non_thinking_fallback_not_same_request():
    fake = client([
        completion(finish_reason="length", output_tokens=520, content="", reasoning=None),
        completion({"ok": True}),
    ])
    provider = GroqProvider(api_key="test", client=fake, token_budget=TokenBudgetManager(7000))
    assert asyncio.run(provider.complete_json("prompt", {"articles": ["x" * 400]}, agent="news_analyst", max_output_tokens=480)) == {"ok": True}
    first, second = fake.chat.completions.requests
    assert first["model"] == "openai/gpt-oss-20b"
    assert second["model"] == "qwen/qwen3.6-27b"
    assert second["reasoning_effort"] == "none"
    assert first["messages"] != second["messages"]
