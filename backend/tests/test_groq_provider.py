import asyncio
import json
from types import SimpleNamespace

import pytest

from app.llm.base import LLMProviderError
from app.llm.groq_provider import GroqProvider, extract_json_object, normalize_json_response


def completion(payload=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload or {"ok": True})) )], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18), _request_id="req_groq")


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


def test_groq_market_request_uses_local_json_contract_with_hidden_reasoning():
    fake = client([completion()])
    provider = GroqProvider(api_key="test", client=fake)
    asyncio.run(provider.complete_json("Return ONLY valid JSON.", {"a": 1}, agent="market_analyst", max_output_tokens=900))
    request = fake.chat.completions.requests[0]
    assert request["reasoning_format"] == "hidden"
    assert "response_format" not in request
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
            message=SimpleNamespace(content='{"ok": true}', reasoning_content="internal reasoning", tool_calls=None),
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        _request_id="req_gpt_oss",
    )
    fake = client([sdk_response])
    provider = GroqProvider(api_key="test", client=fake)
    assert asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=900)) == {"ok": True}


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
