import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from app.llm.base import LLMProviderError
from app.llm.groq_provider import GroqProvider, TokenBudgetManager, extract_json_object, normalize_json_response
from app.core.config import settings


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
    fake = client([
        completion({"partial": True}, finish_reason="length", output_tokens=2000),
        completion({"still": "partial"}, finish_reason="length", output_tokens=2000),
        completion({"qwen": "partial"}, finish_reason="length", output_tokens=2000),
    ])
    provider = GroqProvider(api_key="test", client=fake)
    with pytest.raises(LLMProviderError, match="incomplete") as error:
        asyncio.run(provider.complete_json("prompt", {}, agent="news_analyst", max_output_tokens=2000))
    assert error.value.category == "llm_incomplete_response"
    assert fake.chat.completions.calls == 3
    assert fake.chat.completions.requests[-1]["model"] == "qwen/qwen3.6-27b"


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
    provider = GroqProvider(api_key="test", client=client([malformed, malformed, malformed]))
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
    fake = client([rejected, completion()])
    provider = GroqProvider(api_key="test", client=fake)
    assert asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=900)) == {"ok": True}
    assert [request["model"] for request in fake.chat.completions.requests] == ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]


def test_groq_rate_limit_retries_but_quota_does_not():
    rate_limited = ProviderException(429, {"error": {"code": "rate_limit_exceeded"}})
    fake = client([rate_limited, completion()])
    delays = []
    provider = GroqProvider(api_key="test", client=fake, sleep=delays.append, random_value=lambda: 0)
    assert asyncio.run(provider.complete_json("system", {}, agent="news_analyst", max_output_tokens=900)) == {"ok": True}
    assert fake.chat.completions.calls == 2 and delays == [1.25]

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
        ("news_analyst", 4096, 800),
        ("market_analyst", 4913, 900),
        ("document_rag_agent", 2000, 800),
        ("research_synthesizer", 9000, 1600),
    ]:
        asyncio.run(provider.complete_json("prompt", {}, agent=agent, max_output_tokens=requested))
        assert fake.chat.completions.requests[-1]["max_completion_tokens"] == expected


def test_tpm_guard_delays_when_request_would_exceed_safe_limit():
    budget = TokenBudgetManager(7000)
    assert budget.reserve_or_delay(6500) == 0
    assert budget.reserve_or_delay(1200) > 0


def test_concurrent_agents_share_one_tpm_budget_without_over_reserving():
    budget = TokenBudgetManager(7000)
    barrier = threading.Barrier(3)
    results = []
    def reserve():
        barrier.wait()
        results.append(budget.reserve_or_delay(4000))
    workers = [threading.Thread(target=reserve) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()
    assert results.count(0.0) == 1
    assert sum(delay > 0 for delay in results) == 1
    assert budget.available() == 3000


def test_low_remaining_tpm_waits_until_minimum_viable_output_budget_before_send():
    now = [0.0]
    budget = TokenBudgetManager(7000, clock=lambda: now[0])
    assert budget.reserve_or_delay(6900) == 0
    fake = client([completion()])
    waits = []
    def advance(seconds):
        waits.append(seconds)
        now[0] += seconds
    provider = GroqProvider(api_key="test", client=fake, sleep=advance, token_budget=budget)
    assert asyncio.run(provider.complete_json("prompt", {}, agent="news_analyst", max_output_tokens=800)) == {"ok": True}
    assert waits and sum(waits) >= 60
    assert fake.chat.completions.requests[0]["max_completion_tokens"] >= 300


def test_request_uses_current_available_output_budget_instead_of_waiting_for_configured_cap():
    budget = TokenBudgetManager(7000)
    # Leave a useful but smaller-than-configured final completion budget.
    assert budget.reserve_or_delay(5700) == 0
    fake = client([completion()])
    provider = GroqProvider(api_key="test", client=fake, token_budget=budget)
    asyncio.run(provider.complete_json("prompt", {}, agent="research_synthesizer", max_output_tokens=1600))
    request = fake.chat.completions.requests[0]
    input_tokens = provider._messages("prompt", {}, settings.final_agent_input_token_limit)[1]
    assert request["max_completion_tokens"] == 1300 - input_tokens


def test_token_429_respects_retry_after_and_max_retry_count():
    limited = ProviderException(429, {"error": {"type": "tokens", "code": "rate_limit_exceeded"}})
    limited.response.headers = {"retry-after": "3"}
    fake = client([limited, completion()])
    delays = []
    provider = GroqProvider(api_key="test", client=fake, sleep=delays.append, random_value=lambda: 0, token_budget=TokenBudgetManager(7000))
    assert asyncio.run(provider.complete_json("prompt", {}, agent="news_analyst", max_output_tokens=700)) == {"ok": True}
    assert delays == [3.25] and fake.chat.completions.calls == 2


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


def test_visible_length_after_compact_retry_falls_back_to_qwen_once():
    fake = client([
        completion({"partial": True}, finish_reason="length", output_tokens=900),
        completion({"partial": True}, finish_reason="length", output_tokens=900),
        completion({"ok": True}),
    ])
    provider = GroqProvider(api_key="test", client=fake, token_budget=TokenBudgetManager(7000))
    assert asyncio.run(provider.complete_json("prompt", {}, agent="market_analyst", max_output_tokens=900)) == {"ok": True}
    assert [request["model"] for request in fake.chat.completions.requests] == [
        "openai/gpt-oss-20b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b",
    ]


def test_synthesizer_length_retry_uses_the_minimal_final_contract():
    fake = client([
        completion({"partial": True}, finish_reason="length", output_tokens=1000),
        completion({"ok": True}),
    ])
    provider = GroqProvider(api_key="test", client=fake, token_budget=TokenBudgetManager(7000))
    assert asyncio.run(provider.complete_json("normal synthesis prompt", {"news_summary": {"summary": "x" * 500}}, agent="research_synthesizer", max_output_tokens=1000)) == {"ok": True}
    assert "one sentence, maximum 30 words" in fake.chat.completions.requests[1]["messages"][0]["content"]
    assert "normal synthesis prompt" not in fake.chat.completions.requests[1]["messages"][0]["content"]


def test_primary_models_route_research_agents_to_gpt_oss_and_synthesizer_to_qwen():
    fake = client([completion(), completion(), completion(), completion()])
    provider = GroqProvider(api_key="test", client=fake, token_budget=TokenBudgetManager(7000))
    for agent in ("news_analyst", "market_analyst", "document_rag_agent", "research_synthesizer"):
        asyncio.run(provider.complete_json("prompt", {}, agent=agent, max_output_tokens=5000))
    assert [request["model"] for request in fake.chat.completions.requests] == [
        "openai/gpt-oss-20b", "openai/gpt-oss-20b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b",
    ]


def test_qwen_fallback_is_attempted_once_and_reuses_tpm_scheduler():
    primary_failure = ProviderException(500, {"error": {"message": "server unavailable"}})
    qwen_failure = ProviderException(500, {"error": {"message": "server unavailable"}})
    fake = client([primary_failure, primary_failure, primary_failure, qwen_failure])
    budget = TokenBudgetManager(7000)
    provider = GroqProvider(api_key="test", client=fake, sleep=lambda _: None, random_value=lambda: 0, token_budget=budget)
    with pytest.raises(LLMProviderError):
        asyncio.run(provider.complete_json("prompt", {}, agent="news_analyst", max_output_tokens=800))
    assert [request["model"] for request in fake.chat.completions.requests].count("qwen/qwen3.6-27b") == 1
    assert fake.chat.completions.requests[-1]["model"] == "qwen/qwen3.6-27b"
    assert budget.available() == 7000
