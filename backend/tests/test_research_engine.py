import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.api.v1.research import get_orchestrator
from app.core.database import get_db
from app.providers.base import ProviderError
from app.schemas.research import DocumentAnalysis, Evidence, MarketAnalysis, NewsAnalysis, ResearchSynthesis
from app.services.agents import AgentFailure, DocumentRagAgent, MarketAnalystAgent, NewsAnalystAgent, ResearchSynthesizer, _normalize_synthesis_response, _validate
from app.services.rag_service import RagService
from app.services.research_orchestrator import ResearchOrchestrator
from app.services.research_service import ResearchService
from .conftest import auth_headers, signup
from .fakes import FakeLLMProvider, FakeResearchProvider


def make_orchestrator(llm=None):
    llm = llm or FakeLLMProvider()
    provider = FakeResearchProvider()
    return ResearchOrchestrator(ResearchService(provider, provider), RagService(llm), MarketAnalystAgent(llm), NewsAnalystAgent(llm), DocumentRagAgent(llm), ResearchSynthesizer(llm))


def test_complete_research_orchestration_succeeds_with_valid_market_analyst_json(client):
    account = signup(client)
    client.app.dependency_overrides[get_orchestrator] = make_orchestrator
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200, response.text
    job = response.json()["data"]
    assert job["status"] == "PENDING"
    finished = client.get(f"/api/v1/research/{job['id']}", headers=auth_headers(account["access_token"])).json()["data"]
    assert finished["status"] == "COMPLETED"
    assert finished["report_id"]


def test_repeated_job_polling_does_not_reexecute_agents_or_llm(client):
    account = signup(client)
    llm = FakeLLMProvider()
    client.app.dependency_overrides[get_orchestrator] = lambda: make_orchestrator(llm)
    headers = auth_headers(account["access_token"])
    created = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=headers).json()["data"]
    calls_after_job = len(llm.calls)
    for _ in range(3):
        response = client.get(f"/api/v1/research/{created['id']}", headers=headers)
        assert response.status_code == 200
    assert len(llm.calls) == calls_after_job


def test_research_jobs_are_not_visible_across_tenants(client):
    first = signup(client, email="one@example.com", organization_name="One")
    second = signup(client, email="two@example.com", organization_name="Two")
    client.app.dependency_overrides[get_orchestrator] = make_orchestrator
    created = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(first["access_token"])).json()["data"]
    response = client.get(f"/api/v1/research/{created['id']}", headers=auth_headers(second["access_token"]))
    assert response.status_code == 404


def test_rag_retrieval_is_scoped_to_organization(client):
    first = signup(client, email="rag-one@example.com", organization_name="Rag One")
    second = signup(client, email="rag-two@example.com", organization_name="Rag Two")
    db = next(client.app.dependency_overrides[get_db]())
    llm = FakeLLMProvider()
    rag = RagService(llm)
    first_org = UUID(first["user"]["organization"]["id"])
    second_org = UUID(second["user"]["organization"]["id"])
    rag.ingest_text(db, first_org, "Private note", "First organization confidential forecast.")
    rag.ingest_text(db, second_org, "Private note", "Second organization confidential forecast.")
    result = rag.retrieve(db, first_org, "confidential forecast", None)
    assert len(result) == 1
    assert result[0].content.startswith("First organization")


def test_malformed_llm_output_completes_with_verified_source_fallback(client):
    account = signup(client)
    client.app.dependency_overrides[get_orchestrator] = lambda: make_orchestrator(FakeLLMProvider(malformed=True))
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    job_id = response.json()["data"]["id"]
    completed = client.get(f"/api/v1/research/{job_id}", headers=auth_headers(account["access_token"])).json()["data"]
    assert completed["status"] == "COMPLETED"
    assert completed["result"]["final_result"]["confidence"] == 0.2


def test_provider_failure_completes_with_verified_source_fallback(client):
    account = signup(client)
    client.app.dependency_overrides[get_orchestrator] = lambda: make_orchestrator(FakeLLMProvider(fail=True))
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    job_id = response.json()["data"]["id"]
    assert client.get(f"/api/v1/research/{job_id}", headers=auth_headers(account["access_token"])).json()["data"]["status"] == "COMPLETED"


def test_market_provider_failure_preserves_other_evidence_and_completes_partial_research(client):
    class MarketUnavailableProvider(FakeResearchProvider):
        def get_market_data(self, ticker, from_date, to_date):
            raise ProviderError("historical providers unavailable", status_code=503)

    provider = MarketUnavailableProvider()
    llm = FakeLLMProvider()
    client.app.dependency_overrides[get_orchestrator] = lambda: ResearchOrchestrator(ResearchService(provider, provider), RagService(llm), MarketAnalystAgent(llm), NewsAnalystAgent(llm), DocumentRagAgent(llm), ResearchSynthesizer(llm))
    account = signup(client)
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    job_id = response.json()["data"]["id"]
    completed = client.get(f"/api/v1/research/{job_id}", headers=auth_headers(account["access_token"])).json()["data"]
    assert completed["status"] == "COMPLETED"
    assert completed["result"]["market_summary"]["evidence"] == []
    assert completed["result"]["news_summary"]["evidence"]


def test_news_provider_failure_preserves_market_evidence_and_completes_partial_research(client):
    class NewsUnavailableProvider(FakeResearchProvider):
        def get_news(self, ticker, from_date, to_date):
            raise ProviderError("down")

    provider = NewsUnavailableProvider()
    client.app.dependency_overrides[get_orchestrator] = lambda: ResearchOrchestrator(ResearchService(provider, provider), RagService(FakeLLMProvider()), MarketAnalystAgent(FakeLLMProvider()), NewsAnalystAgent(FakeLLMProvider()), DocumentRagAgent(FakeLLMProvider()), ResearchSynthesizer(FakeLLMProvider()))
    account = signup(client)
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    job_id = response.json()["data"]["id"]
    completed = client.get(f"/api/v1/research/{job_id}", headers=auth_headers(account["access_token"])).json()["data"]
    assert completed["status"] == "COMPLETED"
    assert completed["result"]["news_summary"]["evidence"] == []
    assert completed["result"]["market_summary"]["evidence"]


def test_analysis_rejects_claim_with_unknown_evidence():
    with pytest.raises(ValueError):
        MarketAnalysis(summary="test", metrics={}, signals=[{"claim": "unsupported", "evidence": [Evidence(source_type="MARKET", source_id="wrong", snippet="wrong").model_dump()]}], evidence=[])


def test_market_claim_matching_supplied_evidence_is_added_to_top_level_evidence():
    evidence = Evidence(source_type="MARKET", source_id="market-1", snippet="Close 100")

    class ClaimOnlyLLM:
        async def complete_json(self, *args, **kwargs):
            return {"agent": "market_analyst", "summary": "Valid.", "metrics": {}, "signals": [{"claim": "Observed close.", "evidence": [evidence.model_dump()]}], "evidence": []}

    result = asyncio.run(MarketAnalystAgent(ClaimOnlyLLM())._complete(MarketAnalysis, "prompt", {}, require_evidence=True, supplied_evidence=[evidence]))
    assert result.signals[0].evidence == [evidence]
    assert result.evidence == [evidence]


def test_deterministic_market_fallback_never_accepts_an_invented_claim():
    supplied = Evidence(source_type="MARKET", source_id="market-1", snippet="Close 100")
    invented = Evidence(source_type="MARKET", source_id="invented", snippet="Altered")

    class InventedEvidenceLLM:
        async def complete_json(self, *args, **kwargs):
            return {"agent": "market_analyst", "summary": "Invalid.", "metrics": {}, "signals": [{"claim": "Unsupported.", "evidence": [invented.model_dump()]}], "evidence": [invented.model_dump()]}

    result = asyncio.run(MarketAnalystAgent(InventedEvidenceLLM())._complete(MarketAnalysis, "prompt", {}, require_evidence=True, supplied_evidence=[supplied]))
    assert result.signals == []
    assert result.evidence == [supplied]


def test_synthesis_rejects_risk_that_is_not_cited_in_top_level_evidence():
    with pytest.raises(ValueError, match="Synthesis claims must cite evidence"):
        from app.schemas.research import ResearchSynthesis
        ResearchSynthesis(
            executive_summary="summary", company_overview="overview", market_analysis="market", news_analysis="news",
            key_risks=[{"claim": "unsupported", "evidence": [{"source_type": "NEWS", "source_id": "wrong", "snippet": "wrong"}]}],
            key_opportunities=[], evidence=[{"source_type": "NEWS", "source_id": "right", "snippet": "right"}],
            confidence=0.5, generated_at=datetime.now(timezone.utc),
        )


@pytest.mark.parametrize(
    ("risks", "opportunities", "expected_risks", "expected_opportunities"),
    [
        ([{"claim": "Risk", "evidence": [{"source_type": "NEWS", "source_id": "news_001", "snippet": "Evidence"}]}], [], 1, 0),
        ("Risk as string", [], 1, 0),
        ([], "Opportunity as string", 0, 1),
        ("Risk as string", "Opportunity as string", 1, 1),
    ],
)
def test_synthesizer_normalizes_only_string_array_fields_to_evidence_backed_arrays(risks, opportunities, expected_risks, expected_opportunities):
    payload = {
        "executive_summary": "Summary", "company_overview": "Overview", "market_analysis": "Market", "news_analysis": "News",
        "key_risks": risks, "key_opportunities": opportunities,
        "evidence": [{"source_type": "NEWS", "source_id": "news_001", "snippet": "Evidence"}],
        "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = _validate(ResearchSynthesis, _normalize_synthesis_response(payload), agent="research_synthesizer")
    assert len(result.key_risks) == expected_risks
    assert len(result.key_opportunities) == expected_opportunities


def test_synthesizer_normalization_does_not_coerce_unrelated_malformed_fields():
    payload = {
        "executive_summary": ["not text"], "company_overview": "Overview", "market_analysis": "Market", "news_analysis": "News",
        "key_risks": "Risk", "key_opportunities": [],
        "evidence": [{"source_type": "NEWS", "source_id": "news_001", "snippet": "Evidence"}],
        "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with pytest.raises(AgentFailure, match="invalid structured analysis"):
        _validate(ResearchSynthesis, _normalize_synthesis_response(payload), agent="research_synthesizer")


@pytest.mark.parametrize("evidence", [
    [{"source_type": "NEWS", "snippet": "Required snippet"}],
    [{"source_type": "NEWS", "source_id": "news_001"}],
])
def test_synthesis_schema_requires_source_id_and_snippet(evidence):
    from app.schemas.research import ResearchSynthesis
    with pytest.raises(ValueError):
        ResearchSynthesis(
            executive_summary="summary", company_overview="overview", market_analysis="market", news_analysis="news",
            key_risks=[], key_opportunities=[], evidence=evidence, confidence=0.5, generated_at=datetime.now(timezone.utc),
        )


def test_market_agent_valid_json_parses_to_market_analysis():
    class ValidLLM:
        async def complete_json(self, *args, **kwargs):
            return {"agent": "market_analyst", "summary": "valid", "metrics": {"start_close": 100.0}, "signals": [], "evidence": []}

    result = asyncio.run(MarketAnalystAgent(ValidLLM())._complete(MarketAnalysis, "prompt", {}))
    assert result == MarketAnalysis(summary="valid", metrics={"start_close": 100.0}, signals=[], evidence=[])


def test_news_agent_valid_json_parses_to_news_analysis():
    class ValidLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            evidence = payload["evidence"]
            return {"agent": "news_analyst", "summary": "valid", "themes": ["earnings"], "signals": [], "evidence": evidence}

    article = SimpleNamespace(id="news-1", title="NVIDIA update", summary="Summary", url="https://example.test/news", published_at=datetime.now(timezone.utc))
    result = asyncio.run(NewsAnalystAgent(ValidLLM()).analyze({"question": "What changed?", "news": [article]}))
    assert result == NewsAnalysis(summary="valid", themes=["earnings"], signals=[], evidence=[{"source_type": "NEWS", "source_id": "news-1", "snippet": "NVIDIA update", "url": "https://example.test/news"}])


def test_news_agent_retries_with_fallback_when_primary_returns_empty_evidence():
    class CompactLLM:
        calls = []
        async def complete_json(self, prompt, payload, **kwargs):
            self.calls.append(kwargs.get("force_fallback", False))
            evidence = payload["evidence"]
            if not kwargs.get("force_fallback"):
                return {"agent": "news_analyst", "summary": "No material change.", "themes": [], "signals": [], "evidence": []}
            return {"agent": "news_analyst", "summary": "No material change.", "themes": [], "signals": [], "evidence": evidence[:1]}

    article = SimpleNamespace(id="news-1", title="NVIDIA update", summary="Summary", url="https://example.test/news", published_at=datetime.now(timezone.utc))
    llm = CompactLLM()
    result = asyncio.run(NewsAnalystAgent(llm).analyze({"question": "What changed?", "news": [article]}))
    assert [item.source_id for item in result.evidence] == ["news-1"]
    assert llm.calls == [False, True]


def test_market_agent_schema_invalid_json_uses_deterministic_fallback():
    class SchemaInvalidLLM:
        async def complete_json(self, *args, **kwargs):
            return {"agent": "market_analyst", "summary": "valid", "metrics": "not an object", "signals": [], "evidence": []}

    result = asyncio.run(MarketAnalystAgent(SchemaInvalidLLM())._complete(MarketAnalysis, "prompt", {}))
    assert result.metrics == {}
    assert result.signals == []


def test_market_agent_uses_deterministic_fallback_when_provider_exhausts_tokens():
    class IncompleteLLM:
        async def complete_json(self, *args, **kwargs):
            from app.llm.base import LLMProviderError
            raise LLMProviderError("Groq response was incomplete because it reached the output token limit.", category="llm_incomplete_response")

    result = asyncio.run(MarketAnalystAgent(IncompleteLLM())._complete(MarketAnalysis, "prompt", {}))
    assert result.metrics == {}
    assert result.evidence == []


def test_final_synthesis_receives_only_independent_compact_summaries():
    class CapturingLLM:
        payload = None
        async def complete_json(self, prompt, payload, **kwargs):
            self.payload = payload
            manifest_id = payload["evidence_manifest"][0]["evidence_id"]
            return {"executive_summary": "Based on available data.", "company_overview": "Available data.", "market_analysis": "Available data.", "news_analysis": "Available data.", "key_risks": [], "key_opportunities": [], "evidence_ids": [manifest_id], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    evidence = Evidence(source_type="NEWS", source_id="n1", snippet="compact", url="https://example.test")
    llm = CapturingLLM()
    result = asyncio.run(ResearchSynthesizer(llm).synthesize(None, NewsAnalysis(summary="news", themes=[], signals=[], evidence=[evidence]), DocumentAnalysis(summary="rag", findings=[], evidence=[])))
    assert result.confidence == 0.5
    assert set(llm.payload) == {"news_summary", "market_summary", "rag_summary", "evidence_manifest"}
    assert llm.payload["market_summary"] is None
    assert "articles" not in str(llm.payload) and "chunks" not in str(llm.payload)


@pytest.mark.parametrize("source_type, prefix", [("NEWS", "news"), ("DOCUMENT", "rag")])
def test_synthesis_accepts_manifest_backed_evidence_and_restores_canonical_value(source_type, prefix):
    class ManifestLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            entry = next(item for item in payload["evidence_manifest"] if item["evidence_id"].startswith(prefix))
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [{"claim": "Risk.", "evidence_ids": [entry["evidence_id"]]}], "key_opportunities": [], "evidence_ids": [entry["evidence_id"]], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    evidence = Evidence(source_type=source_type, source_id="source-1", snippet="Canonical title", url="https://example.test/source")
    news = NewsAnalysis(summary="news", themes=[], signals=[], evidence=[evidence]) if source_type == "NEWS" else None
    rag = DocumentAnalysis(summary="rag", findings=[], evidence=[evidence]) if source_type == "DOCUMENT" else None
    result = asyncio.run(ResearchSynthesizer(ManifestLLM()).synthesize(None, news, rag))
    assert result.evidence[0].model_dump() == evidence.model_dump()
    assert result.key_risks[0].evidence[0].model_dump() == evidence.model_dump()


@pytest.mark.parametrize("bad_reference", ["news_999", "Canonical title", "https://example.test/source", "news_001_changed"])
def test_synthesis_discards_fabricated_manifest_reference_and_uses_safe_fallback(bad_reference):
    class InvalidEvidenceLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            copied = {"evidence_id": bad_reference, "source_type": "NEWS", "source_id": "source-1", "snippet": "Canonical title", "url": "https://example.test/source"}
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [], "key_opportunities": [], "evidence": [copied], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    news = NewsAnalysis(summary="news", themes=[], signals=[], evidence=[Evidence(source_type="NEWS", source_id="source-1", snippet="Canonical title", url="https://example.test/source")])
    result = asyncio.run(ResearchSynthesizer(InvalidEvidenceLLM()).synthesize(None, news, None))
    assert result.confidence == 0.2
    assert [item.source_id for item in result.evidence] == ["source-1"]


def test_synthesis_discards_duplicate_or_unavailable_evidence_reference():
    class InvalidEvidenceLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            original = next(summary["evidence"][0] for key, summary in payload.items() if key.endswith("_summary") and summary)
            copied = {"evidence_id": "news_001", **original}
            unavailable = {"evidence_id": "rag_001", **original}
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [{"claim": "Risk.", "evidence": [unavailable]}], "key_opportunities": [], "evidence": [copied, copied], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    news = NewsAnalysis(summary="news", themes=[], signals=[], evidence=[Evidence(source_type="NEWS", source_id="source-1", snippet="Canonical title")])
    result = asyncio.run(ResearchSynthesizer(InvalidEvidenceLLM()).synthesize(None, news, None))
    assert result.confidence == 0.2
    assert [item.source_id for item in result.evidence] == ["source-1"]


def test_synthesis_uses_valid_claim_evidence_when_legacy_branch_top_level_evidence_is_empty():
    class CapturingLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            assert payload["evidence_manifest"][0]["evidence_id"] == "news_001"
            source = payload["news_summary"]["evidence"][0]
            cited = {"evidence_id": "news_001", **source}
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market unavailable.", "news_analysis": "News available.", "key_risks": [], "key_opportunities": [], "evidence": [cited], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    evidence = Evidence(source_type="NEWS", source_id="news-1", snippet="News evidence", url="https://example.test/news")
    validated = NewsAnalysis(summary="news", themes=[], signals=[{"claim": "Signal", "evidence": [evidence.model_dump()]}], evidence=[evidence])
    # Simulates historical/persisted state whose claim evidence survived but
    # top-level evidence was previously serialized as an empty list.
    legacy = validated.model_copy(update={"evidence": []})
    result = asyncio.run(ResearchSynthesizer(CapturingLLM()).synthesize(None, legacy, None))
    assert result.evidence[0].source_id == "news-1"


def test_synthesis_accepts_manifest_source_id_and_restores_persisted_evidence():
    class ManifestLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            entry = payload["evidence_manifest"][0]
            cited = {"source_type": entry["source_type"], "source_id": entry["evidence_id"], "snippet": entry["title"], "url": entry["source"]}
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [], "key_opportunities": [], "evidence": [cited], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    original = Evidence(source_type="NEWS", source_id="persisted-news-uuid", snippet="Canonical news", url="https://example.test/news")
    result = asyncio.run(ResearchSynthesizer(ManifestLLM()).synthesize(None, NewsAnalysis(summary="news", themes=[], signals=[], evidence=[original]), None))
    assert result.evidence[0].source_id == "persisted-news-uuid"


def test_synthesis_resolves_id_only_citations_to_canonical_metadata():
    class IdOnlyLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            entry = payload["evidence_manifest"][0]
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [{"claim": "Risk.", "evidence_ids": [entry["evidence_id"]]}], "key_opportunities": [], "evidence_ids": [entry["evidence_id"]], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    source = Evidence(source_type="NEWS", source_id="persisted-news", snippet="Canonical title", url="https://example.test/news")
    result = asyncio.run(ResearchSynthesizer(IdOnlyLLM()).synthesize(None, NewsAnalysis(summary="news", themes=[], signals=[], evidence=[source]), None))
    assert [item.model_dump() for item in result.evidence] == [source.model_dump()]
    assert [item.model_dump() for item in result.key_risks[0].evidence] == [source.model_dump()]


def test_synthesis_uses_800_450_350_stages_and_keeps_only_manifest_evidence():
    class StagedLLM:
        def __init__(self):
            self.budgets = []

        async def complete_json(self, prompt, payload, **kwargs):
            self.budgets.append(kwargs["max_output_tokens"])
            evidence_id = payload["evidence_manifest"][0]["evidence_id"]
            if len(self.budgets) < 3:
                return {"executive_summary": "Bad", "company_overview": "Bad", "market_analysis": "Bad", "news_analysis": "Bad", "key_risks": [], "key_opportunities": [], "evidence_ids": ["invented_001"], "confidence": 0.1, "generated_at": datetime.now(timezone.utc).isoformat()}
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [], "key_opportunities": [], "evidence_ids": [evidence_id], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    source = Evidence(source_type="NEWS", source_id="source-1", snippet="Canonical")
    llm = StagedLLM()
    result = asyncio.run(ResearchSynthesizer(llm).synthesize(None, NewsAnalysis(summary="news", themes=[], signals=[], evidence=[source]), None))
    assert llm.budgets == [1200, 600, 400]
    assert [item.source_id for item in result.evidence] == ["source-1"]


def test_synthesis_discards_invalid_manifest_source_id():
    class InvalidManifestLLM:
        async def complete_json(self, prompt, payload, **kwargs):
            return {"executive_summary": "Summary.", "company_overview": "Overview.", "market_analysis": "Market.", "news_analysis": "News.", "key_risks": [], "key_opportunities": [], "evidence_ids": ["news_999"], "confidence": 0.5, "generated_at": datetime.now(timezone.utc).isoformat()}

    source = Evidence(source_type="NEWS", source_id="persisted-news", snippet="Canonical")
    result = asyncio.run(ResearchSynthesizer(InvalidManifestLLM()).synthesize(None, NewsAnalysis(summary="news", themes=[], signals=[], evidence=[source]), None))
    assert result.confidence == 0.2
    assert [item.source_id for item in result.evidence] == ["persisted-news"]


def test_partial_agent_failure_preserves_successful_summary():
    orchestrator = make_orchestrator()
    async def successful():
        return "news summary"
    async def failed():
        raise AgentFailure("market unavailable")
    job = SimpleNamespace(id="job")
    assert asyncio.run(orchestrator._run_branch(job, "NVDA", "news_analyst", successful())) == "news summary"
    assert asyncio.run(orchestrator._run_branch(job, "NVDA", "market_analyst", failed())) is None


def test_research_and_reports_are_paginated_and_tenant_scoped(client):
    first = signup(client, email="reports-one@example.com", organization_name="Reports One")
    second = signup(client, email="reports-two@example.com", organization_name="Reports Two")
    client.app.dependency_overrides[get_orchestrator] = make_orchestrator
    headers = auth_headers(first["access_token"])
    for question in ["Analyze NVIDIA's recent performance and major risks.", "Analyze NVIDIA's current opportunity and risks."]:
        assert client.post("/api/v1/research", json={"ticker": "NVDA", "question": question}, headers=headers).status_code == 200
    jobs = client.get("/api/v1/research?page=1&page_size=1", headers=headers).json()["data"]
    assert len(jobs["items"]) == 1 and jobs["total"] == 2
    reports = client.get("/api/v1/reports?page=1&page_size=1", headers=headers).json()["data"]
    assert len(reports["items"]) == 1 and reports["total"] == 2
    assert client.get(f"/api/v1/reports/{reports['items'][0]['id']}", headers=auth_headers(second["access_token"])).status_code == 404


def test_admin_can_delete_only_its_organization_research_history(client):
    first = signup(client, email="clear-one@example.com", organization_name="Clear One")
    second = signup(client, email="clear-two@example.com", organization_name="Clear Two")
    client.app.dependency_overrides[get_orchestrator] = make_orchestrator
    for account in (first, second):
        client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze recent performance."}, headers=auth_headers(account["access_token"]))
    deleted = client.delete("/api/v1/research", headers=auth_headers(first["access_token"]))
    assert deleted.status_code == 200
    assert deleted.json()["data"]["deleted_jobs"] == 1
    assert client.get("/api/v1/research", headers=auth_headers(first["access_token"])).json()["data"]["total"] == 0
    assert client.get("/api/v1/research", headers=auth_headers(second["access_token"])).json()["data"]["total"] == 1


def test_delete_research_history_cors_preflight_allows_delete(client):
    response = client.options("/api/v1/research", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE"})
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]
