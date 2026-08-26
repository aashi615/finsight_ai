import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.api.v1.research import get_orchestrator
from app.schemas.research import DocumentAnalysis, Evidence, MarketAnalysis, NewsAnalysis, ResearchPlan, ResearchSynthesis
from app.services.agents import DocumentRagAgent, MarketAnalystAgent, NewsAnalystAgent, ResearchSynthesizer
from app.services.query_planner import QueryPlanner
from app.services.rag_service import RagService
from app.services.research_orchestrator import ResearchOrchestrator
from app.services.research_service import ResearchService
from .conftest import auth_headers, signup
from .fakes import FakeLLMProvider, FakeResearchProvider


class PlannerLLM:
    def __init__(self, response):
        self.response = response

    async def complete_json(self, *args, **kwargs):
        return self.response


@pytest.mark.parametrize(
    ("question", "response", "companies", "market", "news", "documents"),
    [
        ("What are Apple's latest AI developments?", {"companies": ["AAPL"], "needs_market": False, "needs_news": True, "needs_documents": False}, ["AAPL"], False, True, False),
        ("How has Apple's stock performed?", {"companies": ["AAPL"], "needs_market": True, "needs_news": False, "needs_documents": False}, ["AAPL"], True, False, False),
        ("What risks does Microsoft mention in its filings?", {"companies": ["MSFT"], "needs_market": False, "needs_news": False, "needs_documents": True}, ["MSFT"], False, False, True),
        ("Compare Apple and Microsoft based on stock performance and recent news.", {"companies": ["AAPL", "MSFT"], "needs_market": True, "needs_news": True, "needs_documents": False}, ["AAPL", "MSFT"], True, True, False),
    ],
)
def test_query_planner_validates_semantic_llm_plans(question, response, companies, market, news, documents):
    plan = asyncio.run(QueryPlanner(PlannerLLM(response)).analyze(question, "NVDA"))
    assert plan.companies == companies
    assert plan.needs_market is market
    assert plan.needs_news is news
    assert plan.needs_documents is documents


def test_query_planner_uses_legacy_all_sources_fallback_for_malformed_output():
    plan = asyncio.run(QueryPlanner(PlannerLLM({"companies": []})).analyze("Analyze NVIDIA.", "NVDA"))
    assert plan == ResearchPlan(companies=["NVDA"], needs_market=True, needs_news=True, needs_documents=True, reasoning="Planner unavailable; using the complete research flow.")


@pytest.mark.parametrize("payload", [
    {"companies": ["aapl"], "needs_market": True, "needs_news": False, "needs_documents": False},
    {"companies": ["AAPL"], "needs_market": "true", "needs_news": False, "needs_documents": False},
    {"companies": ["AAPL"], "needs_market": False, "needs_news": False, "needs_documents": False},
])
def test_research_plan_rejects_non_contract_output(payload):
    with pytest.raises(ValueError):
        ResearchPlan.model_validate(payload)


class StaticPlanner:
    def __init__(self, plan):
        self.plan = plan

    async def analyze(self, *args):
        return self.plan

    @staticmethod
    def selected_agents(plan):
        return QueryPlanner.selected_agents(plan)

    @staticmethod
    def skipped_agents(plan):
        return QueryPlanner.skipped_agents(plan)


class TrackingResearchService:
    def __init__(self):
        self.market_calls = 0
        self.news_calls = 0

    async def get_market_data_async(self, db, ticker, from_date, to_date):
        self.market_calls += 1
        return None, []

    def get_news(self, db, ticker, from_date, to_date, limit):
        self.news_calls += 1
        return None, []


class TrackingRag:
    def __init__(self):
        self.calls = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        return []


class TrackingAgent:
    def __init__(self, result):
        self.calls = 0
        self.result = result

    async def analyze(self, *args):
        self.calls += 1
        return self.result


class StubSynthesizer:
    async def synthesize(self, *args):
        evidence = Evidence(source_type="TEST", source_id="test", snippet="test evidence")
        return ResearchSynthesis(executive_summary="Summary", company_overview="Overview", market_analysis="Market", news_analysis="News", growth_catalysts=[], key_risks=[], key_opportunities=[], competitive_landscape="Competition", valuation="Unavailable", conclusion="Conclusion", evidence=[evidence], confidence=0.5, generated_at=datetime.now(timezone.utc))


def run_conditional_plan(plan=None, planner=None):
    service, rag = TrackingResearchService(), TrackingRag()
    market = TrackingAgent(MarketAnalysis(summary="Market", metrics={}, signals=[], evidence=[]))
    news = TrackingAgent(NewsAnalysis(summary="News", themes=[], signals=[], evidence=[]))
    documents = TrackingAgent(DocumentAnalysis(summary="Documents", findings=[], evidence=[]))
    orchestrator = ResearchOrchestrator(service, rag, market, news, documents, StubSynthesizer(), planner or StaticPlanner(plan))
    db = SimpleNamespace(commit=lambda: None)
    job = SimpleNamespace(id="job-1", organization_id="organization-1", question="Test query", result=None)
    company = SimpleNamespace(id="company-1", ticker="NVDA")
    asyncio.run(orchestrator._run_agents(db, job, company))
    return service, rag, market, news, documents


def test_news_only_plan_executes_only_news_branch_and_source():
    service, rag, market, news, documents = run_conditional_plan(ResearchPlan(companies=["NVDA"], needs_market=False, needs_news=True, needs_documents=False))
    assert (service.market_calls, service.news_calls, rag.calls) == (0, 1, 0)
    assert (market.calls, news.calls, documents.calls) == (0, 1, 0)


def test_market_only_plan_executes_only_market_branch_and_source():
    service, rag, market, news, documents = run_conditional_plan(ResearchPlan(companies=["NVDA"], needs_market=True, needs_news=False, needs_documents=False))
    assert (service.market_calls, service.news_calls, rag.calls) == (1, 0, 0)
    assert (market.calls, news.calls, documents.calls) == (1, 0, 0)


def test_documents_only_plan_executes_only_document_branch_and_source():
    service, rag, market, news, documents = run_conditional_plan(ResearchPlan(companies=["NVDA"], needs_market=False, needs_news=False, needs_documents=True))
    assert (service.market_calls, service.news_calls, rag.calls) == (0, 0, 1)
    assert (market.calls, news.calls, documents.calls) == (0, 0, 1)


def test_all_source_plan_executes_every_branch():
    service, rag, market, news, documents = run_conditional_plan(ResearchPlan(companies=["NVDA"], needs_market=True, needs_news=True, needs_documents=True))
    assert (service.market_calls, service.news_calls, rag.calls) == (1, 1, 1)
    assert (market.calls, news.calls, documents.calls) == (1, 1, 1)


def test_malformed_planner_output_uses_all_source_fallback_without_crashing():
    service, rag, market, news, documents = run_conditional_plan(planner=QueryPlanner(PlannerLLM({"companies": []})))
    assert (service.market_calls, service.news_calls, rag.calls) == (1, 1, 1)
    assert (market.calls, news.calls, documents.calls) == (1, 1, 1)


def test_existing_research_result_shape_remains_compatible(client):
    llm, provider = FakeLLMProvider(), FakeResearchProvider()
    client.app.dependency_overrides[get_orchestrator] = lambda: ResearchOrchestrator(ResearchService(provider, provider), RagService(llm), MarketAnalystAgent(llm), NewsAnalystAgent(llm), DocumentRagAgent(llm), ResearchSynthesizer(llm))
    account = signup(client, email="planner-compat@example.com", organization_name="Planner Compatibility")
    created = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze NVIDIA's recent performance and risks."}, headers=auth_headers(account["access_token"]))
    assert created.status_code == 200
    result = client.get(f"/api/v1/research/{created.json()['data']['id']}", headers=auth_headers(account["access_token"])).json()["data"]["result"]
    assert {"news_summary", "market_summary", "rag_summary", "final_result"}.issubset(result)
