from uuid import UUID

import pytest

from app.api.v1.research import get_orchestrator
from app.core.database import get_db
from app.schemas.research import Evidence, MarketAnalysis
from app.services.agents import DocumentRagAgent, MarketAnalystAgent, NewsAnalystAgent, ResearchSynthesizer
from app.services.rag_service import RagService
from app.services.research_orchestrator import ResearchOrchestrator
from app.services.research_service import ResearchService
from .conftest import auth_headers, signup
from .fakes import FakeLLMProvider, FakeResearchProvider


def make_orchestrator(llm=None):
    llm = llm or FakeLLMProvider()
    provider = FakeResearchProvider()
    return ResearchOrchestrator(ResearchService(provider, provider), RagService(llm), MarketAnalystAgent(llm), NewsAnalystAgent(llm), DocumentRagAgent(llm), ResearchSynthesizer(llm))


def test_research_request_runs_agents_and_persists_tenant_scoped_result(client):
    account = signup(client)
    client.app.dependency_overrides[get_orchestrator] = make_orchestrator
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200, response.text
    job = response.json()["data"]
    assert job["status"] == "PENDING"
    finished = client.get(f"/api/v1/research/{job['id']}", headers=auth_headers(account["access_token"])).json()["data"]
    assert finished["status"] == "COMPLETED"
    assert finished["report_id"]


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


def test_malformed_llm_output_is_rejected_and_job_fails(client):
    account = signup(client)
    client.app.dependency_overrides[get_orchestrator] = lambda: make_orchestrator(FakeLLMProvider(malformed=True))
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    job_id = response.json()["data"]["id"]
    assert client.get(f"/api/v1/research/{job_id}", headers=auth_headers(account["access_token"])).json()["data"]["status"] == "FAILED"


def test_provider_failure_is_controlled_and_job_fails(client):
    account = signup(client)
    client.app.dependency_overrides[get_orchestrator] = lambda: make_orchestrator(FakeLLMProvider(fail=True))
    response = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze the company's recent performance and major risks."}, headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    job_id = response.json()["data"]["id"]
    assert client.get(f"/api/v1/research/{job_id}", headers=auth_headers(account["access_token"])).json()["data"]["status"] == "FAILED"


def test_analysis_rejects_claim_with_unknown_evidence():
    with pytest.raises(ValueError):
        MarketAnalysis(summary="test", metrics={}, signals=[{"claim": "unsupported", "evidence": [Evidence(source_type="MARKET", source_id="wrong", snippet="wrong").model_dump()]}], evidence=[])


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
