from app.api.v1.research import get_orchestrator
from .conftest import auth_headers, signup
from .test_research_engine import make_orchestrator


def test_health_auth_protected_research_report_and_tenant_smoke(client):
    assert client.get("/api/v1/health").status_code == 200
    first = signup(client, email="smoke-one@example.com", organization_name="Smoke One")
    login = client.post("/api/v1/auth/login", json={"email": "smoke-one@example.com", "password": "strong-password"})
    assert login.status_code == 200
    assert client.get("/api/v1/research").status_code == 401
    client.app.dependency_overrides[get_orchestrator] = make_orchestrator
    headers = auth_headers(first["access_token"])
    created = client.post("/api/v1/research", json={"ticker": "NVDA", "question": "Analyze recent performance, major risks, and growth opportunities."}, headers=headers)
    assert created.status_code == 200
    job = client.get(f"/api/v1/research/{created.json()['data']['id']}", headers=headers).json()["data"]
    assert job["status"] == "COMPLETED"
    assert client.get(f"/api/v1/reports/{job['report_id']}", headers=headers).status_code == 200
    second = signup(client, email="smoke-two@example.com", organization_name="Smoke Two")
    assert client.get(f"/api/v1/reports/{job['report_id']}", headers=auth_headers(second["access_token"])).status_code == 404
