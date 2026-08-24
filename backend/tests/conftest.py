import os
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.api.deps import get_current_user
from app.core.database import Base, get_db
import app.models
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(engine)

    def override_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)


def signup(client, *, name="Aashi", email="aashi@example.com", organization_name="Aashi Research"):
    response = client.post("/api/v1/auth/signup", json={"name": name, "email": email, "password": "strong-password", "organization_name": organization_name})
    assert response.status_code == 201, response.text
    return response.json()["data"]


def auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}
