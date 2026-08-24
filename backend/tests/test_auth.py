from .conftest import auth_headers, signup


def test_signup_and_me(client):
    account = signup(client)
    assert account["user"]["role"] == "ADMIN"
    response = client.get("/api/v1/auth/me", headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    assert response.json()["data"]["email"] == "aashi@example.com"


def test_duplicate_email_rejected(client):
    signup(client)
    response = client.post("/api/v1/auth/signup", json={"name": "Other", "email": "aashi@example.com", "password": "strong-password", "organization_name": "Other Research"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EMAIL_ALREADY_EXISTS"


def test_login_and_invalid_password(client):
    signup(client)
    good = client.post("/api/v1/auth/login", json={"email": "aashi@example.com", "password": "strong-password"})
    assert good.status_code == 200
    bad = client.post("/api/v1/auth/login", json={"email": "aashi@example.com", "password": "wrong-password"})
    assert bad.status_code == 401


def test_protected_route_rejects_anonymous(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_token_without_required_expiration_is_rejected(client):
    import jwt
    from app.core.config import settings
    account = signup(client)
    user = account["user"]
    token = jwt.encode({"user_id": user["id"], "organization_id": user["organization"]["id"], "role": "ADMIN"}, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    response = client.get("/api/v1/auth/me", headers=auth_headers(token))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_TOKEN"
