from app.core.security import create_access_token
from app.models.user import Role
from uuid import UUID
from .conftest import auth_headers, signup


def test_admin_can_access_stats(client):
    account = signup(client)
    response = client.get("/api/v1/admin/stats", headers=auth_headers(account["access_token"]))
    assert response.status_code == 200
    assert response.json()["data"]["total_users"] == 1


def test_analyst_is_forbidden(client):
    account = signup(client)
    user = account["user"]
    org_id = user["organization"]["id"]
    # Alter the persisted role through the app's test database route dependency.
    from app.core.database import get_db
    db = next(client.app.dependency_overrides[get_db]())
    from app.models.user import User
    db_user = db.get(User, UUID(user["id"]))
    db_user.role = Role.ANALYST
    db.commit()
    token = create_access_token(user_id=user["id"], organization_id=org_id, role="ADMIN")
    response = client.get("/api/v1/admin/stats", headers=auth_headers(token))
    assert response.status_code == 403
