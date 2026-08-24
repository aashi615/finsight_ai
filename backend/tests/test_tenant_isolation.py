from .conftest import auth_headers, signup


def test_tenants_only_see_their_own_organization_and_members(client):
    first = signup(client, name="User A", email="a@example.com", organization_name="Organization A")
    second = signup(client, name="User B", email="b@example.com", organization_name="Organization B")
    org_a = first["user"]["organization"]
    org_b = second["user"]["organization"]

    response_a = client.get("/api/v1/organization", headers=auth_headers(first["access_token"]))
    response_b = client.get("/api/v1/organization", headers=auth_headers(second["access_token"]))
    assert response_a.json()["data"]["id"] == org_a["id"]
    assert response_b.json()["data"]["id"] == org_b["id"]

    # No organization ID is accepted by these APIs; tenant context comes exclusively from the JWT-resolved user.
    members_a = client.get("/api/v1/organization/members", headers=auth_headers(first["access_token"]))
    assert members_a.status_code == 200
    assert [member["email"] for member in members_a.json()["data"]] == ["a@example.com"]
    assert "b@example.com" not in str(members_a.json())
