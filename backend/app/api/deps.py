from collections.abc import Callable
from uuid import UUID
import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import api_error
from app.models.user import Role, User
from app.repositories.user_repository import UserRepository

bearer = HTTPBearer(auto_error=False)
users = UserRepository()


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if not credentials:
        raise api_error(401, "UNAUTHORIZED", "Authentication is required.")
    try:
        claims = jwt.decode(credentials.credentials, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm], options={"require": ["exp", "user_id", "organization_id", "role"]})
        user_id, organization_id = UUID(claims["user_id"]), UUID(claims["organization_id"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise api_error(401, "INVALID_TOKEN", "The access token is invalid or expired.")
    user = users.get_by_id_in_organization(db, user_id, organization_id)
    if not user:
        raise api_error(401, "UNAUTHORIZED", "The authenticated user no longer exists.")
    # Role is taken from the database, never trusted from client-provided state or stale token claims.
    return user


def require_role(role: Role) -> Callable:
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise api_error(403, "FORBIDDEN", "You do not have permission to perform this action.")
        return current_user
    return dependency
