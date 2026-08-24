from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.core.exceptions import api_error
from app.core.security import create_access_token, hash_password, verify_password
from app.models.organization import Organization
from app.models.user import Role, User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest, SignupRequest


class AuthService:
    users = UserRepository()

    def signup(self, db: Session, payload: SignupRequest) -> User:
        if self.users.get_by_email(db, str(payload.email)):
            raise api_error(409, "EMAIL_ALREADY_EXISTS", "An account with this email already exists.")
        organization = Organization(name=payload.organization_name.strip())
        user = User(name=payload.name.strip(), email=str(payload.email).lower(), password_hash=hash_password(payload.password), role=Role.ADMIN, organization=organization)
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise api_error(409, "EMAIL_ALREADY_EXISTS", "An account with this email already exists.")
        db.refresh(user)
        return user

    def login(self, db: Session, payload: LoginRequest) -> User:
        user = self.users.get_by_email(db, str(payload.email))
        if not user or not verify_password(payload.password, user.password_hash):
            raise api_error(401, "INVALID_CREDENTIALS", "Email or password is incorrect.")
        return user

    def token_for(self, user: User) -> str:
        return create_access_token(user_id=str(user.id), organization_id=str(user.organization_id), role=user.role.value)
