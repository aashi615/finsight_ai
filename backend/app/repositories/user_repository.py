from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.user import User


class UserRepository:
    def get_by_email(self, db: Session, email: str) -> User | None:
        return db.scalar(select(User).where(User.email == email.lower()))

    def get_by_id_in_organization(self, db: Session, user_id: UUID, organization_id: UUID) -> User | None:
        return db.scalar(select(User).where(User.id == user_id, User.organization_id == organization_id))

    def list_by_organization(self, db: Session, organization_id: UUID) -> list[User]:
        return list(db.scalars(select(User).where(User.organization_id == organization_id).order_by(User.created_at)))

    def count_by_organization(self, db: Session, organization_id: UUID) -> int:
        return len(self.list_by_organization(db, organization_id))
