from sqlalchemy.orm import Session
from app.core.exceptions import api_error
from app.models.user import User
from app.repositories.organization_repository import OrganizationRepository
from app.repositories.user_repository import UserRepository


class OrganizationService:
    organizations = OrganizationRepository()
    users = UserRepository()

    def get_current_organization(self, db: Session, current_user: User):
        organization = self.organizations.get_by_id_for_tenant(db, current_user.organization_id, current_user.organization_id)
        if not organization:
            raise api_error(404, "ORGANIZATION_NOT_FOUND", "Organization was not found.")
        return organization

    def update_current_organization(self, db: Session, current_user: User, name: str):
        organization = self.get_current_organization(db, current_user)
        organization.name = name.strip()
        db.commit()
        db.refresh(organization)
        return organization
