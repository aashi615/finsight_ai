from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models.organization import Organization


class OrganizationRepository:
    def get_by_name(self, db: Session, name: str) -> Organization | None:
        """Find the workspace selected during signup, ignoring casing/spacing."""
        normalized = name.strip().casefold()
        return db.scalar(
            select(Organization)
            .where(func.lower(Organization.name) == normalized)
            .order_by(Organization.created_at, Organization.id)
        )

    def get_by_id_for_tenant(self, db: Session, organization_id: UUID, tenant_id: UUID) -> Organization | None:
        # Explicit tenant predicate prevents a caller from reading a different organization.
        return db.scalar(select(Organization).where(Organization.id == organization_id, Organization.id == tenant_id))
