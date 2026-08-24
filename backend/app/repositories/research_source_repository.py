from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.research_source import ResearchSource


class ResearchSourceRepository:
    def list_for_company(self, db: Session, company_id: UUID) -> list[ResearchSource]:
        return list(db.scalars(select(ResearchSource).where(ResearchSource.company_id == company_id).order_by(ResearchSource.published_at.desc().nullslast())))

    def existing_urls(self, db: Session, company_id: UUID, urls: list[str]) -> set[str]:
        if not urls:
            return set()
        return set(db.scalars(select(ResearchSource.source_url).where(ResearchSource.company_id == company_id, ResearchSource.source_url.in_(urls))))
