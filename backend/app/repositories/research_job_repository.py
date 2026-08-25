from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from app.models.research_job import ResearchJob


class ResearchJobRepository:
    def add(self, db: Session, job: ResearchJob) -> None:
        db.add(job)

    def get_by_id_in_organization(self, db: Session, job_id: UUID, organization_id: UUID) -> ResearchJob | None:
        return db.scalar(select(ResearchJob).where(ResearchJob.id == job_id, ResearchJob.organization_id == organization_id))

    def list_in_organization(self, db: Session, organization_id: UUID, offset: int, limit: int) -> tuple[list[ResearchJob], int]:
        query = select(ResearchJob).where(ResearchJob.organization_id == organization_id)
        total = len(list(db.scalars(query)))
        return list(db.scalars(query.order_by(ResearchJob.created_at.desc()).offset(offset).limit(limit))), total

    def delete_all_in_organization(self, db: Session, organization_id: UUID) -> int:
        result = db.execute(delete(ResearchJob).where(ResearchJob.organization_id == organization_id))
        return int(result.rowcount or 0)
