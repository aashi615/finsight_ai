from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from app.models.research_report import ResearchReport


class ResearchReportRepository:
    def add(self, db: Session, report: ResearchReport) -> None:
        db.add(report)

    def get_by_id_in_organization(self, db: Session, report_id: UUID, organization_id: UUID) -> ResearchReport | None:
        return db.scalar(select(ResearchReport).where(ResearchReport.id == report_id, ResearchReport.organization_id == organization_id))

    def get_for_job_in_organization(self, db: Session, job_id: UUID, organization_id: UUID) -> ResearchReport | None:
        return db.scalar(select(ResearchReport).where(ResearchReport.research_job_id == job_id, ResearchReport.organization_id == organization_id))

    def list_in_organization(self, db: Session, organization_id: UUID, offset: int, limit: int) -> tuple[list[ResearchReport], int]:
        query = select(ResearchReport).where(ResearchReport.organization_id == organization_id)
        total = len(list(db.scalars(query)))
        return list(db.scalars(query.order_by(ResearchReport.created_at.desc()).offset(offset).limit(limit))), total

    def delete_all_in_organization(self, db: Session, organization_id: UUID) -> int:
        result = db.execute(delete(ResearchReport).where(ResearchReport.organization_id == organization_id))
        return int(result.rowcount or 0)
