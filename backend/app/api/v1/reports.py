from uuid import UUID
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import api_error
from app.models.user import User
from app.repositories.research_report_repository import ResearchReportRepository
from app.schemas.common import SuccessResponse
from app.schemas.research import PaginatedResearchReports, ResearchReportOut

router = APIRouter(prefix="/reports", tags=["reports"])
reports = ResearchReportRepository()


@router.get("", response_model=SuccessResponse[PaginatedResearchReports])
def list_reports(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    items, total = reports.list_in_organization(db, current_user.organization_id, (page - 1) * page_size, page_size)
    return SuccessResponse(data=PaginatedResearchReports(items=items, page=page, page_size=page_size, total=total))


@router.get("/{report_id}", response_model=SuccessResponse[ResearchReportOut])
def get_report(report_id: UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    report = reports.get_by_id_in_organization(db, report_id, current_user.organization_id)
    if not report:
        raise api_error(404, "RESEARCH_REPORT_NOT_FOUND", "Research report was not found.")
    return SuccessResponse(data=report)
