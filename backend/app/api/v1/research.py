from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.rate_limit import limit_research
from app.core.exceptions import api_error
from app.llm.openai_provider import OpenAIProvider
from app.models.user import User
from app.providers.finnhub import FinnhubProvider
from app.repositories.research_job_repository import ResearchJobRepository
from app.repositories.research_report_repository import ResearchReportRepository
from app.schemas.common import SuccessResponse
from app.schemas.research import PaginatedResearchJobs, ResearchJobOut, ResearchRequest
from app.services.agents import DocumentRagAgent, MarketAnalystAgent, NewsAnalystAgent, ResearchSynthesizer
from app.services.rag_service import RagService
from app.services.research_orchestrator import ResearchOrchestrator
from app.services.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])
jobs = ResearchJobRepository()
reports = ResearchReportRepository()


def get_orchestrator() -> ResearchOrchestrator:
    llm = OpenAIProvider()
    provider = FinnhubProvider()
    return ResearchOrchestrator(ResearchService(provider, provider), RagService(llm), MarketAnalystAgent(llm), NewsAnalystAgent(llm), DocumentRagAgent(llm), ResearchSynthesizer(llm))


def job_out(db: Session, job) -> ResearchJobOut:
    report = reports.get_for_job_in_organization(db, job.id, job.organization_id)
    return ResearchJobOut.model_validate(job).model_copy(update={"report_id": report.id if report else None})


@router.post("", response_model=SuccessResponse[ResearchJobOut])
def create_research(payload: ResearchRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(limit_research), orchestrator: ResearchOrchestrator = Depends(get_orchestrator)):
    job = orchestrator.create_job(db, current_user, payload)
    background_tasks.add_task(orchestrator.run_job, db, job.id)
    return SuccessResponse(data=job_out(db, job), message="Research job created.")


@router.get("", response_model=SuccessResponse[PaginatedResearchJobs])
def list_research(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    items, total = jobs.list_in_organization(db, current_user.organization_id, (page - 1) * page_size, page_size)
    return SuccessResponse(data=PaginatedResearchJobs(items=[job_out(db, item) for item in items], page=page, page_size=page_size, total=total))


@router.get("/{job_id}", response_model=SuccessResponse[ResearchJobOut])
def get_research(job_id: UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = jobs.get_by_id_in_organization(db, job_id, current_user.organization_id)
    if not job:
        raise api_error(404, "RESEARCH_JOB_NOT_FOUND", "Research job was not found.")
    return SuccessResponse(data=job_out(db, job))
