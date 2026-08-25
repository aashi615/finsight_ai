from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, require_role
from app.core.database import get_db
from app.core.rate_limit import limit_research
from app.core.exceptions import api_error
from app.llm.openai_provider import OpenAIProvider
from app.models.user import User
from app.models.user import Role
from app.providers.finnhub import FinnhubProvider
from app.providers.fallback_market import FallbackMarketDataProvider
from app.providers.yahoo_finance import YahooFinanceProvider
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
    finnhub = FinnhubProvider()
    market_provider = FallbackMarketDataProvider(finnhub, YahooFinanceProvider())
    return ResearchOrchestrator(ResearchService(market_provider, finnhub), RagService(llm), MarketAnalystAgent(llm), NewsAnalystAgent(llm), DocumentRagAgent(llm), ResearchSynthesizer(llm))


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


@router.delete("", response_model=SuccessResponse[dict[str, int]])
def delete_research_history(db: Session = Depends(get_db), current_user: User = Depends(require_role(Role.ADMIN))):
    # Reports are explicitly removed first for databases that do not enforce FK cascades.
    reports.delete_all_in_organization(db, current_user.organization_id)
    deleted_jobs = jobs.delete_all_in_organization(db, current_user.organization_id)
    db.commit()
    return SuccessResponse(data={"deleted_jobs": deleted_jobs}, message="Research history deleted.")


@router.get("/{job_id}", response_model=SuccessResponse[ResearchJobOut])
def get_research(job_id: UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    job = jobs.get_by_id_in_organization(db, job_id, current_user.organization_id)
    if not job:
        raise api_error(404, "RESEARCH_JOB_NOT_FOUND", "Research job was not found.")
    return SuccessResponse(data=job_out(db, job))
