import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.research_job import ResearchJob, ResearchJobStatus
from app.models.research_report import ResearchReport
from app.models.company import Company
from app.repositories.research_report_repository import ResearchReportRepository
from app.models.user import User
from app.repositories.research_job_repository import ResearchJobRepository
from app.schemas.research import ResearchRequest, ResearchSynthesis
from app.services.agents import AgentFailure, DocumentRagAgent, MarketAnalystAgent, NewsAnalystAgent, ResearchSynthesizer
from app.services.rag_service import RagService
from app.services.research_service import ResearchService

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    def __init__(self, research_service: ResearchService, rag_service: RagService, market_agent: MarketAnalystAgent, news_agent: NewsAnalystAgent, document_agent: DocumentRagAgent, synthesizer: ResearchSynthesizer):
        self.research_service = research_service
        self.rag_service = rag_service
        self.market_agent = market_agent
        self.news_agent = news_agent
        self.document_agent = document_agent
        self.synthesizer = synthesizer
        self.jobs = ResearchJobRepository()
        self.reports = ResearchReportRepository()

    def create_job(self, db: Session, current_user: User, request: ResearchRequest) -> ResearchJob:
        company = self.research_service.resolve_company(db, request.ticker)
        job = ResearchJob(organization_id=current_user.organization_id, created_by=current_user.id, company_id=company.id, status=ResearchJobStatus.PENDING, question=request.question)
        self.jobs.add(db, job)
        db.commit()
        db.refresh(job)
        return job

    def run_job(self, db: Session, job_id) -> None:
        job = db.get(ResearchJob, job_id)
        if not job or job.status != ResearchJobStatus.PENDING:
            return
        job.status = ResearchJobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        db.commit()
        company = db.get(Company, job.company_id)
        if not company:
            job.status = ResearchJobStatus.FAILED
            job.error_message = "Research company is no longer available."
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return
        started = time.monotonic()
        logger.info("research job started", extra={"job_id": str(job.id), "company": company.ticker})
        try:
            result = asyncio.run(self._run_agents(db, job, company))
            job.status = ResearchJobStatus.COMPLETED
            job.result = result.model_dump(mode="json")
            job.completed_at = datetime.now(timezone.utc)
            self.reports.add(db, ResearchReport(organization_id=job.organization_id, research_job_id=job.id, company_id=company.id, title=f"{company.ticker} research report", executive_summary=result.executive_summary, report_data=job.result))
            db.commit()
            db.refresh(job)
            logger.info("research job completed", extra={"job_id": str(job.id), "company": company.ticker, "duration_seconds": round(time.monotonic() - started, 3)})
            return job
        except Exception as exc:
            db.rollback()
            job.status = ResearchJobStatus.FAILED
            job.error_message = "Historical market data is unavailable from all configured providers." if self._is_market_data_failure(exc) else "Research processing failed."
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("research job failed", extra={"job_id": str(job.id), "company": company.ticker, "duration_seconds": round(time.monotonic() - started, 3), "error_type": type(exc).__name__, "error_category": "research_processing"})
            return job

    @staticmethod
    def _is_market_data_failure(exc: Exception) -> bool:
        return isinstance(exc, HTTPException) and exc.status_code == 503 and isinstance(exc.detail, dict) and exc.detail.get("code") == "PROVIDER_UNAVAILABLE"

    async def _run_agents(self, db: Session, job: ResearchJob, company) -> ResearchSynthesis:
        # Fetch sync providers off the event loop; yfinance is blocking.
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=30)
        try:
            _, market_rows = await self.research_service.get_market_data_async(db, company.ticker, from_date, to_date)
        except HTTPException as exc:
            # Price history is useful but must not make all research unavailable.  The
            # market agent and synthesis prompt explicitly represent this limitation
            # and will never infer prices or returns without supplied price evidence.
            if not self._is_market_data_failure(exc):
                raise
            market_rows = []
            logger.warning(
                "continuing research without historical market data",
                extra={"job_id": str(job.id), "company": company.ticker, "provider_status_code": exc.status_code, "error_category": "historical_market_data_unavailable"},
            )
        _, news_rows = self.research_service.get_news(db, company.ticker, from_date, to_date, limit=20)
        context = {"company": company, "market": market_rows, "news": news_rows, "question": job.question}
        chunks = self.rag_service.retrieve(db, job.organization_id, job.question, company.id)
        market_result, news_result, document_result = await asyncio.gather(
            self._run_agent(job, company.ticker, "market_analyst", self.market_agent.analyze(context)),
            self._run_agent(job, company.ticker, "news_analyst", self.news_agent.analyze(context)),
            self._run_agent(job, company.ticker, "document_rag_agent", self.document_agent.analyze(job.question, chunks)),
        )
        return await self.synthesizer.synthesize(company, market_result, news_result, document_result)

    async def _run_agent(self, job: ResearchJob, ticker: str, agent_name: str, work):
        started = time.monotonic()
        try:
            result = await work
            logger.info("research agent completed", extra={"job_id": str(job.id), "company": ticker, "agent": agent_name, "duration_seconds": round(time.monotonic() - started, 3), "success": True})
            return result
        except Exception:
            logger.warning("research agent failed", extra={"job_id": str(job.id), "company": ticker, "agent": agent_name, "duration_seconds": round(time.monotonic() - started, 3), "success": False, "error_category": "agent_failure"})
            raise
