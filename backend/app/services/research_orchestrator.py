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
            result, summaries = asyncio.run(self._run_agents(db, job, company))
            job.status = ResearchJobStatus.COMPLETED
            job.result = {**result.model_dump(mode="json"), **summaries, "final_result": result.model_dump(mode="json")}
            job.completed_at = datetime.now(timezone.utc)
            self.reports.add(db, ResearchReport(organization_id=job.organization_id, research_job_id=job.id, company_id=company.id, title=f"{company.ticker} research report", executive_summary=result.executive_summary, report_data=job.result))
            db.commit()
            db.refresh(job)
            logger.info("research job completed", extra={"job_id": str(job.id), "company": company.ticker, "duration_seconds": round(time.monotonic() - started, 3)})
            return job
        except Exception as exc:
            db.rollback()
            job.status = ResearchJobStatus.FAILED
            category = getattr(exc, "category", "research_processing")
            job.error_message = "Historical market data is unavailable from all configured providers." if self._is_market_data_failure(exc) else self._failure_message(category)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning("research job failed", extra={"job_id": str(job.id), "company": company.ticker, "duration_seconds": round(time.monotonic() - started, 3), "error_type": type(exc).__name__, "error_category": category, "error_detail": str(exc), "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None})
            return job

    @staticmethod
    def _is_market_data_failure(exc: Exception) -> bool:
        return isinstance(exc, HTTPException) and exc.status_code == 503 and isinstance(exc.detail, dict) and exc.detail.get("code") == "PROVIDER_UNAVAILABLE"

    @staticmethod
    def _failure_message(category: str) -> str:
        messages = {"llm_quota_exhausted": "Research processing failed: the LLM provider quota is exhausted.", "llm_rate_limit": "Research processing failed: the LLM provider is rate limited.", "llm_timeout": "Research processing failed: the LLM provider timed out.", "llm_incomplete_response": "Research processing failed: the LLM response was incomplete.", "llm_invalid_response": "Research processing failed: the LLM returned an invalid structured response.", "llm_provider_error": "Research processing failed: the LLM provider rejected the request."}
        return messages.get(category, "Research processing failed.")

    async def _run_agents(self, db: Session, job: ResearchJob, company) -> tuple[ResearchSynthesis, dict]:
        # Fetch sync providers off the event loop; yfinance is blocking.
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=30)
        _, market_rows = await self.research_service.get_market_data_async(db, company.ticker, from_date, to_date)
        _, news_rows = self.research_service.get_news(db, company.ticker, from_date, to_date, limit=10)
        context = {"company": company, "market": market_rows, "news": news_rows, "question": job.question}
        chunks = self.rag_service.retrieve(db, job.organization_id, job.question, company.id, limit=3)
        # Logical branches are independent. They are deliberately scheduled one
        # at a time; the provider's rolling TPM manager then remains the single
        # source of truth and a gather burst cannot exhaust Groq's 8k TPM limit.
        market_result = await self._run_branch(job, company.ticker, "market_analyst", self.market_agent.analyze(context))
        news_result = await self._run_branch(job, company.ticker, "news_analyst", self.news_agent.analyze(context))
        document_result = await self._run_branch(job, company.ticker, "document_rag_agent", self.document_agent.analyze(job.question, chunks))
        summaries = {
            "news_summary": news_result.model_dump(mode="json") if news_result else None,
            "market_summary": market_result.model_dump(mode="json") if market_result else None,
            "rag_summary": document_result.model_dump(mode="json") if document_result else None,
        }
        # Persist branch state before synthesis so successful independent work is
        # available if final synthesis (or a later branch) fails.
        job.result = summaries
        db.commit()
        final_result = await self.synthesizer.synthesize(market_result, news_result, document_result)
        return final_result, summaries

    async def _run_branch(self, job: ResearchJob, ticker: str, agent_name: str, work):
        try:
            return await self._run_agent(job, ticker, agent_name, work)
        except Exception:
            # A missing branch is explicit to synthesis. Successful siblings are
            # retained rather than being erased by an all-or-nothing gather.
            return None

    async def _run_agent(self, job: ResearchJob, ticker: str, agent_name: str, work):
        started = time.monotonic()
        try:
            result = await work
            logger.info("research agent completed", extra={"job_id": str(job.id), "company": ticker, "agent": agent_name, "duration_seconds": round(time.monotonic() - started, 3), "success": True})
            return result
        except Exception as exc:
            logger.warning("research agent failed", extra={"job_id": str(job.id), "company": ticker, "agent": agent_name, "duration_seconds": round(time.monotonic() - started, 3), "success": False, "error_type": type(exc).__name__, "error_category": getattr(exc, "category", "agent_failure"), "error_detail": str(exc), "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None})
            raise
