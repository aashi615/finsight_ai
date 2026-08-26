import logging

from app.core.config import settings
from app.llm.base import LLMProvider
from app.prompts import planner
from app.schemas.research import ResearchPlan

logger = logging.getLogger(__name__)


class QueryPlanner:
    """Small validated routing layer that safely falls back to legacy routing."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @staticmethod
    def fallback(ticker: str) -> ResearchPlan:
        return ResearchPlan(companies=[ticker], needs_market=True, needs_news=True, needs_documents=True, reasoning="Planner unavailable; using the complete research flow.")

    async def analyze(self, question: str, fallback_ticker: str) -> ResearchPlan:
        try:
            response = await self.llm.complete_json(planner.PROMPT, {"question": question[:2000], "fallback_ticker": fallback_ticker}, agent="query_planner", max_output_tokens=settings.planner_max_output_tokens)
            plan = ResearchPlan.model_validate(response)
        except Exception as exc:
            logger.warning("query_plan_failed", extra={"company": fallback_ticker, "error_type": type(exc).__name__, "error_category": getattr(exc, "category", "planner_invalid_response"), "error_detail": str(exc)})
            fallback = self.fallback(fallback_ticker)
            logger.info("query_plan_fallback_used", extra={"company": fallback_ticker, "fallback_used": True, "reason": "planner_failure", "planned_companies": fallback.companies, "selected_agents": self.selected_agents(fallback), "skipped_agents": self.skipped_agents(fallback)})
            return fallback
        logger.info("query_plan_created", extra={"company": fallback_ticker, "planned_companies": plan.companies, "selected_agents": self.selected_agents(plan), "skipped_agents": self.skipped_agents(plan)})
        return plan

    @staticmethod
    def selected_agents(plan: ResearchPlan) -> list[str]:
        return [agent for required, agent in ((plan.needs_market, "market_analyst"), (plan.needs_news, "news_analyst"), (plan.needs_documents, "document_rag_agent")) if required]

    @staticmethod
    def skipped_agents(plan: ResearchPlan) -> list[str]:
        selected = set(QueryPlanner.selected_agents(plan))
        return [agent for agent in ("market_analyst", "news_analyst", "document_rag_agent") if agent not in selected]
