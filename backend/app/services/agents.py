import asyncio
import logging
from datetime import datetime, timezone
from pydantic import ValidationError
from app.core.config import settings
from app.llm.base import LLMProvider, LLMProviderError
from app.models.document import DocumentChunk
from app.prompts import document, market, news, synthesis
from app.schemas.research import DocumentAnalysis, Evidence, MarketAnalysis, NewsAnalysis, ResearchSynthesis, evidence_identity

logger = logging.getLogger(__name__)


class AgentFailure(Exception):
    def __init__(self, message: str, *, category: str = "llm_invalid_response"):
        super().__init__(message)
        self.category = category


def _validate(model, payload, *, agent: str | None = None):
    try:
        result = model.model_validate(payload)
        logger.info("groq_schema_validation_succeeded", extra={"agent": agent, "response_parsing_stage": "pydantic_validation"})
        return result
    except ValidationError as exc:
        failures = [f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']} ({error['type']})" for error in exc.errors()[:3]]
        detail = "; ".join(failures) or "unknown field"
        logger.warning("groq_schema_validation_failed", extra={"agent": agent, "response_parsing_stage": "pydantic_validation", "validation_failure_fields": detail})
        raise AgentFailure(f"LLM returned invalid structured analysis (invalid fields: {detail}).", category="llm_invalid_response") from exc


def _only_supplied_evidence(result, supplied: list[Evidence]):
    allowed = {evidence_identity(item) for item in supplied}
    returned = {evidence_identity(item) for item in result.evidence}
    if not returned.issubset(allowed):
        raise AgentFailure("Agent returned unsupported evidence.", category="llm_invalid_response")
    return result


async def _complete_validated(llm: LLMProvider, model, prompt: str, payload: dict, *, agent: str, max_output_tokens: int):
    """Use the provider fallback once when local schema validation rejects GPT-OSS."""
    try:
        response = await llm.complete_json(prompt, payload, agent=agent, max_output_tokens=max_output_tokens)
        return _validate(model, response, agent=agent)
    except AgentFailure:
        if agent == "research_synthesizer":
            raise
        try:
            response = await llm.complete_json(prompt, payload, agent=agent, max_output_tokens=max_output_tokens, force_fallback=True)
            return _validate(model, response, agent=agent)
        except LLMProviderError:
            raise


class MarketAnalystAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, context: dict) -> MarketAnalysis:
        rows = context["market"]
        # Compact, evenly-spaced price points retain the trend without shipping a
        # month of OHLCV rows to the model.
        rows = _sample(rows, settings.market_history_points_limit)
        evidence = [Evidence(source_type="MARKET", source_id=str(row.id), snippet=f"{row.timestamp.date()}: close {row.close}") for row in rows]
        if not rows:
            return MarketAnalysis(summary="Historical market-price data is unavailable from the configured provider. Do not infer or fabricate historical price performance.", metrics={}, signals=[], evidence=[])
        closes = [float(row.close) for row in rows]
        metrics = {"start_close": closes[0], "end_close": closes[-1], "price_change": closes[-1] - closes[0], "price_change_percent": ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] else 0.0, "high_close": max(closes), "low_close": min(closes)}
        payload = {"question": context["question"][:500], "market": [{"timestamp": row.timestamp.date().isoformat(), "close": str(row.close), "volume": row.volume} for row in rows], "calculated_metrics": metrics, "evidence": [item.model_dump() for item in evidence]}
        result = await self._complete(MarketAnalysis, market.PROMPT, payload)
        return _only_supplied_evidence(result.model_copy(update={"metrics": metrics}), evidence)
    async def _complete(self, model, prompt, payload):
        try: return await _complete_validated(self.llm, model, prompt, payload, agent="market_analyst", max_output_tokens=settings.market_max_output_tokens)
        except LLMProviderError as exc: raise AgentFailure("Market agent failed.", category=exc.category) from exc


class NewsAnalystAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, context: dict) -> NewsAnalysis:
        articles = context["news"]
        evidence = [Evidence(source_type="NEWS", source_id=str(article.id), snippet=article.title, url=article.url) for article in articles]
        if not articles:
            return NewsAnalysis(summary="No recent news is available.", themes=[], signals=[], evidence=[])
        evidence = evidence[:settings.news_article_limit]
        payload = {"question": context["question"][:500], "articles": [{"source_id": str(article.id), "title": article.title[:180], "summary": (article.summary or "")[:settings.news_article_snippet_chars], "published_at": article.published_at.date().isoformat()} for article in articles[:settings.news_article_limit]], "evidence": [item.model_dump() for item in evidence]}
        try: return _only_supplied_evidence(await _complete_validated(self.llm, NewsAnalysis, news.PROMPT, payload, agent="news_analyst", max_output_tokens=settings.news_max_output_tokens), evidence)
        except LLMProviderError as exc: raise AgentFailure("News agent failed.", category=exc.category) from exc


class DocumentRagAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, question: str, chunks: list[DocumentChunk]) -> DocumentAnalysis:
        chunks = chunks[:settings.rag_top_k]
        evidence = [Evidence(source_type="DOCUMENT", source_id=str(chunk.id), snippet=chunk.content[:500], url=chunk.source_url) for chunk in chunks]
        if not chunks:
            return DocumentAnalysis(summary="No tenant-owned documents matched this request.", findings=[], evidence=[])
        evidence = evidence[:settings.rag_top_k]
        payload = {"question": question[:500], "chunks": [{"document_id": str(chunk.document_id), "source": chunk.source_url, "page_number": chunk.page_number, "section": chunk.section, "text": chunk.content[:settings.rag_chunk_chars]} for chunk in chunks], "evidence": [item.model_dump() for item in evidence]}
        try: return _only_supplied_evidence(await _complete_validated(self.llm, DocumentAnalysis, document.PROMPT, payload, agent="document_rag_agent", max_output_tokens=settings.rag_max_output_tokens), evidence)
        except LLMProviderError as exc: raise AgentFailure("Document agent failed.", category=exc.category) from exc


class ResearchSynthesizer:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def synthesize(self, market_analysis: MarketAnalysis | None, news_analysis: NewsAnalysis | None, document_analysis: DocumentAnalysis | None) -> ResearchSynthesis:
        analyses = [analysis for analysis in (market_analysis, news_analysis, document_analysis) if analysis is not None]
        allowed = {evidence_identity(item) for analysis in analyses for item in analysis.evidence}
        if not allowed:
            raise AgentFailure("Research cannot be synthesized because every branch was unavailable or returned no evidence.", category="research_insufficient_evidence")
        # This is intentionally the final boundary: no raw API or document data
        # is allowed beyond the three compact, independently-produced summaries.
        payload = {"news_summary": news_analysis.model_dump() if news_analysis else None, "market_summary": market_analysis.model_dump() if market_analysis else None, "rag_summary": document_analysis.model_dump() if document_analysis else None}
        try: result = await _complete_validated(self.llm, ResearchSynthesis, synthesis.PROMPT, payload, agent="research_synthesizer", max_output_tokens=settings.final_max_output_tokens)
        except LLMProviderError as exc: raise AgentFailure("Synthesizer failed.", category=exc.category) from exc
        returned = {evidence_identity(item) for item in result.evidence}
        if not returned.issubset(allowed):
            raise AgentFailure("Synthesizer returned unsupported evidence.", category="llm_invalid_response")
        return result


def _sample(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    indexes = {round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)}
    return [item for index, item in enumerate(items) if index in indexes]
