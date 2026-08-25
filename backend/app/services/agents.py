import asyncio
import logging
from datetime import datetime, timezone
from pydantic import ValidationError
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
        locations = [".".join(str(part) for part in error["loc"]) for error in exc.errors()[:3]]
        detail = ", ".join(locations) or "unknown field"
        logger.warning("groq_schema_validation_failed", extra={"agent": agent, "response_parsing_stage": "pydantic_validation", "validation_failure_fields": detail})
        raise AgentFailure(f"LLM returned invalid structured analysis (invalid fields: {detail}).", category="llm_invalid_response") from exc


def _only_supplied_evidence(result, supplied: list[Evidence]):
    allowed = {evidence_identity(item) for item in supplied}
    returned = {evidence_identity(item) for item in result.evidence}
    if not returned.issubset(allowed):
        raise AgentFailure("Agent returned unsupported evidence.", category="llm_invalid_response")
    return result


class MarketAnalystAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, context: dict) -> MarketAnalysis:
        rows = context["market"]
        evidence = [Evidence(source_type="MARKET", source_id=str(row.id), snippet=f"{row.timestamp.date()}: close {row.close}") for row in rows]
        if not rows:
            return MarketAnalysis(summary="Historical market-price data is unavailable from the configured provider. Do not infer or fabricate historical price performance.", metrics={}, signals=[], evidence=[])
        closes = [float(row.close) for row in rows]
        metrics = {"start_close": closes[0], "end_close": closes[-1], "price_change": closes[-1] - closes[0], "price_change_percent": ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] else 0.0, "high_close": max(closes), "low_close": min(closes)}
        payload = {"question": context["question"], "market": [{"timestamp": row.timestamp.isoformat(), "open": str(row.open), "high": str(row.high), "low": str(row.low), "close": str(row.close), "volume": row.volume} for row in rows], "calculated_metrics": metrics, "evidence": [item.model_dump() for item in evidence]}
        result = await self._complete(MarketAnalysis, market.PROMPT, payload)
        return _only_supplied_evidence(result.model_copy(update={"metrics": metrics}), evidence)
    async def _complete(self, model, prompt, payload):
        try: return _validate(model, await self.llm.complete_json(prompt, payload, agent="market_analyst", max_output_tokens=2000), agent="market_analyst")
        except LLMProviderError as exc: raise AgentFailure("Market agent failed.", category=exc.category) from exc


class NewsAnalystAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, context: dict) -> NewsAnalysis:
        articles = context["news"]
        evidence = [Evidence(source_type="NEWS", source_id=str(article.id), snippet=article.title, url=article.url) for article in articles]
        if not articles:
            return NewsAnalysis(summary="No recent news is available.", themes=[], signals=[], evidence=[])
        evidence = evidence[:10]
        payload = {"question": context["question"], "articles": [{"title": article.title, "summary": (article.summary or "")[:500], "published_at": article.published_at.isoformat()} for article in articles[:10]], "evidence": [item.model_dump() for item in evidence]}
        try: return _only_supplied_evidence(_validate(NewsAnalysis, await self.llm.complete_json(news.PROMPT, payload, agent="news_analyst", max_output_tokens=2000), agent="news_analyst"), evidence)
        except LLMProviderError as exc: raise AgentFailure("News agent failed.", category=exc.category) from exc


class DocumentRagAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, question: str, chunks: list[DocumentChunk]) -> DocumentAnalysis:
        evidence = [Evidence(source_type="DOCUMENT", source_id=str(chunk.id), snippet=chunk.content[:500], url=chunk.source_url) for chunk in chunks]
        if not chunks:
            return DocumentAnalysis(summary="No tenant-owned documents matched this request.", findings=[], evidence=[])
        evidence = evidence[:3]
        payload = {"question": question, "chunks": [{"document_id": str(chunk.document_id), "source": chunk.source_url, "page_number": chunk.page_number, "section": chunk.section, "chunk_index": chunk.chunk_index, "text": chunk.content[:800]} for chunk in chunks[:3]], "evidence": [item.model_dump() for item in evidence]}
        try: return _only_supplied_evidence(_validate(DocumentAnalysis, await self.llm.complete_json(document.PROMPT, payload, agent="document_rag_agent", max_output_tokens=700), agent="document_rag_agent"), evidence)
        except LLMProviderError as exc: raise AgentFailure("Document agent failed.", category=exc.category) from exc


class ResearchSynthesizer:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def synthesize(self, company, market_analysis: MarketAnalysis, news_analysis: NewsAnalysis, document_analysis: DocumentAnalysis) -> ResearchSynthesis:
        allowed = {evidence_identity(item) for item in [*market_analysis.evidence, *news_analysis.evidence, *document_analysis.evidence]}
        if not allowed:
            raise AgentFailure("Research cannot be synthesized without evidence.")
        supplied_evidence = [*market_analysis.evidence, *news_analysis.evidence, *document_analysis.evidence]
        payload = {"company": {"ticker": company.ticker, "name": company.name}, "market": market_analysis.model_dump(), "news": news_analysis.model_dump(), "documents": document_analysis.model_dump(), "allowed_evidence": [item.model_dump() for item in supplied_evidence]}
        try: result = _validate(ResearchSynthesis, await self.llm.complete_json(synthesis.PROMPT, payload, agent="research_synthesizer", max_output_tokens=1500), agent="research_synthesizer")
        except LLMProviderError as exc: raise AgentFailure("Synthesizer failed.", category=exc.category) from exc
        returned = {evidence_identity(item) for item in result.evidence}
        if not returned.issubset(allowed):
            raise AgentFailure("Synthesizer returned unsupported evidence.", category="llm_invalid_response")
        return result
