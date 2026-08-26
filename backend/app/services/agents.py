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


def _with_stable_evidence_ids(analysis, source_agent: str):
    """Assign deterministic, internal references to an upstream evidence set."""
    evidence = [
        item.model_copy(update={"evidence_id": f"{source_agent}_{index:03d}"})
        for index, item in enumerate(analysis.evidence, start=1)
    ]
    return analysis.model_copy(update={"evidence": evidence})


def _branch_evidence(analysis) -> list[Evidence]:
    """Return canonical evidence whether it is stored top-level or on claims.

    New Pydantic outputs require claim evidence to also be top-level.  This
    additionally supports already-persisted/legacy branch state where the
    evidence was retained only beneath ``signals`` or ``findings``.
    """
    candidates = list(analysis.evidence)
    for claim in [*getattr(analysis, "signals", []), *getattr(analysis, "findings", [])]:
        candidates.extend(claim.evidence)
    canonical: list[Evidence] = []
    seen = set()
    for item in candidates:
        identity = evidence_identity(item)
        if identity not in seen:
            seen.add(identity)
            canonical.append(item)
    return canonical


def _canonical_synthesis_evidence(result: ResearchSynthesis, manifest: dict[str, Evidence]) -> ResearchSynthesis:
    """Validate model-selected manifest IDs and restore canonical evidence data."""
    # The final-model contract uses source_id as the compact manifest reference
    # so Pydantic still receives its required public fields. Older responses
    # using the internal evidence_id remain accepted during rollout.
    def manifest_id(item: Evidence) -> str | None:
        return item.evidence_id or item.source_id

    returned_ids = [manifest_id(item) for item in result.evidence]
    claim_items = [*result.key_risks, *result.key_opportunities]
    claim_ids = [manifest_id(item) for claim in claim_items for item in claim.evidence]
    unsupported = sorted({item for item in [*returned_ids, *claim_ids] if not item or item not in manifest})
    duplicates = sorted({item for item in returned_ids if item and returned_ids.count(item) > 1})
    missing_top_level = sorted({item for item in claim_ids if item and item not in returned_ids})
    logger.info(
        "synthesizer_evidence_references_received",
        extra={"agent": "research_synthesizer", "returned_evidence_ids": returned_ids, "claim_evidence_ids": claim_ids},
    )
    if unsupported or duplicates or missing_top_level:
        logger.warning(
            "synthesizer_evidence_validation_failed",
            extra={
                "agent": "research_synthesizer",
                "unsupported_evidence_ids": unsupported,
                "duplicate_evidence_ids": duplicates,
                "missing_top_level_evidence_ids": missing_top_level,
            },
        )
        raise AgentFailure("Synthesizer returned unsupported evidence.", category="llm_invalid_response")

    def canonical(items: list[Evidence]) -> list[Evidence]:
        return [manifest[manifest_id(item)] for item in items]

    risks = [claim.model_copy(update={"evidence": canonical(claim.evidence)}) for claim in result.key_risks]
    opportunities = [claim.model_copy(update={"evidence": canonical(claim.evidence)}) for claim in result.key_opportunities]
    return result.model_copy(update={"evidence": canonical(result.evidence), "key_risks": risks, "key_opportunities": opportunities})


def _normalize_synthesis_response(payload: dict) -> dict:
    """Apply the one narrow compatibility normalization before Pydantic.

    The public contract is arrays of evidence-backed claims. Some models return
    one bare risk/opportunity string despite otherwise valid JSON. Preserve it
    as a one-item array only when the response already contains a concrete
    top-level evidence object; otherwise leave the payload untouched so strict
    Pydantic validation rejects it.
    """
    if not isinstance(payload, dict):
        return payload
    top_level_evidence = payload.get("evidence")
    if not isinstance(top_level_evidence, list) or not top_level_evidence:
        return payload
    normalized = dict(payload)
    for field in ("key_risks", "key_opportunities"):
        value = normalized.get(field)
        if isinstance(value, str) and value.strip():
            normalized[field] = [{"claim": value.strip(), "evidence": [top_level_evidence[0]]}]
            logger.info("synthesizer_array_field_normalized", extra={"agent": "research_synthesizer", "normalized_field": field})
    return normalized


async def _complete_validated(llm: LLMProvider, model, prompt: str, payload: dict, *, agent: str, max_output_tokens: int, require_evidence: bool = False, supplied_evidence: list[Evidence] | None = None):
    """Use the provider fallback once when local schema validation rejects GPT-OSS."""
    def validate_response(response):
        result = _validate(model, response, agent=agent)
        if require_evidence and not result.evidence:
            logger.warning("agent_required_evidence_missing", extra={"agent": agent, "supplied_evidence_count": len(payload.get("evidence", [])), "returned_evidence_count": 0})
            raise AgentFailure("LLM returned no evidence despite supplied source evidence.", category="llm_invalid_response")
        if supplied_evidence is not None:
            result = _only_supplied_evidence(result, supplied_evidence)
        return result

    try:
        response = await llm.complete_json(prompt, payload, agent=agent, max_output_tokens=max_output_tokens)
        return validate_response(response)
    except AgentFailure:
        if agent == "research_synthesizer":
            raise
        try:
            response = await llm.complete_json(prompt, payload, agent=agent, max_output_tokens=max_output_tokens, force_fallback=True)
            return validate_response(response)
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
        result = await self._complete(MarketAnalysis, market.PROMPT, payload, require_evidence=True, supplied_evidence=evidence)
        return result.model_copy(update={"metrics": metrics})
    async def _complete(self, model, prompt, payload, *, require_evidence: bool = False, supplied_evidence: list[Evidence] | None = None):
        try: return await _complete_validated(self.llm, model, prompt, payload, agent="market_analyst", max_output_tokens=settings.market_max_output_tokens, require_evidence=require_evidence, supplied_evidence=supplied_evidence)
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
        try: return await _complete_validated(self.llm, NewsAnalysis, news.PROMPT, payload, agent="news_analyst", max_output_tokens=settings.news_max_output_tokens, require_evidence=True, supplied_evidence=evidence)
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
        try: return await _complete_validated(self.llm, DocumentAnalysis, document.PROMPT, payload, agent="document_rag_agent", max_output_tokens=settings.rag_max_output_tokens, require_evidence=True, supplied_evidence=evidence)
        except LLMProviderError as exc: raise AgentFailure("Document agent failed.", category=exc.category) from exc


class ResearchSynthesizer:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def synthesize(self, market_analysis: MarketAnalysis | None, news_analysis: NewsAnalysis | None, document_analysis: DocumentAnalysis | None) -> ResearchSynthesis:
        # Normalize legacy/persisted branch summaries before testing whether
        # synthesis has usable evidence. Any valid branch evidence is enough.
        market_analysis = market_analysis.model_copy(update={"evidence": _branch_evidence(market_analysis)}) if market_analysis else None
        news_analysis = news_analysis.model_copy(update={"evidence": _branch_evidence(news_analysis)}) if news_analysis else None
        document_analysis = document_analysis.model_copy(update={"evidence": _branch_evidence(document_analysis)}) if document_analysis else None
        market_analysis = _with_stable_evidence_ids(market_analysis, "market") if market_analysis else None
        news_analysis = _with_stable_evidence_ids(news_analysis, "news") if news_analysis else None
        document_analysis = _with_stable_evidence_ids(document_analysis, "rag") if document_analysis else None
        analyses = [analysis for analysis in (market_analysis, news_analysis, document_analysis) if analysis is not None]
        manifest = {item.evidence_id: item for analysis in analyses for item in analysis.evidence if item.evidence_id}
        if not manifest:
            raise AgentFailure("Research cannot be synthesized because every branch was unavailable or returned no evidence.", category="research_insufficient_evidence")
        # This is intentionally the final boundary: no raw API or document data
        # is allowed beyond the three compact, independently-produced summaries.
        evidence_manifest = [
            {
                "evidence_id": evidence_id,
                "source_agent": evidence_id.rsplit("_", 1)[0],
                "source_type": item.source_type,
                "title": item.snippet,
                "source": item.url,
            }
            for evidence_id, item in manifest.items()
        ]
        logger.info(
            "synthesizer_evidence_manifest_created",
            # Keep production diagnostics traceable without logging document
            # snippets, article titles, URLs, or other tenant content.
            extra={"agent": "research_synthesizer", "valid_upstream_evidence_ids": list(manifest), "evidence_manifest": [{"evidence_id": item["evidence_id"], "source_agent": item["source_agent"], "source_type": item["source_type"]} for item in evidence_manifest]},
        )
        payload = {"news_summary": news_analysis.model_dump() if news_analysis else None, "market_summary": market_analysis.model_dump() if market_analysis else None, "rag_summary": document_analysis.model_dump() if document_analysis else None, "evidence_manifest": evidence_manifest}
        try:
            response = await self.llm.complete_json(synthesis.PROMPT, payload, agent="research_synthesizer", max_output_tokens=settings.final_max_output_tokens)
            result = _validate(ResearchSynthesis, _normalize_synthesis_response(response), agent="research_synthesizer")
            return _canonical_synthesis_evidence(result, manifest)
        except AgentFailure as primary_error:
            # Qwen is already the normal final model. A manifest/schema failure
            # gets one smaller corrective prompt through the same provider TPM
            # scheduler, rather than failing a job after an otherwise usable run.
            logger.warning("synthesizer_validation_retry", extra={"agent": "research_synthesizer", "error_category": primary_error.category})
            try:
                response = await self.llm.complete_json(synthesis.COMPACT_RETRY_PROMPT, payload, agent="research_synthesizer", max_output_tokens=settings.final_max_output_tokens)
                return _canonical_synthesis_evidence(_validate(ResearchSynthesis, _normalize_synthesis_response(response), agent="research_synthesizer"), manifest)
            except LLMProviderError as exc:
                raise AgentFailure("Synthesizer failed.", category=exc.category) from exc
        except LLMProviderError as exc: raise AgentFailure("Synthesizer failed.", category=exc.category) from exc


def _sample(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    indexes = {round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)}
    return [item for index, item in enumerate(items) if index in indexes]
