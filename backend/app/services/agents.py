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


def _normalize_intermediate_evidence(payload: dict, supplied: list[Evidence] | None) -> dict:
    """Restore required top-level citations only when claims cite exact inputs.

    Specialist schemas require every claim citation to also be present in the
    top-level evidence list. Models occasionally omit this duplicate list even
    while returning exact supplied citations. This normalizes that structural
    omission before Pydantic; an altered/invented claim is deliberately left
    unchanged and therefore still rejected by strict validation.
    """
    if not isinstance(payload, dict) or not supplied:
        return payload
    allowed = {evidence_identity(item): item for item in supplied}
    claims = [*payload.get("signals", []), *payload.get("findings", [])]
    cited: list[dict] = []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("evidence"), list):
            return payload
        for item in claim["evidence"]:
            try:
                candidate = Evidence.model_validate(item)
            except ValidationError:
                return payload
            if evidence_identity(candidate) not in allowed:
                return payload
            cited.append(allowed[evidence_identity(candidate)].model_dump())
    if not cited:
        return payload
    existing = payload.get("evidence")
    if not isinstance(existing, list):
        return payload
    normalized = dict(payload)
    canonical = []
    seen = set()
    for item in [*existing, *cited]:
        try:
            candidate = Evidence.model_validate(item)
        except ValidationError:
            return payload
        identity = evidence_identity(candidate)
        if identity not in allowed:
            return payload
        if identity not in seen:
            seen.add(identity)
            canonical.append(allowed[identity].model_dump())
    if canonical != existing:
        normalized["evidence"] = canonical
        logger.info("agent_top_level_evidence_normalized", extra={"agent": payload.get("agent"), "returned_evidence_count": len(canonical)})
    return normalized


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


def _resolve_synthesis_evidence_ids(payload: dict, manifest: dict[str, Evidence]) -> dict:
    """Turn the final model's ID-only citations into the public Evidence shape.

    The model never supplies source metadata at this boundary. Every accepted
    ID must be an exact locally-created manifest key, then the corresponding
    canonical record is restored for Pydantic and the public API.
    """
    if not isinstance(payload, dict):
        return payload

    def resolve_ids(value, *, field: str, require_one: bool = False) -> tuple[list[str], list[dict]]:
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise AgentFailure(f"Synthesizer returned invalid {field}.", category="llm_invalid_response")
        if require_one and not value:
            raise AgentFailure(f"Synthesizer returned empty {field}.", category="llm_invalid_response")
        if len(value) != len(set(value)):
            raise AgentFailure("Synthesizer returned duplicate evidence IDs.", category="llm_invalid_response")
        unknown = [item for item in value if item not in manifest]
        if unknown:
            raise AgentFailure("Synthesizer returned unsupported evidence IDs.", category="llm_invalid_response")
        return value, [{**manifest[item].model_dump(), "evidence_id": item} for item in value]

    normalized = dict(payload)
    top_ids, top_evidence = resolve_ids(normalized.pop("evidence_ids", None), field="evidence_ids", require_one=True)
    normalized["evidence"] = top_evidence
    for field in ("key_risks", "key_opportunities"):
        claims = normalized.get(field)
        if not isinstance(claims, list):
            continue
        repaired = []
        for claim in claims:
            if not isinstance(claim, dict):
                repaired.append(claim)
                continue
            item = dict(claim)
            claim_ids, claim_evidence = resolve_ids(item.pop("evidence_ids", None), field=f"{field}.evidence_ids", require_one=True)
            if any(evidence_id not in top_ids for evidence_id in claim_ids):
                raise AgentFailure("Synthesizer claim cites evidence absent from top-level evidence_ids.", category="llm_invalid_response")
            item["evidence"] = claim_evidence
            repaired.append(item)
        normalized[field] = repaired
    return _normalize_synthesis_response(normalized)


def _restore_manifest_evidence(payload: dict, manifest: dict[str, Evidence]) -> dict:
    """Replace model-copied citations with the canonical evidence records.

    A model is only trusted to select a manifest ID.  It is not trusted to
    reproduce source metadata verbatim: titles, URLs, and IDs commonly drift
    despite an otherwise useful answer.  Resolving known IDs before schema
    validation removes that fragile duplication while rejecting unknown IDs in
    the existing final validation step.
    """
    if not isinstance(payload, dict):
        return payload

    def resolve(item):
        if not isinstance(item, dict):
            return item
        evidence_id = item.get("evidence_id") or item.get("source_id")
        canonical = manifest.get(evidence_id)
        # evidence_id is intentionally excluded from public serialization, but
        # it must survive this private model-to-model hand-off.
        return {**canonical.model_dump(), "evidence_id": canonical.evidence_id} if canonical else item

    def resolve_list(items):
        return [resolve(item) for item in items] if isinstance(items, list) else items

    normalized = dict(payload)
    top_level = resolve_list(normalized.get("evidence"))
    normalized["evidence"] = top_level
    cited = []
    for field in ("key_risks", "key_opportunities"):
        claims = normalized.get(field)
        if not isinstance(claims, list):
            continue
        repaired_claims = []
        for claim in claims:
            if not isinstance(claim, dict):
                repaired_claims.append(claim)
                continue
            repaired = dict(claim)
            repaired["evidence"] = resolve_list(repaired.get("evidence"))
            repaired_claims.append(repaired)
            if isinstance(repaired["evidence"], list):
                cited.extend(repaired["evidence"])
        normalized[field] = repaired_claims

    # The top-level evidence list is a redundant display index. Add valid
    # claim citations that a model omitted there so the strict public schema is
    # preserved without ever adding evidence that was not in the manifest.
    if isinstance(top_level, list):
        canonical_identities = {evidence_identity(item) for item in manifest.values()}
        seen = set()
        deduplicated = []
        # `resolve` returns dicts; validate only canonical values to avoid
        # accidentally repairing an unknown or malformed citation.
        for item in [*top_level, *cited]:
            try:
                candidate = Evidence.model_validate(item)
            except ValidationError:
                deduplicated.append(item)
                continue
            identity = evidence_identity(candidate)
            if identity in canonical_identities and identity not in seen:
                deduplicated.append({**candidate.model_dump(), "evidence_id": candidate.evidence_id})
                seen.add(identity)
            elif identity not in canonical_identities:
                # Preserve an invalid citation so the final validation reports
                # it instead of silently accepting altered source content.
                deduplicated.append(item)
        normalized["evidence"] = deduplicated
    return _normalize_synthesis_response(normalized)


def _deterministic_synthesis(analyses: list, manifest: dict[str, Evidence]) -> ResearchSynthesis:
    """Safe availability fallback when all final-model attempts are unusable."""
    by_type = {analysis.__class__.__name__: analysis.summary for analysis in analyses}
    return ResearchSynthesis(
        executive_summary="Based on available data, source evidence was collected but automated synthesis was unavailable.",
        company_overview="Company-specific interpretation is unavailable because the final synthesis model did not return a valid response.",
        market_analysis=by_type.get("MarketAnalysis", "Historical market-price data could not be assessed from available sources."),
        news_analysis=by_type.get("NewsAnalysis", "No usable recent-news analysis was available."),
        key_risks=[],
        key_opportunities=[],
        evidence=list(manifest.values()),
        confidence=0.2,
        generated_at=datetime.now(timezone.utc),
    )


def _log_deterministic_fallback(agent: str, exc: Exception) -> None:
    logger.warning(
        "agent_deterministic_fallback_used",
        extra={"agent": agent, "error_category": getattr(exc, "category", "llm_invalid_response"), "error_detail": str(exc)},
    )


async def _complete_validated(llm: LLMProvider, model, prompt: str, payload: dict, *, agent: str, max_output_tokens: int, require_evidence: bool = False, supplied_evidence: list[Evidence] | None = None):
    """Use the provider fallback once when local schema validation rejects GPT-OSS."""
    def validate_response(response):
        if supplied_evidence is not None:
            response = _normalize_intermediate_evidence(response, supplied_evidence)
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
        try:
            return await _complete_validated(self.llm, model, prompt, payload, agent="market_analyst", max_output_tokens=settings.market_max_output_tokens, require_evidence=require_evidence, supplied_evidence=supplied_evidence)
        except (LLMProviderError, AgentFailure) as exc:
            # The calculated metrics and provider evidence are already trusted.
            # Keep them usable if narrative generation is unavailable.
            _log_deterministic_fallback("market_analyst", exc)
            return MarketAnalysis(summary="Market source data was collected; automated narrative analysis was unavailable.", metrics=payload.get("calculated_metrics", {}), signals=[], evidence=supplied_evidence or [])


class NewsAnalystAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, context: dict) -> NewsAnalysis:
        articles = context["news"]
        evidence = [Evidence(source_type="NEWS", source_id=str(article.id), snippet=article.title, url=article.url) for article in articles]
        if not articles:
            return NewsAnalysis(summary="No recent news is available.", themes=[], signals=[], evidence=[])
        evidence = evidence[:settings.news_article_limit]
        payload = {"question": context["question"][:500], "articles": [{"source_id": str(article.id), "title": article.title[:180], "summary": (article.summary or "")[:settings.news_article_snippet_chars], "published_at": article.published_at.date().isoformat()} for article in articles[:settings.news_article_limit]], "evidence": [item.model_dump() for item in evidence]}
        try:
            return await _complete_validated(self.llm, NewsAnalysis, news.PROMPT, payload, agent="news_analyst", max_output_tokens=settings.news_max_output_tokens, require_evidence=True, supplied_evidence=evidence)
        except (LLMProviderError, AgentFailure) as exc:
            _log_deterministic_fallback("news_analyst", exc)
            return NewsAnalysis(summary="Recent news source records were collected; automated narrative analysis was unavailable.", themes=[], signals=[], evidence=evidence)


class DocumentRagAgent:
    def __init__(self, llm: LLMProvider): self.llm = llm
    async def analyze(self, question: str, chunks: list[DocumentChunk]) -> DocumentAnalysis:
        chunks = chunks[:settings.rag_top_k]
        evidence = [Evidence(source_type="DOCUMENT", source_id=str(chunk.id), snippet=chunk.content[:500], url=chunk.source_url) for chunk in chunks]
        if not chunks:
            return DocumentAnalysis(summary="No tenant-owned documents matched this request.", findings=[], evidence=[])
        evidence = evidence[:settings.rag_top_k]
        payload = {"question": question[:500], "chunks": [{"document_id": str(chunk.document_id), "source": chunk.source_url, "page_number": chunk.page_number, "section": chunk.section, "text": chunk.content[:settings.rag_chunk_chars]} for chunk in chunks], "evidence": [item.model_dump() for item in evidence]}
        try:
            return await _complete_validated(self.llm, DocumentAnalysis, document.PROMPT, payload, agent="document_rag_agent", max_output_tokens=settings.rag_max_output_tokens, require_evidence=True, supplied_evidence=evidence)
        except (LLMProviderError, AgentFailure) as exc:
            _log_deterministic_fallback("document_rag_agent", exc)
            return DocumentAnalysis(summary="Relevant tenant document excerpts were collected; automated narrative analysis was unavailable.", findings=[], evidence=evidence)


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
            result = _validate(ResearchSynthesis, _resolve_synthesis_evidence_ids(response, manifest), agent="research_synthesizer")
            return _canonical_synthesis_evidence(result, manifest)
        except (AgentFailure, LLMProviderError) as primary_error:
            # Qwen is already the normal final model. A manifest/schema failure
            # gets one smaller corrective prompt through the same provider TPM
            # scheduler, rather than failing a job after an otherwise usable run.
            logger.warning("synthesizer_validation_retry", extra={"agent": "research_synthesizer", "error_category": primary_error.category})
            try:
                response = await self.llm.complete_json(synthesis.COMPACT_RETRY_PROMPT, payload, agent="research_synthesizer", max_output_tokens=settings.final_compact_max_output_tokens, force_fallback=True)
                result = _validate(ResearchSynthesis, _resolve_synthesis_evidence_ids(response, manifest), agent="research_synthesizer")
                return _canonical_synthesis_evidence(result, manifest)
            except (LLMProviderError, AgentFailure) as retry_error:
                try:
                    response = await self.llm.complete_json(synthesis.FALLBACK_PROMPT, payload, agent="research_synthesizer", max_output_tokens=settings.final_fallback_max_output_tokens, force_fallback=True)
                    result = _validate(ResearchSynthesis, _resolve_synthesis_evidence_ids(response, manifest), agent="research_synthesizer")
                    return _canonical_synthesis_evidence(result, manifest)
                except (LLMProviderError, AgentFailure) as fallback_error:
                    _log_deterministic_fallback("research_synthesizer", fallback_error)
                    return _deterministic_synthesis(analyses, manifest)


def _sample(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    indexes = {round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)}
    return [item for index, item in enumerate(items) if index in indexes]
