from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.models.research_job import ResearchJobStatus


class Evidence(BaseModel):
    # Internal, stable reference used only between agents.  It is excluded from
    # serialized API/report output so the existing public Evidence shape stays
    # unchanged.
    evidence_id: str | None = Field(default=None, exclude=True)
    source_type: str
    source_id: str
    snippet: str = Field(min_length=1, max_length=1000)
    url: str | None = None


class EvidenceBackedClaim(BaseModel):
    claim: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


def evidence_identity(item: Evidence) -> tuple[str, str, str, str | None]:
    return item.source_type, item.source_id, item.snippet, item.url


class EvidenceBoundAnalysis(BaseModel):
    """Base type for analyses whose claims must cite their own evidence set."""

    evidence: list[Evidence]

    @model_validator(mode="after")
    def claims_must_use_listed_evidence(self):
        allowed = {evidence_identity(item) for item in self.evidence}
        for claim in getattr(self, "signals", []) + getattr(self, "findings", []):
            if not {evidence_identity(item) for item in claim.evidence}.issubset(allowed):
                raise ValueError("Claims must cite evidence included in the analysis.")
        return self


class MarketAnalysis(EvidenceBoundAnalysis):
    agent: Literal["market_analyst"] = "market_analyst"
    summary: str
    metrics: dict[str, float]
    signals: list[EvidenceBackedClaim]
    evidence: list[Evidence]


class NewsAnalysis(EvidenceBoundAnalysis):
    agent: Literal["news_analyst"] = "news_analyst"
    summary: str
    themes: list[str]
    signals: list[EvidenceBackedClaim]
    evidence: list[Evidence]


class DocumentAnalysis(EvidenceBoundAnalysis):
    agent: Literal["document_rag_agent"] = "document_rag_agent"
    summary: str
    findings: list[EvidenceBackedClaim]
    evidence: list[Evidence]


class ResearchSynthesis(EvidenceBoundAnalysis):
    executive_summary: str
    company_overview: str
    market_analysis: str
    news_analysis: str
    growth_catalysts: list[EvidenceBackedClaim]
    key_risks: list[EvidenceBackedClaim]
    key_opportunities: list[EvidenceBackedClaim]
    competitive_landscape: str
    valuation: str
    conclusion: str
    evidence: list[Evidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    generated_at: datetime

    @model_validator(mode="after")
    def synthesis_claims_must_cite_evidence(self):
        allowed = {evidence_identity(item) for item in self.evidence}
        for claim in [*self.growth_catalysts, *self.key_risks, *self.key_opportunities]:
            if not {evidence_identity(item) for item in claim.evidence}.issubset(allowed):
                raise ValueError("Synthesis claims must cite evidence included in the synthesis.")
        return self


class ResearchRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    question: str = Field(min_length=5, max_length=2000)


class ResearchJobOut(BaseModel):
    id: UUID
    status: ResearchJobStatus
    question: str
    result: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    report_id: UUID | None = None
    model_config = ConfigDict(from_attributes=True)


class ResearchReportOut(BaseModel):
    id: UUID
    research_job_id: UUID
    company_id: UUID | None
    title: str
    executive_summary: str
    report_data: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class PaginatedResearchJobs(BaseModel):
    items: list[ResearchJobOut]
    page: int
    page_size: int
    total: int


class PaginatedResearchReports(BaseModel):
    items: list[ResearchReportOut]
    page: int
    page_size: int
    total: int
