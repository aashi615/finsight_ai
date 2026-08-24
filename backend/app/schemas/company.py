from datetime import datetime
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from app.models.research_source import SourceType


class CompanyOut(BaseModel):
    id: UUID
    ticker: str
    name: str
    exchange: str | None
    country: str | None
    sector: str | None
    industry: str | None
    model_config = ConfigDict(from_attributes=True)


class MarketDataOut(BaseModel):
    id: UUID
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    source: str
    model_config = ConfigDict(from_attributes=True)


class NewsArticleOut(BaseModel):
    id: UUID
    title: str
    summary: str | None
    url: str
    published_at: datetime
    source_name: str
    author: str | None
    model_config = ConfigDict(from_attributes=True)


class ResearchSourceOut(BaseModel):
    id: UUID
    source_type: SourceType
    source_name: str
    source_url: str
    title: str | None
    published_at: datetime | None
    metadata: dict | None = Field(validation_alias="metadata_json", serialization_alias="metadata")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class CompanyContextOut(BaseModel):
    company: CompanyOut
    market: list[MarketDataOut]
    news: list[NewsArticleOut]
    sources: list[ResearchSourceOut]
