from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol


class ProviderError(Exception):
    """A provider failed without exposing implementation details to API clients."""


class UnknownTickerError(ProviderError):
    pass


@dataclass(frozen=True)
class CompanyData:
    ticker: str
    name: str
    exchange: str | None = None
    country: str | None = None
    sector: str | None = None
    industry: str | None = None


@dataclass(frozen=True)
class MarketBarData:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    source: str


@dataclass(frozen=True)
class NewsArticleData:
    title: str
    summary: str | None
    url: str
    published_at: datetime
    source_name: str
    author: str | None


class MarketDataProvider(Protocol):
    def get_company(self, ticker: str) -> CompanyData: ...
    def get_market_data(self, ticker: str, from_date: date, to_date: date) -> list[MarketBarData]: ...


class NewsProvider(Protocol):
    def get_news(self, ticker: str, from_date: date, to_date: date) -> list[NewsArticleData]: ...
