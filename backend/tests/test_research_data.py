from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fastapi import HTTPException
import pytest
from app.core.database import get_db
from app.api.v1.companies import get_research_service
from app.providers.base import CompanyData, MarketBarData, NewsArticleData, ProviderError
from app.services.research_service import ResearchService
from .conftest import auth_headers, signup


class FakeMarketProvider:
    def __init__(self):
        self.company_calls = 0
        self.market_calls = 0
        self.fail_market = False

    def get_company(self, ticker: str) -> CompanyData:
        self.company_calls += 1
        return CompanyData(ticker=ticker, name="NVIDIA Corporation", exchange="NASDAQ", country="US", sector="Technology", industry="Semiconductors")

    def get_market_data(self, ticker: str, from_date: date, to_date: date) -> list[MarketBarData]:
        self.market_calls += 1
        if self.fail_market:
            raise ProviderError("down")
        return [MarketBarData(timestamp=datetime.combine(from_date, datetime.min.time(), tzinfo=timezone.utc), open=Decimal("100.10"), high=Decimal("105.20"), low=Decimal("99.50"), close=Decimal("104.75"), volume=123456, source="fake-market")]


class FakeNewsProvider:
    def __init__(self):
        self.news_calls = 0
        self.fail_news = False

    def get_news(self, ticker: str, from_date: date, to_date: date) -> list[NewsArticleData]:
        self.news_calls += 1
        if self.fail_news:
            raise ProviderError("down")
        published_at = datetime.combine(from_date, datetime.min.time(), tzinfo=timezone.utc)
        return [
            NewsArticleData(title="NVIDIA update", summary="A market update", url="https://example.test/nvda#first", published_at=published_at, source_name="Fake News", author="Author"),
            NewsArticleData(title="NVIDIA update", summary="A market update", url="https://example.test/nvda#second", published_at=published_at, source_name="Fake News", author="Author"),
        ]


def service_and_db(client):
    market = FakeMarketProvider()
    news = FakeNewsProvider()
    db = next(client.app.dependency_overrides[get_db]())
    return ResearchService(market_provider=market, news_provider=news), market, news, db


def test_company_ticker_is_normalized_and_persisted_once(client):
    service, market, _, db = service_and_db(client)
    first = service.resolve_company(db, " nvda ")
    second = service.resolve_company(db, "NVDA")
    assert first.ticker == "NVDA"
    assert second.id == first.id
    assert market.company_calls == 1


def test_market_data_is_normalized_persisted_and_cached(client):
    service, market, _, db = service_and_db(client)
    _, first = service.get_market_data(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3))
    _, second = service.get_market_data(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3))
    assert len(first) == len(second) == 1
    assert first[0].close == Decimal("104.750000")
    assert market.market_calls == 1


def test_invalid_market_date_range_and_provider_failure_are_clean(client):
    service, market, _, db = service_and_db(client)
    with pytest.raises(HTTPException) as invalid_range:
        service.get_market_data(db, "NVDA", date(2026, 1, 3), date(2026, 1, 1))
    assert invalid_range.value.status_code == 422
    market.fail_market = True
    with pytest.raises(HTTPException) as failure:
        service.get_market_data(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3))
    assert failure.value.status_code == 503
    assert failure.value.detail["code"] == "PROVIDER_UNAVAILABLE"


def test_news_is_normalized_deduplicated_and_persisted(client):
    service, _, news, db = service_and_db(client)
    _, articles = service.get_news(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3), limit=20)
    assert len(articles) == 1
    assert articles[0].url == "https://example.test/nvda"
    assert news.news_calls == 1


def test_news_provider_failure_is_clean(client):
    service, _, news, db = service_and_db(client)
    news.fail_news = True
    with pytest.raises(HTTPException) as failure:
        service.get_news(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3), limit=20)
    assert failure.value.status_code == 503


def test_company_context_contains_persisted_company_market_news_and_sources(client):
    service, _, _, db = service_and_db(client)
    start = date.today() - timedelta(days=1)
    service.get_market_data(db, "NVDA", start, date.today())
    service.get_news(db, "NVDA", start, date.today(), limit=20)
    context = service.build_company_context(db, "NVDA")
    assert context["company"].ticker == "NVDA"
    assert len(context["market"]) == 1
    assert len(context["news"]) == 1
    assert {source.source_type.value for source in context["sources"]} == {"MARKET", "NEWS"}


def test_authenticated_company_routes_use_service_and_return_normalized_data(client):
    account = signup(client)
    market = FakeMarketProvider()
    news = FakeNewsProvider()
    client.app.dependency_overrides[get_research_service] = lambda: ResearchService(market_provider=market, news_provider=news)
    headers = auth_headers(account["access_token"])
    company = client.get("/api/v1/companies/nvda", headers=headers)
    market_response = client.get("/api/v1/companies/NVDA/market?from=2026-01-01&to=2026-01-03", headers=headers)
    news_response = client.get("/api/v1/companies/NVDA/news?from=2026-01-01&to=2026-01-03", headers=headers)
    context_response = client.get("/api/v1/companies/NVDA/context", headers=headers)
    assert company.status_code == market_response.status_code == news_response.status_code == context_response.status_code == 200
    assert company.json()["data"]["ticker"] == "NVDA"
    assert market_response.json()["data"][0]["close"] == "104.750000"
    assert len(news_response.json()["data"]) == 1
    assert {item["source_type"] for item in context_response.json()["data"]["sources"]} == {"MARKET", "NEWS"}
