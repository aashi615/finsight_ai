from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fastapi import HTTPException
import httpx
import pytest
from app.core.database import get_db
from app.core.config import settings
from app.api.v1.companies import get_research_service
from app.providers.base import CompanyData, MarketBarData, NewsArticleData, ProviderError, UnknownTickerError
from app.providers.finnhub import FinnhubProvider
from app.providers.fallback_market import FallbackMarketDataProvider
from app.providers.yahoo_finance import YahooFinanceProvider
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


def test_company_name_and_ticker_resolution_use_provider_canonical_symbol(client):
    class NameResolvingMarket(FakeMarketProvider):
        def get_company(self, query):
            self.company_calls += 1
            mapping = {"NVIDIA": "NVDA", "NVDA": "NVDA", "MICROSOFT": "MSFT", "MSFT": "MSFT"}
            canonical = mapping.get(query.strip().upper())
            if not canonical:
                raise UnknownTickerError("not found")
            return CompanyData(ticker=canonical, name="Microsoft Corporation" if canonical == "MSFT" else "NVIDIA Corporation")

    db = next(client.app.dependency_overrides[get_db]())
    provider = NameResolvingMarket()
    service = ResearchService(provider, FakeNewsProvider())
    assert service.resolve_company(db, "NVIDIA").ticker == "NVDA"
    assert service.resolve_company(db, "nvda").ticker == "NVDA"
    assert service.resolve_company(db, "Microsoft").ticker == "MSFT"
    assert service.resolve_company(db, "microsoft").ticker == "MSFT"
    assert service.resolve_company(db, "MSFT").ticker == "MSFT"


def test_unknown_company_is_resolution_error_not_database_error(client):
    class UnknownCompanyMarket(FakeMarketProvider):
        def get_company(self, query):
            raise UnknownTickerError("not found")

    db = next(client.app.dependency_overrides[get_db]())
    with pytest.raises(HTTPException) as failure:
        ResearchService(UnknownCompanyMarket(), FakeNewsProvider()).resolve_company(db, "not-a-company")
    assert failure.value.status_code == 404
    assert failure.value.detail["code"] == "UNKNOWN_TICKER"


def test_company_resolution_provider_failure_is_not_database_error(client):
    class UnavailableMarket(FakeMarketProvider):
        def get_company(self, query):
            raise ProviderError("provider unavailable")

    db = next(client.app.dependency_overrides[get_db]())
    with pytest.raises(HTTPException) as failure:
        ResearchService(UnavailableMarket(), FakeNewsProvider()).resolve_company(db, "Microsoft")
    assert failure.value.status_code == 503
    assert failure.value.detail["code"] == "PROVIDER_UNAVAILABLE"


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


def test_finnhub_candle_403_retains_status_without_exposing_the_api_key(monkeypatch):
    api_key = "do-not-log-this-key"

    def forbidden_get(url, *, params, timeout):
        request = httpx.Request("GET", url, params=params)
        response = httpx.Response(403, request=request, json={"error": "You don't have access to this resource."})
        response.raise_for_status()

    monkeypatch.setattr(httpx, "get", forbidden_get)
    provider = FinnhubProvider(api_key=api_key, base_url="https://finnhub.test/api/v1")
    with pytest.raises(ProviderError) as failure:
        provider.get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 3))
    assert failure.value.status_code == 403
    assert api_key not in str(failure.value)


def test_finnhub_resolves_company_name_to_canonical_ticker(monkeypatch):
    responses = [
        {"name": ""},
        {"result": [{"symbol": "NVDA", "description": "NVIDIA Corporation"}]},
        {"name": "NVIDIA Corporation", "exchange": "NASDAQ", "country": "US", "finnhubIndustry": "Semiconductors"},
    ]

    def fake_get(url, *, params, timeout):
        request = httpx.Request("GET", url, params=params)
        return type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: responses.pop(0)})()

    monkeypatch.setattr(httpx, "get", fake_get)
    company = FinnhubProvider(api_key="test-key", base_url="https://finnhub.test/api/v1").get_company("NVIDIA Corporation")
    assert company.ticker == "NVDA"
    assert company.name == "NVIDIA Corporation"


@pytest.mark.parametrize("payload", [
    {"s": "ok", "t": [], "o": [], "h": [], "l": [], "c": []},
    {"s": "ok", "t": [1], "o": [1], "h": [1], "l": [1], "c": [1], "v": ["bad"]},
])
def test_finnhub_200_malformed_candle_payload_is_controlled_provider_error(monkeypatch, payload):
    def fake_get(url, *, params, timeout):
        return type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: payload})()
    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ProviderError, match="malformed"):
        FinnhubProvider(api_key="test-key").get_market_data("MSFT", date(2026, 1, 1), date(2026, 1, 3))


def test_finnhub_200_malformed_news_payload_is_controlled_provider_error(monkeypatch):
    def fake_get(url, *, params, timeout):
        return type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: [{"headline": "missing url"}]})()
    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(ProviderError, match="malformed"):
        FinnhubProvider(api_key="test-key").get_news("MSFT", date(2026, 1, 1), date(2026, 1, 3))


class StaticMarketProvider:
    def __init__(self, bars=None, error=None):
        self.bars = bars or []
        self.error = error
        self.calls = 0

    def get_market_data(self, ticker, from_date, to_date):
        self.calls += 1
        if self.error:
            raise self.error
        return self.bars


def test_fallback_uses_finnhub_when_historical_data_is_available():
    bar = FakeMarketProvider().get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 1))[0]
    primary, yahoo = StaticMarketProvider([bar]), StaticMarketProvider()
    bars = FallbackMarketDataProvider(primary, yahoo).get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 3))
    assert bars == [bar]
    assert primary.calls == 1 and yahoo.calls == 0


def test_finnhub_403_falls_back_to_yahoo_with_real_normalized_rows(caplog):
    bar = MarketBarData(timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), open=Decimal("100"), high=Decimal("110"), low=Decimal("99"), close=Decimal("105"), volume=1234, source="yahoo_finance")
    primary = StaticMarketProvider(error=ProviderError("forbidden", status_code=403))
    yahoo = StaticMarketProvider([bar])
    bars = FallbackMarketDataProvider(primary, yahoo).get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 3))
    assert bars == [bar]
    assert yahoo.calls == 1
    assert "falling back to Yahoo Finance" in caplog.text
    assert "Yahoo Finance historical market data fetched successfully" in caplog.text
    assert "forbidden" not in caplog.text


def test_both_historical_market_providers_failing_raises_clear_error():
    provider = FallbackMarketDataProvider(StaticMarketProvider(error=ProviderError("forbidden", status_code=403)), StaticMarketProvider(error=ProviderError("empty history")))
    with pytest.raises(ProviderError, match="Finnhub and Yahoo Finance"):
        provider.get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 3))


def test_yahoo_history_is_normalized_and_invalid_rows_are_rejected(monkeypatch):
    class History:
        empty = False
        def iterrows(self):
            yield datetime(2026, 1, 2, 16, tzinfo=timezone.utc), {"Open": 100.1, "High": 105.2, "Low": 99.5, "Close": 104.75, "Volume": 123456}
    class Ticker:
        def __init__(self, symbol): self.symbol = symbol
        def history(self, **kwargs):
            assert kwargs["start"] == date(2026, 1, 1)
            assert kwargs["end"] == date(2026, 1, 4)  # Yahoo's exclusive end includes Jan 3.
            assert kwargs["interval"] == "1d" and kwargs["auto_adjust"] is False
            return History()
    monkeypatch.setattr("app.providers.yahoo_finance.yf.Ticker", Ticker)
    bars = YahooFinanceProvider().get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 3))
    assert bars == [MarketBarData(timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), open=Decimal("100.1"), high=Decimal("105.2"), low=Decimal("99.5"), close=Decimal("104.75"), volume=123456, source="yahoo_finance")]


def test_yahoo_empty_history_raises_clear_provider_error(monkeypatch):
    class History: empty = True
    class Ticker:
        def __init__(self, symbol): pass
        def history(self, **kwargs): return History()
    monkeypatch.setattr("app.providers.yahoo_finance.yf.Ticker", Ticker)
    with pytest.raises(ProviderError, match="no historical market data"):
        YahooFinanceProvider(retries=0).get_market_data("NVDA", date(2026, 1, 1), date(2026, 1, 3))


def test_yahoo_empty_transient_response_retries_before_returning_data(monkeypatch):
    class EmptyHistory:
        empty = True
    class History:
        empty = False
        def iterrows(self):
            yield datetime(2026, 1, 2, tzinfo=timezone.utc), {"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1}
    class Ticker:
        calls = 0
        def __init__(self, symbol): pass
        def history(self, **kwargs):
            Ticker.calls += 1
            return EmptyHistory() if Ticker.calls == 1 else History()
    monkeypatch.setattr("app.providers.yahoo_finance.yf.Ticker", Ticker)
    waits = []
    bars = YahooFinanceProvider(retries=1, sleep=waits.append).get_market_data("MSFT", date(2026, 1, 1), date(2026, 1, 3))
    assert len(bars) == 1
    assert waits == [settings.yahoo_retry_seconds]


def test_yahoo_transient_timeout_retries_once_before_accepting_valid_history(monkeypatch):
    class History:
        empty = False
        def iterrows(self):
            yield datetime(2026, 1, 2, tzinfo=timezone.utc), {"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1}
    class Ticker:
        calls = 0
        def __init__(self, symbol): pass
        def history(self, **kwargs):
            Ticker.calls += 1
            if Ticker.calls == 1:
                raise TimeoutError("provider timeout")
            return History()
    monkeypatch.setattr("app.providers.yahoo_finance.yf.Ticker", Ticker)
    waits = []
    bars = YahooFinanceProvider(retries=1, sleep=waits.append).get_market_data("MSFT", date(2026, 1, 1), date(2026, 1, 3))
    assert len(bars) == 1
    assert waits == [settings.yahoo_retry_seconds]


def test_news_is_normalized_deduplicated_and_persisted(client):
    service, _, news, db = service_and_db(client)
    _, articles = service.get_news(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3), limit=20)
    assert len(articles) == 1
    assert articles[0].url == "https://example.test/nvda"
    assert news.news_calls == 1


def test_news_ingestion_is_idempotent_across_repeated_provider_batches(client):
    service, _, _, db = service_and_db(client)
    start = date(2026, 1, 1)
    _, first = service.get_news(db, "NVDA", start, date(2026, 1, 3), limit=20)
    _, second = service.get_news(db, "NVDA", start, date(2026, 1, 3), limit=20)
    assert len(first) == len(second) == 1
    assert first[0].url == second[0].url == "https://example.test/nvda"


def test_news_ingestion_recovers_after_duplicate_url_conflict(client):
    service, _, _, db = service_and_db(client)
    company = service.resolve_company(db, "NVDA")
    published_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    duplicate = NewsArticleData(title="Duplicate", summary=None, url="https://example.test/duplicate#fragment", published_at=published_at, source_name="Test", author=None)
    service._persist_news(db, company, [duplicate, duplicate])
    # A second ingestion is the production race/conflict shape: it must be a
    # no-op instead of surfacing uq_news_articles_url.
    service._persist_news(db, company, [duplicate])
    _, articles = service.get_news(db, "NVDA", date(2026, 1, 1), date(2026, 1, 3), limit=20)
    assert [item.url for item in articles].count("https://example.test/duplicate") == 1


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
