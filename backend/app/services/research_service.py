import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urlsplit, urlunsplit
from sqlalchemy.orm import Session
from app.core.exceptions import api_error
from app.models.company import Company
from app.models.market_data import MarketData
from app.models.news_article import NewsArticle
from app.models.research_source import ResearchSource, SourceType
from app.providers.base import CompanyData, MarketDataProvider, NewsArticleData, NewsProvider, ProviderError, UnknownTickerError
from app.repositories.company_repository import CompanyRepository
from app.repositories.market_data_repository import MarketDataRepository
from app.repositories.news_repository import NewsRepository
from app.repositories.research_source_repository import ResearchSourceRepository


def canonical_url(url: str) -> str:
    """Remove fragments so obvious duplicate article links share one canonical identity."""
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("invalid URL")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


class ResearchService:
    def __init__(self, market_provider: MarketDataProvider, news_provider: NewsProvider):
        self.market_provider = market_provider
        self.news_provider = news_provider
        self.companies = CompanyRepository()
        self.market_data = MarketDataRepository()
        self.news = NewsRepository()
        self.sources = ResearchSourceRepository()

    def resolve_company(self, db: Session, ticker: str) -> Company:
        normalized_ticker = ticker.strip().upper()
        company = self.companies.get_by_ticker(db, normalized_ticker)
        if company:
            return company
        try:
            data = self.market_provider.get_company(normalized_ticker)
        except UnknownTickerError:
            raise api_error(404, "UNKNOWN_TICKER", "Company ticker was not found.")
        except ProviderError:
            raise api_error(503, "PROVIDER_UNAVAILABLE", "Company data provider is unavailable.")
        canonical_ticker = data.ticker.strip().upper()
        if not data.name or not canonical_ticker:
            raise api_error(502, "MALFORMED_PROVIDER_RESPONSE", "Company data provider returned invalid data.")
        company = Company(ticker=canonical_ticker, name=data.name.strip(), exchange=data.exchange, country=data.country, sector=data.sector, industry=data.industry)
        self.companies.create(db, company)
        db.commit()
        db.refresh(company)
        return company

    def get_market_data(self, db: Session, ticker: str, from_date: date, to_date: date) -> tuple[Company, list[MarketData]]:
        self._validate_range(from_date, to_date)
        company = self.resolve_company(db, ticker)
        cached = self.market_data.list_for_range(db, company.id, from_date, to_date)
        if cached:
            return company, cached
        try:
            bars = self.market_provider.get_market_data(company.ticker, from_date, to_date)
        except ProviderError as exc:
            raise api_error(503, "PROVIDER_UNAVAILABLE", "Market data provider is unavailable.") from exc
        self._persist_market_data(db, company, bars)
        return company, self.market_data.list_for_range(db, company.id, from_date, to_date)

    async def get_market_data_async(self, db: Session, ticker: str, from_date: date, to_date: date) -> tuple[Company, list[MarketData]]:
        """Fetch the blocking external historical provider without blocking the event loop."""
        self._validate_range(from_date, to_date)
        company = self.resolve_company(db, ticker)
        cached = self.market_data.list_for_range(db, company.id, from_date, to_date)
        if cached:
            return company, cached
        try:
            bars = await asyncio.to_thread(self.market_provider.get_market_data, company.ticker, from_date, to_date)
        except ProviderError as exc:
            raise api_error(503, "PROVIDER_UNAVAILABLE", "Market data provider is unavailable.") from exc
        self._persist_market_data(db, company, bars)
        return company, self.market_data.list_for_range(db, company.id, from_date, to_date)

    def get_news(self, db: Session, ticker: str, from_date: date, to_date: date, limit: int) -> tuple[Company, list[NewsArticle]]:
        self._validate_range(from_date, to_date)
        company = self.resolve_company(db, ticker)
        try:
            articles = self.news_provider.get_news(company.ticker, from_date, to_date)
        except ProviderError:
            raise api_error(503, "PROVIDER_UNAVAILABLE", "News provider is unavailable.")
        self._persist_news(db, company, articles)
        return company, self.news.list_for_range(db, company.id, from_date, to_date, limit)

    def build_company_context(self, db: Session, ticker: str) -> dict:
        company = self.resolve_company(db, ticker)
        to_date = date.today()
        from_date = to_date - timedelta(days=30)
        market = self.market_data.list_for_range(db, company.id, from_date, to_date)
        news = self.news.list_for_range(db, company.id, from_date, to_date, limit=20)
        sources = self.sources.list_for_company(db, company.id)
        return {"company": company, "market": market, "news": news, "sources": sources}

    def _persist_market_data(self, db: Session, company: Company, bars: list) -> None:
        timestamps_by_source: dict[str, list[datetime]] = {}
        for bar in bars:
            timestamps_by_source.setdefault(bar.source, []).append(bar.timestamp)
        existing = {(source, timestamp) for source, timestamps in timestamps_by_source.items() for timestamp in self.market_data.existing_timestamps(db, company.id, timestamps, source)}
        source_urls = [f"market://{bar.source}/{company.ticker}/{bar.timestamp.isoformat()}" for bar in bars]
        existing_sources = self.sources.existing_urls(db, company.id, source_urls)
        for bar in bars:
            if (bar.source, bar.timestamp) not in existing:
                db.add(MarketData(company_id=company.id, timestamp=bar.timestamp, open=bar.open, high=bar.high, low=bar.low, close=bar.close, volume=bar.volume, source=bar.source))
            source_url = f"market://{bar.source}/{company.ticker}/{bar.timestamp.isoformat()}"
            if source_url not in existing_sources:
                db.add(ResearchSource(company_id=company.id, source_type=SourceType.MARKET, source_name=bar.source, source_url=source_url, title=f"{company.ticker} market data", published_at=bar.timestamp, metadata_json={"timestamp": bar.timestamp.isoformat()}))
        db.commit()

    def _persist_news(self, db: Session, company: Company, articles: list[NewsArticleData]) -> None:
        normalized: list[tuple[NewsArticleData, str]] = []
        try:
            normalized = [(article, canonical_url(article.url)) for article in articles]
        except ValueError:
            raise api_error(502, "MALFORMED_PROVIDER_RESPONSE", "News provider returned an invalid article URL.")
        urls = list(dict.fromkeys(url for _, url in normalized))
        existing_articles = self.news.existing_urls(db, urls)
        existing_sources = self.sources.existing_urls(db, company.id, urls)
        for article, url in normalized:
            if url not in existing_articles:
                db.add(NewsArticle(company_id=company.id, title=article.title.strip(), summary=article.summary, url=url, published_at=article.published_at, source_name=article.source_name, author=article.author))
                existing_articles.add(url)
            if url not in existing_sources:
                db.add(ResearchSource(company_id=company.id, source_type=SourceType.NEWS, source_name=article.source_name, source_url=url, title=article.title.strip(), published_at=article.published_at, metadata_json={"author": article.author} if article.author else None))
                existing_sources.add(url)
        db.commit()

    @staticmethod
    def _validate_range(from_date: date, to_date: date) -> None:
        if from_date > to_date:
            raise api_error(422, "INVALID_DATE_RANGE", "'from' date must be on or before 'to' date.")
