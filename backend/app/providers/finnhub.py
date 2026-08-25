from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import httpx
from app.core.config import settings
from app.providers.base import CompanyData, MarketBarData, NewsArticleData, ProviderError, UnknownTickerError


class FinnhubProvider:
    """Small adapter that keeps Finnhub response formats outside application services."""
    source_name = "finnhub"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key if api_key is not None else settings.finnhub_api_key
        self.base_url = (base_url or settings.finnhub_base_url).rstrip("/")

    def _get(self, path: str, params: dict) -> object:
        if not self.api_key:
            raise ProviderError("Market data provider is not configured.")
        try:
            response = httpx.get(f"{self.base_url}{path}", params={**params, "token": self.api_key}, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("Provider is unavailable or returned malformed data.", status_code=exc.response.status_code) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Provider is unavailable or returned malformed data.") from exc

    def get_company(self, ticker: str) -> CompanyData:
        data = self._get("/stock/profile2", {"symbol": ticker})
        if not isinstance(data, dict) or not data.get("name"):
            raise UnknownTickerError("Ticker was not found.")
        return CompanyData(ticker=ticker, name=str(data["name"]), exchange=data.get("exchange"), country=data.get("country"), industry=data.get("finnhubIndustry"))

    def get_market_data(self, ticker: str, from_date: date, to_date: date) -> list[MarketBarData]:
        start = int(datetime.combine(from_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end = int(datetime.combine(to_date, datetime.max.time(), tzinfo=timezone.utc).timestamp())
        data = self._get("/stock/candle", {"symbol": ticker, "resolution": "D", "from": start, "to": end})
        if not isinstance(data, dict) or data.get("s") == "no_data":
            return []
        required = ("t", "o", "h", "l", "c", "v")
        if any(not isinstance(data.get(key), list) for key in required):
            raise ProviderError("Provider returned malformed market data.")
        rows = zip(data["t"], data["o"], data["h"], data["l"], data["c"], data["v"], strict=True)
        try:
            return [MarketBarData(timestamp=datetime.fromtimestamp(int(t), tz=timezone.utc), open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(l)), close=Decimal(str(c)), volume=int(v), source=self.source_name) for t, o, h, l, c, v in rows]
        except (TypeError, ValueError, InvalidOperation) as exc:
            raise ProviderError("Provider returned malformed market data.") from exc

    def get_news(self, ticker: str, from_date: date, to_date: date) -> list[NewsArticleData]:
        data = self._get("/company-news", {"symbol": ticker, "from": from_date.isoformat(), "to": to_date.isoformat()})
        if not isinstance(data, list):
            raise ProviderError("Provider returned malformed news data.")
        articles: list[NewsArticleData] = []
        try:
            for item in data:
                if not isinstance(item, dict) or not item.get("headline") or not item.get("url") or not item.get("datetime"):
                    raise ValueError
                articles.append(NewsArticleData(title=str(item["headline"]), summary=item.get("summary") or None, url=str(item["url"]), published_at=datetime.fromtimestamp(int(item["datetime"]), tz=timezone.utc), source_name=str(item.get("source") or self.source_name), author=item.get("author") or None))
        except (TypeError, ValueError, OSError) as exc:
            raise ProviderError("Provider returned malformed news data.") from exc
        return articles
