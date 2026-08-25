import logging
from datetime import date

from app.providers.base import MarketBarData, MarketDataProvider, ProviderError

logger = logging.getLogger(__name__)


class FallbackMarketDataProvider:
    """Uses the primary historical provider first, then Yahoo Finance on failure."""

    def __init__(self, primary: MarketDataProvider, fallback: MarketDataProvider):
        self.primary = primary
        self.fallback = fallback

    def get_company(self, ticker: str):
        return self.primary.get_company(ticker)

    def get_market_data(self, ticker: str, from_date: date, to_date: date) -> list[MarketBarData]:
        try:
            bars = self.primary.get_market_data(ticker, from_date, to_date)
            if not bars:
                raise ProviderError("Primary provider returned no historical market data.")
            self._success(ticker, "finnhub", bars, from_date, to_date)
            return bars
        except ProviderError as primary_error:
            logger.warning(
                "Finnhub historical market data unavailable; falling back to Yahoo Finance",
                extra={"ticker": ticker, "provider": "finnhub", "provider_status_code": primary_error.status_code, "from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
            )
            try:
                bars = self.fallback.get_market_data(ticker, from_date, to_date)
            except ProviderError as fallback_error:
                logger.warning(
                    "Yahoo Finance historical market data unavailable",
                    extra={"ticker": ticker, "provider": "yahoo_finance", "from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
                )
                raise ProviderError("Historical market data is unavailable from Finnhub and Yahoo Finance.") from fallback_error
            if not bars:
                raise ProviderError("Yahoo Finance returned no historical market data.")
            self._success(ticker, "yahoo_finance", bars, from_date, to_date)
            return bars

    @staticmethod
    def _success(ticker: str, provider: str, bars: list[MarketBarData], from_date: date, to_date: date) -> None:
        logger.info(
            f"{provider.replace('_', ' ').title()} historical market data fetched successfully",
            extra={"ticker": ticker, "provider": provider, "row_count": len(bars), "from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
        )
