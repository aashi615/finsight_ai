from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math

import yfinance as yf

from app.providers.base import MarketBarData, ProviderError


class YahooFinanceProvider:
    """Adapter that normalizes Yahoo daily OHLCV history into MarketBarData."""

    source_name = "yahoo_finance"

    def get_market_data(self, ticker: str, from_date: date, to_date: date) -> list[MarketBarData]:
        if from_date > to_date:
            raise ProviderError("Yahoo Finance received an invalid historical date range.")
        # Yahoo's end boundary is exclusive. Never ask it for a future day.
        effective_end = min(to_date, date.today())
        if from_date > effective_end:
            raise ProviderError("Yahoo Finance historical date range is in the future.")
        try:
            history = yf.Ticker(ticker).history(
                start=from_date,
                end=effective_end + timedelta(days=1),
                interval="1d",
                auto_adjust=False,
            )
        except Exception as exc:
            raise ProviderError("Yahoo Finance historical market data is unavailable.") from exc
        if history is None or history.empty:
            raise ProviderError("Yahoo Finance returned no historical market data.")

        bars: list[MarketBarData] = []
        try:
            for index, row in history.iterrows():
                timestamp = self._timestamp(index)
                open_, high, low, close = (self._decimal(row[field]) for field in ("Open", "High", "Low", "Close"))
                volume = self._volume(row["Volume"])
                bars.append(MarketBarData(timestamp=timestamp, open=open_, high=high, low=low, close=close, volume=volume, source=self.source_name))
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise ProviderError("Yahoo Finance returned malformed historical market data.") from exc

        unique = {bar.timestamp: bar for bar in bars}
        normalized = sorted(unique.values(), key=lambda bar: bar.timestamp)
        if not normalized:
            raise ProviderError("Yahoo Finance returned no valid historical market data.")
        return normalized

    @staticmethod
    def _timestamp(value) -> datetime:
        converted = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
        if not isinstance(converted, datetime):
            raise ValueError("invalid market date")
        # Daily bars represent a trading date; persist each at midnight UTC consistently.
        return datetime.combine(converted.date(), time.min, tzinfo=timezone.utc)

    @staticmethod
    def _decimal(value) -> Decimal:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("invalid OHLC value")
        return Decimal(str(value))

    @staticmethod
    def _volume(value) -> int:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("invalid volume")
        return int(number)
