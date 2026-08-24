from datetime import date, datetime, time, timezone
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.market_data import MarketData


class MarketDataRepository:
    def list_for_range(self, db: Session, company_id: UUID, from_date: date, to_date: date) -> list[MarketData]:
        start = datetime.combine(from_date, time.min, tzinfo=timezone.utc)
        end = datetime.combine(to_date, time.max, tzinfo=timezone.utc)
        return list(db.scalars(select(MarketData).where(MarketData.company_id == company_id, MarketData.timestamp >= start, MarketData.timestamp <= end).order_by(MarketData.timestamp)))

    def existing_timestamps(self, db: Session, company_id: UUID, timestamps: list[datetime], source: str) -> set[datetime]:
        if not timestamps:
            return set()
        return set(db.scalars(select(MarketData.timestamp).where(MarketData.company_id == company_id, MarketData.source == source, MarketData.timestamp.in_(timestamps))))
