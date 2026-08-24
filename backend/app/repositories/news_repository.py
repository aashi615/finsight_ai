from datetime import date, datetime, time, timezone
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.news_article import NewsArticle


class NewsRepository:
    def list_for_range(self, db: Session, company_id: UUID, from_date: date, to_date: date, limit: int) -> list[NewsArticle]:
        start = datetime.combine(from_date, time.min, tzinfo=timezone.utc)
        end = datetime.combine(to_date, time.max, tzinfo=timezone.utc)
        return list(db.scalars(select(NewsArticle).where(NewsArticle.company_id == company_id, NewsArticle.published_at >= start, NewsArticle.published_at <= end).order_by(NewsArticle.published_at.desc()).limit(limit)))

    def existing_urls(self, db: Session, urls: list[str]) -> set[str]:
        if not urls:
            return set()
        return set(db.scalars(select(NewsArticle.url).where(NewsArticle.url.in_(urls))))
