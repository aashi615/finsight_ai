from datetime import date, timedelta
from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.providers.finnhub import FinnhubProvider
from app.providers.fallback_market import FallbackMarketDataProvider
from app.providers.yahoo_finance import YahooFinanceProvider
from app.schemas.common import SuccessResponse
from app.schemas.company import CompanyContextOut, CompanyOut, MarketDataOut, NewsArticleOut
from app.services.research_service import ResearchService

router = APIRouter(prefix="/companies", tags=["companies"])


def get_research_service() -> ResearchService:
    finnhub = FinnhubProvider()
    return ResearchService(market_provider=FallbackMarketDataProvider(finnhub, YahooFinanceProvider()), news_provider=finnhub)


def resolve_dates(from_date: date | None, to_date: date | None) -> tuple[date, date]:
    end = to_date or date.today()
    return from_date or end - timedelta(days=30), end


@router.get("/{ticker}", response_model=SuccessResponse[CompanyOut])
def get_company(ticker: str = Path(min_length=1, max_length=20), db: Session = Depends(get_db), _: User = Depends(get_current_user), service: ResearchService = Depends(get_research_service)):
    return SuccessResponse(data=service.resolve_company(db, ticker))


@router.get("/{ticker}/market", response_model=SuccessResponse[list[MarketDataOut]])
def get_market(ticker: str = Path(min_length=1, max_length=20), from_date: date | None = Query(default=None, alias="from"), to_date: date | None = Query(default=None, alias="to"), db: Session = Depends(get_db), _: User = Depends(get_current_user), service: ResearchService = Depends(get_research_service)):
    start, end = resolve_dates(from_date, to_date)
    _, market = service.get_market_data(db, ticker, start, end)
    return SuccessResponse(data=market)


@router.get("/{ticker}/news", response_model=SuccessResponse[list[NewsArticleOut]])
def get_news(ticker: str = Path(min_length=1, max_length=20), limit: int = Query(default=20, ge=1, le=100), from_date: date | None = Query(default=None, alias="from"), to_date: date | None = Query(default=None, alias="to"), db: Session = Depends(get_db), _: User = Depends(get_current_user), service: ResearchService = Depends(get_research_service)):
    start, end = resolve_dates(from_date, to_date)
    _, news = service.get_news(db, ticker, start, end, limit)
    return SuccessResponse(data=news)


@router.get("/{ticker}/context", response_model=SuccessResponse[CompanyContextOut])
def get_context(ticker: str = Path(min_length=1, max_length=20), db: Session = Depends(get_db), _: User = Depends(get_current_user), service: ResearchService = Depends(get_research_service)):
    return SuccessResponse(data=service.build_company_context(db, ticker))
