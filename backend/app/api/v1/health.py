from sqlalchemy import text
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import api_error
from app.schemas.common import SuccessResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=SuccessResponse[dict])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise api_error(503, "DATABASE_UNHEALTHY", "Database is unavailable.")
    return SuccessResponse(data={"status": "healthy", "application": "healthy", "database": "healthy"})
