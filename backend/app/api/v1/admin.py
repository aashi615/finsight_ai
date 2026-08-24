from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import require_role
from app.core.database import get_db
from app.models.user import Role, User
from app.repositories.user_repository import UserRepository
from app.schemas.common import SuccessResponse

router = APIRouter(prefix="/admin", tags=["admin"])
users = UserRepository()


@router.get("/stats", response_model=SuccessResponse[dict])
def stats(db: Session = Depends(get_db), current_user: User = Depends(require_role(Role.ADMIN))):
    return SuccessResponse(data={"total_users": users.count_by_organization(db, current_user.organization_id), "total_reports": 0})
