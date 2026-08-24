from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, require_role
from app.core.database import get_db
from app.models.user import Role, User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import UserOut
from app.schemas.common import SuccessResponse
from app.schemas.organization import OrganizationOut, OrganizationUpdate
from app.services.organization_service import OrganizationService

router = APIRouter(prefix="/organization", tags=["organization"])
service = OrganizationService()
users = UserRepository()


@router.get("", response_model=SuccessResponse[OrganizationOut])
def get_organization(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return SuccessResponse(data=service.get_current_organization(db, current_user))


@router.patch("", response_model=SuccessResponse[OrganizationOut])
def update_organization(payload: OrganizationUpdate, db: Session = Depends(get_db), current_user: User = Depends(require_role(Role.ADMIN))):
    return SuccessResponse(data=service.update_current_organization(db, current_user, payload.name), message="Organization updated.")


@router.get("/members", response_model=SuccessResponse[list[UserOut]])
def list_members(db: Session = Depends(get_db), current_user: User = Depends(require_role(Role.ADMIN))):
    return SuccessResponse(data=users.list_by_organization(db, current_user.organization_id))
