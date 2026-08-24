from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.auth import AuthOut, CurrentUserOut, LoginRequest, SignupRequest
from app.schemas.common import SuccessResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
service = AuthService()


def current_user_out(user: User) -> CurrentUserOut:
    return CurrentUserOut(id=user.id, name=user.name, email=user.email, role=user.role, organization=user.organization)


@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=SuccessResponse[AuthOut])
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    user = service.signup(db, payload)
    return SuccessResponse(data=AuthOut(access_token=service.token_for(user), user=current_user_out(user)), message="Account created.")


@router.post("/login", response_model=SuccessResponse[AuthOut])
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = service.login(db, payload)
    return SuccessResponse(data=AuthOut(access_token=service.token_for(user), user=current_user_out(user)), message="Login successful.")


@router.get("/me", response_model=SuccessResponse[CurrentUserOut])
def me(current_user: User = Depends(get_current_user)):
    return SuccessResponse(data=current_user_out(current_user))


@router.post("/logout", response_model=SuccessResponse[dict])
def logout(current_user: User = Depends(get_current_user)):
    return SuccessResponse(data={}, message="Logged out. Discard the access token; it remains valid until expiration.")
