from uuid import UUID
from pydantic import BaseModel, EmailStr, Field
from app.models.user import Role
from app.schemas.organization import OrganizationOut


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    organization_name: str = Field(min_length=2, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    role: Role
    model_config = {"from_attributes": True}


class CurrentUserOut(UserOut):
    organization: OrganizationOut


class AuthOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: CurrentUserOut
