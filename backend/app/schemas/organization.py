from uuid import UUID
from pydantic import BaseModel, Field


class OrganizationOut(BaseModel):
    id: UUID
    name: str
    model_config = {"from_attributes": True}


class OrganizationUpdate(BaseModel):
    name: str = Field(min_length=2, max_length=255)
