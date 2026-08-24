import enum
import uuid
from datetime import datetime
from typing import Any
from sqlalchemy import DateTime, Enum, ForeignKey, Index, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class ResearchJobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ResearchJob(Base):
    __tablename__ = "research_jobs"
    __table_args__ = (Index("ix_research_jobs_organization_created", "organization_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    company_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("companies.id", ondelete="SET NULL"))
    status: Mapped[ResearchJobStatus] = mapped_column(Enum(ResearchJobStatus, name="research_job_status"), nullable=False, default=ResearchJobStatus.PENDING)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
