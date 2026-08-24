"""add tenant research jobs and rag documents

Revision ID: 20260824_0003
Revises: 20260824_0002
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20260824_0003"
down_revision = "20260824_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("documents", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("organization_id", sa.Uuid(), nullable=False), sa.Column("company_id", sa.Uuid()), sa.Column("title", sa.String(length=500), nullable=False), sa.Column("source_url", sa.String(length=2048)), sa.Column("content", sa.Text(), nullable=False), sa.Column("mime_type", sa.String(length=100), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="SET NULL"), sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_documents_organization_company", "documents", ["organization_id", "company_id"], unique=False)
    op.create_table("document_chunks", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("organization_id", sa.Uuid(), nullable=False), sa.Column("document_id", sa.Uuid(), nullable=False), sa.Column("chunk_index", sa.Integer(), nullable=False), sa.Column("page_number", sa.Integer()), sa.Column("section", sa.String(length=255)), sa.Column("content", sa.Text(), nullable=False), sa.Column("embedding", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"), sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"))
    op.create_index("ix_document_chunks_organization_document", "document_chunks", ["organization_id", "document_id"], unique=False)
    op.create_table("research_jobs", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("organization_id", sa.Uuid(), nullable=False), sa.Column("created_by", sa.Uuid(), nullable=False), sa.Column("company_id", sa.Uuid()), sa.Column("status", sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", name="research_job_status"), nullable=False), sa.Column("question", sa.Text(), nullable=False), sa.Column("result", sa.JSON()), sa.Column("error_message", sa.Text()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.Column("completed_at", sa.DateTime(timezone=True)), sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="SET NULL"), sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="RESTRICT"), sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_research_jobs_organization_created", "research_jobs", ["organization_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_research_jobs_organization_created", table_name="research_jobs")
    op.drop_table("research_jobs")
    op.execute("DROP TYPE research_job_status")
    op.drop_index("ix_document_chunks_organization_document", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_organization_company", table_name="documents")
    op.drop_table("documents")
