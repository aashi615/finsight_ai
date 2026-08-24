"""add research reports and job start timestamps"""
from alembic import op
import sqlalchemy as sa

revision = "20260824_0005"
down_revision = "20260824_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("research_jobs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
    op.create_table("research_reports", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("organization_id", sa.Uuid(), nullable=False), sa.Column("research_job_id", sa.Uuid(), nullable=False), sa.Column("company_id", sa.Uuid(), nullable=True), sa.Column("title", sa.String(length=500), nullable=False), sa.Column("executive_summary", sa.Text(), nullable=False), sa.Column("report_data", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="SET NULL"), sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"), sa.ForeignKeyConstraint(["research_job_id"], ["research_jobs.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("research_job_id"))
    op.create_index("ix_research_reports_organization_created", "research_reports", ["organization_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_research_reports_organization_created", table_name="research_reports")
    op.drop_table("research_reports")
    op.drop_column("research_jobs", "started_at")
