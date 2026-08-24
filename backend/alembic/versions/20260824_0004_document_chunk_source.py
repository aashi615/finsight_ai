"""preserve source metadata on document chunks

Revision ID: 20260824_0004
Revises: 20260824_0003
Create Date: 2026-08-24
"""

from alembic import op
import sqlalchemy as sa


revision = "20260824_0004"
down_revision = "20260824_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document_chunks", sa.Column("source_url", sa.String(length=2048), nullable=True))


def downgrade() -> None:
    op.drop_column("document_chunks", "source_url")
