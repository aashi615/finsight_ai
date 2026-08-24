"""initial organizations and users

Revision ID: 20260820_0001
Revises:
Create Date: 2026-08-20
"""
from alembic import op
import sqlalchemy as sa

revision = "20260820_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("organizations", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("name", sa.String(length=255), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.PrimaryKeyConstraint("id"))
    op.create_table("users", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("organization_id", sa.Uuid(), nullable=False), sa.Column("name", sa.String(length=255), nullable=False), sa.Column("email", sa.String(length=320), nullable=False), sa.Column("password_hash", sa.String(length=255), nullable=False), sa.Column("role", sa.Enum("ADMIN", "ANALYST", name="user_role"), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="RESTRICT"), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("email"))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_organization_id", "users", ["organization_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_users_organization_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_table("organizations")
    op.execute("DROP TYPE user_role")
