"""add global research data foundation

Revision ID: 20260824_0002
Revises: 20260820_0001
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20260824_0002"
down_revision = "20260820_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("companies", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("ticker", sa.String(length=20), nullable=False), sa.Column("name", sa.String(length=255), nullable=False), sa.Column("exchange", sa.String(length=100)), sa.Column("country", sa.String(length=100)), sa.Column("sector", sa.String(length=100)), sa.Column("industry", sa.String(length=255)), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("ticker"))
    op.create_index("ix_companies_ticker", "companies", ["ticker"], unique=True)
    op.create_table("market_data", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("company_id", sa.Uuid(), nullable=False), sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False), sa.Column("open", sa.Numeric(precision=18, scale=6), nullable=False), sa.Column("high", sa.Numeric(precision=18, scale=6), nullable=False), sa.Column("low", sa.Numeric(precision=18, scale=6), nullable=False), sa.Column("close", sa.Numeric(precision=18, scale=6), nullable=False), sa.Column("volume", sa.BigInteger(), nullable=False), sa.Column("source", sa.String(length=100), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("company_id", "timestamp", "source", name="uq_market_data_company_timestamp_source"))
    op.create_index("ix_market_data_company_timestamp", "market_data", ["company_id", "timestamp"], unique=False)
    op.create_table("news_articles", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("company_id", sa.Uuid(), nullable=False), sa.Column("title", sa.String(length=500), nullable=False), sa.Column("summary", sa.Text()), sa.Column("url", sa.String(length=2048), nullable=False), sa.Column("published_at", sa.DateTime(timezone=True), nullable=False), sa.Column("source_name", sa.String(length=255), nullable=False), sa.Column("author", sa.String(length=255)), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("url", name="uq_news_articles_url"))
    op.create_index("ix_news_articles_company_published_at", "news_articles", ["company_id", "published_at"], unique=False)
    op.create_table("research_sources", sa.Column("id", sa.Uuid(), nullable=False), sa.Column("company_id", sa.Uuid(), nullable=False), sa.Column("source_type", sa.Enum("MARKET", "NEWS", "DOCUMENT", name="source_type"), nullable=False), sa.Column("source_name", sa.String(length=255), nullable=False), sa.Column("source_url", sa.String(length=2048), nullable=False), sa.Column("title", sa.String(length=500)), sa.Column("published_at", sa.DateTime(timezone=True)), sa.Column("metadata", sa.JSON()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False), sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("company_id", "source_url", name="uq_research_sources_company_url"))
    op.create_index("ix_research_sources_company_published_at", "research_sources", ["company_id", "published_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_research_sources_company_published_at", table_name="research_sources")
    op.drop_table("research_sources")
    op.execute("DROP TYPE source_type")
    op.drop_index("ix_news_articles_company_published_at", table_name="news_articles")
    op.drop_table("news_articles")
    op.drop_index("ix_market_data_company_timestamp", table_name="market_data")
    op.drop_table("market_data")
    op.drop_index("ix_companies_ticker", table_name="companies")
    op.drop_table("companies")
