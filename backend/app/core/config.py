from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    environment: Literal["development", "production", "test"] = "development"
    debug: bool = False

    # ------------------------------------------------------------------
    # Database / Auth
    # ------------------------------------------------------------------

    database_url: str

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ------------------------------------------------------------------
    # External APIs
    # ------------------------------------------------------------------

    finnhub_api_key: str | None = None
    finnhub_base_url: str = "https://finnhub.io/api/v1"

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------

    llm_provider: Literal["groq", "openai"] = "groq"

    # Legacy/general model setting.
    # Keep this for backward compatibility with existing code.
    llm_model: str = "openai/gpt-oss-120b"

    # ------------------------------------------------------------------
    # Groq model routing
    # ------------------------------------------------------------------
    #
    # Research agents:
    #   GPT-OSS-20B
    #
    # Final synthesis:
    #   Qwen 3.6 27B
    #
    # Fallback:
    #   Same Qwen model.
    #
    # IMPORTANT:
    # Synthesizer is intentionally NOT GPT-OSS-20B.
    # This prevents all four agents from competing for the same
    # GPT-OSS TPM bucket.
    # ------------------------------------------------------------------

    groq_research_model: str = "openai/gpt-oss-20b"

    groq_final_model: str = "qwen/qwen3.6-27b"

    groq_reasoning_fallback_model: str = "qwen/qwen3.6-27b"

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    # Requests are effectively serialized by GroqProvider's TPM manager.
    # Keep this value for compatibility with the rest of the application.
    max_llm_concurrency: int = 1

    # ------------------------------------------------------------------
    # Per-agent completion budgets
    # ------------------------------------------------------------------
    #
    # These are MAX completion tokens.
    #
    # Groq TPM = prompt tokens + completion tokens.
    #
    # Therefore these values are deliberately bounded instead of
    # blindly allowing large completions.
    # ------------------------------------------------------------------

    news_max_output_tokens: int = 800
    market_max_output_tokens: int = 900
    rag_max_output_tokens: int = 800
    final_max_output_tokens: int = 1000

    # ------------------------------------------------------------------
    # Groq TPM limits
    # ------------------------------------------------------------------
    #
    # Groq free/on-demand limit currently used by the application:
    # 8000 TPM.
    #
    # We reserve only 7000 locally to leave headroom for estimation
    # error and Groq's rolling-window accounting.
    # ------------------------------------------------------------------

    groq_tpm_limit: int = 8000
    groq_safe_tpm_limit: int = 7000

    # ------------------------------------------------------------------
    # Retry policy
    # ------------------------------------------------------------------

    max_llm_retries: int = 2

    # ------------------------------------------------------------------
    # Input budgets
    # ------------------------------------------------------------------

    research_agent_input_token_limit: int = 1800
    final_agent_input_token_limit: int = 1800

    # ------------------------------------------------------------------
    # Research data limits
    # ------------------------------------------------------------------

    news_article_limit: int = 6
    news_article_snippet_chars: int = 280

    market_history_points_limit: int = 12

    rag_top_k: int = 3
    rag_chunk_chars: int = 600

    # ------------------------------------------------------------------
    # API / frontend
    # ------------------------------------------------------------------

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
        ]
    )

    frontend_url: str | None = "http://localhost:5173"

    max_request_body_bytes: int = 1_000_000

    research_rate_limit_per_minute: int = 5

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value):
        """
        Supports legacy local .env values such as:

            DEBUG=release
            DEBUG=production

        without treating them as debug-enabled.
        """
        if isinstance(value, str) and value.lower() in {
            "release",
            "production",
        }:
            return False

        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_postgres_url(cls, value):
        """
        Managed providers commonly expose:

            postgresql://...

        SQLAlchemy uses the installed Psycopg v3 dialect explicitly.
        """
        if isinstance(value, str) and value.startswith("postgresql://"):
            return (
                "postgresql+psycopg://"
                + value.removeprefix("postgresql://")
            )

        return value

    @model_validator(mode="after")
    def validate_production(self):
        if self.environment == "production":

            if (
                self.debug
                or self.jwt_secret_key
                in {
                    "replace-with-a-long-random-secret",
                    "test-secret",
                }
                or len(self.jwt_secret_key) < 32
            ):
                raise ValueError(
                    "Production requires DEBUG=false and a strong JWT_SECRET_KEY."
                )

            if not self.database_url.startswith("postgresql+"):
                raise ValueError(
                    "Production requires a PostgreSQL DATABASE_URL."
                )

            if not self.cors_origins or "*" in self.cors_origins:
                raise ValueError(
                    "Production requires explicit CORS_ORIGINS."
                )

            if self.llm_provider == "groq" and not self.groq_api_key:
                raise ValueError(
                    "Production requires GROQ_API_KEY when "
                    "LLM_PROVIDER=groq."
                )

        return self


settings = Settings()