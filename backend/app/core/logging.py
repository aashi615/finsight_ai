import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        for key in ("method", "path", "status", "duration_ms", "job_id", "company", "agent", "success", "error_type", "error_category", "error_detail", "cause_type", "provider", "provider_status_code", "model", "attempt", "retry_delay_seconds", "input_tokens", "output_tokens", "total_tokens", "request_id", "response_parsing_stage", "validation_failure_fields", "rate_limit_remaining_tokens", "groq_error_type", "groq_error_code", "groq_error_message", "api_key_configured", "openai_error_type", "openai_error_code", "openai_error_message", "rate_limit_limit_requests", "rate_limit_remaining_requests", "rate_limit_reset_requests", "rate_limit_limit_tokens", "rate_limit_reset_tokens", "retry_after"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
