import asyncio
import json
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from groq import Groq

from app.core.config import settings
from app.llm.base import LLMProviderError
from app.llm.local_embeddings import LocalEmbeddingProvider

logger = logging.getLogger(__name__)


class StructuredOutputParseError(ValueError):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


class TokenBudgetManager:
    """Thread-safe rolling 60-second TPM reservation manager."""

    def __init__(
        self,
        safe_limit: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.safe_limit = safe_limit
        self._clock = clock
        self._reservations: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def available(self) -> int:
        with self._lock:
            self._prune()
            used = sum(tokens for _, tokens in self._reservations)
            return max(0, self.safe_limit - used)

    def reserve_or_delay(self, tokens: int) -> float:
        """
        Reserve capacity if available.

        Returns:
            0.0 -> reservation succeeded
            >0  -> caller must sleep before retrying
        """
        tokens = max(1, int(tokens))

        if tokens > self.safe_limit:
            return -1.0

        with self._lock:
            self._prune()

            used = sum(value for _, value in self._reservations)

            if used + tokens <= self.safe_limit:
                self._reservations.append((self._clock(), tokens))
                return 0.0

            if not self._reservations:
                return 60.0

            oldest_timestamp = self._reservations[0][0]
            remaining = 60.0 - (self._clock() - oldest_timestamp)
            return max(0.05, remaining)

    def release_latest(self, reserved_tokens: int | None = None) -> None:
        """
        Release the most recent reservation.

        Requests are serialized by GroqProvider's semaphore, so the latest
        reservation belongs to the current request.
        """
        with self._lock:
            self._prune()

            if not self._reservations:
                return

            if reserved_tokens is None:
                self._reservations.pop()
                return

            timestamp, tokens = self._reservations[-1]

            if tokens == reserved_tokens:
                self._reservations.pop()

    def reconcile_latest(
        self,
        reserved_tokens: int,
        actual_tokens: int | None,
    ) -> None:
        """
        Replace the pessimistic reservation with actual Groq usage.
        """
        if actual_tokens is None:
            return

        actual_tokens = max(1, int(actual_tokens))

        with self._lock:
            self._prune()

            if not self._reservations:
                return

            timestamp, tokens = self._reservations[-1]

            if tokens == reserved_tokens:
                self._reservations[-1] = (
                    timestamp,
                    actual_tokens,
                )

    def _prune(self) -> None:
        now = self._clock()

        while (
            self._reservations
            and now - self._reservations[0][0] >= 60
        ):
            self._reservations.popleft()


def normalize_json_response(content: str) -> str:
    """Extract the first balanced JSON object/array."""
    if not isinstance(content, str):
        raise StructuredOutputParseError("response_content")

    text = content.strip()

    if not text:
        raise StructuredOutputParseError("empty_response")

    for start, character in enumerate(text):
        if character not in "{[":
            continue

        end = _balanced_json_end(text, start)

        if end is not None:
            return text[start:end]

    raise StructuredOutputParseError("json_extraction")


def _balanced_json_end(text: str, start: int) -> int | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    stack = [closing]
    in_string = False
    escaped = False

    for index in range(start + 1, len(text)):
        character = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False

            continue

        if character == '"':
            in_string = True

        elif character == "{":
            stack.append("}")

        elif character == "[":
            stack.append("]")

        elif character in "}]":
            if not stack or character != stack[-1]:
                return None

            stack.pop()

            if not stack:
                return index + 1

    return None


def extract_json_object(text: str) -> str:
    """Backward-compatible alias."""
    return normalize_json_response(text)


class GroqProvider:
    """
    Groq provider with:

    - serialized requests
    - conservative TPM reservations
    - strict pre-request TPM gate
    - retry-after handling
    - retry reservation cleanup
    - Qwen synthesizer routing
    - Qwen fallback for research agents
    - GPT-OSS low reasoning
    - JSON parsing and validation
    """

    _request_limiter = threading.BoundedSemaphore(1)

    _budget_manager = TokenBudgetManager(
        settings.groq_safe_tpm_limit
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        token_budget: TokenBudgetManager | None = None,
    ):
        self.api_key = (
            api_key
            if api_key is not None
            else settings.groq_api_key
        )

        self.client = client or (
            Groq(
                api_key=self.api_key,
                max_retries=0,
                timeout=30.0,
            )
            if self.api_key
            else None
        )

        self._sleep = sleep
        self._random_value = random_value

        self._token_budget = (
            token_budget or self._budget_manager
        )

        self._embeddings = LocalEmbeddingProvider()

    async def complete_json(
        self,
        prompt: str,
        payload: dict,
        *,
        agent: str,
        max_output_tokens: int,
        force_fallback: bool = False,
    ) -> dict:
        return await asyncio.to_thread(
            self._complete_json,
            prompt,
            payload,
            agent,
            max_output_tokens,
            force_fallback,
        )

    def _complete_json(
        self,
        prompt: str,
        payload: dict,
        agent: str,
        max_output_tokens: int,
        force_fallback: bool = False,
    ) -> dict:

        if not self.client:
            raise LLMProviderError(
                "Groq provider is not configured.",
                category="llm_provider_error",
            )

        primary_model = self._model_for(agent)

        max_output_tokens = self._enforce_output_budget(
            agent,
            max_output_tokens,
        )

        input_limit = (
            settings.final_agent_input_token_limit
            if agent == "research_synthesizer"
            else settings.research_agent_input_token_limit
        )

        with self._request_limiter:

            transport_attempt = 0
            incomplete_retried = False
            fallback_used = (
                force_fallback
                and agent != "research_synthesizer"
            )

            active_model = (
                settings.groq_reasoning_fallback_model
                if fallback_used
                else primary_model
            )

            if active_model.startswith("qwen/"):
                self._verify_model_available(active_model)

            active_prompt = prompt
            force_compact_input = False

            while True:

                request_id = None
                response_details: dict[str, Any] = {}

                choice = None
                message = None
                content = None
                usage = None

                reserved_tokens: int | None = None

                try:

                    messages, estimated_input_tokens, input_compacted = (
                        self._messages(
                            active_prompt,
                            payload,
                            input_limit,
                            force_compact=force_compact_input,
                        )
                    )

                    estimated_total_tokens = (
                        estimated_input_tokens
                        + max_output_tokens
                    )

                    # ==================================================
                    # HARD TPM GATE
                    # ==================================================

                    reserved_tokens = self._wait_for_tpm(
                        agent=agent,
                        model=active_model,
                        estimated_input=estimated_input_tokens,
                        requested_output=max_output_tokens,
                        estimated_total=estimated_total_tokens,
                        retry_number=transport_attempt,
                    )

                    request = {
                        "model": active_model,
                        "messages": messages,
                        "max_completion_tokens": max_output_tokens,
                        "temperature": 0.2,
                        "response_format": {
                            "type": "json_object"
                        },
                        **self._reasoning_options(active_model),
                    }

                    logger.info(
                        "llm_request_scheduled",
                        extra={
                            "provider": "groq",
                            "agent": agent,
                            "model": active_model,
                            "primary_or_fallback": (
                                "fallback"
                                if fallback_used
                                else "primary"
                            ),
                            "fallback_used": fallback_used,
                            "estimated_input_tokens": estimated_input_tokens,
                            "requested_max_output_tokens": max_output_tokens,
                            "estimated_total_tokens": estimated_total_tokens,
                            "reserved_tokens": reserved_tokens,
                            "available_tpm_after_reservation": (
                                self._token_budget.available()
                            ),
                            "input_compacted": input_compacted,
                            "retry_number": transport_attempt,
                        },
                    )

                    # ==================================================
                    # GROQ REQUEST
                    # ==================================================

                    response = (
                        self.client.chat.completions.create(
                            **request
                        )
                    )

                    request_id = getattr(
                        response,
                        "_request_id",
                        None,
                    )

                    choice = response.choices[0]
                    message = choice.message

                    content = getattr(
                        message,
                        "content",
                        None,
                    )

                    usage = getattr(
                        response,
                        "usage",
                        None,
                    )

                    actual_total_tokens = getattr(
                        usage,
                        "total_tokens",
                        None,
                    )

                    # ==================================================
                    # RECONCILE TPM
                    # ==================================================

                    self._token_budget.reconcile_latest(
                        reserved_tokens,
                        actual_total_tokens,
                    )

                    reserved_tokens = None

                    response_details = self._response_details(
                        content,
                        choice,
                        message,
                        request_id,
                        usage,
                    )

                    logger.info(
                        "groq_response_received",
                        extra={
                            "provider": "groq",
                            "agent": agent,
                            "model": active_model,
                            "configured_output_limit": max_output_tokens,
                            **response_details,
                        },
                    )

                    # ==================================================
                    # COMPLETION LIMIT
                    # ==================================================

                    if getattr(
                        choice,
                        "finish_reason",
                        None,
                    ) == "length":

                        empty_visible_content = (
                            not isinstance(content, str)
                            or not content.strip()
                        )

                        event = (
                            "groq_reasoning_budget_exhausted"
                            if empty_visible_content
                            else "groq_output_budget_exhausted"
                        )

                        logger.warning(
                            event,
                            extra={
                                "provider": "groq",
                                "agent": agent,
                                "model": active_model,
                                "attempt": transport_attempt,
                                "configured_output_limit": max_output_tokens,
                                **self._response_details(
                                    content,
                                    choice,
                                    message,
                                    request_id,
                                    usage,
                                    include_preview=True,
                                ),
                            },
                        )

                        if not incomplete_retried:

                            incomplete_retried = True
                            force_compact_input = True

                            active_prompt = (
                                f"{prompt}\n\n"
                                "Return the smallest valid JSON object now. "
                                "Use only required fields, empty arrays "
                                "when there are no supported claims, "
                                "and no prose outside JSON."
                            )

                            if (
                                empty_visible_content
                                and self._activate_fallback(
                                    agent,
                                    primary_model,
                                    active_model,
                                    fallback_used,
                                    "empty_visible_content_at_completion_limit",
                                )
                            ):
                                active_model = (
                                    settings.groq_reasoning_fallback_model
                                )
                                fallback_used = True
                                transport_attempt = 0

                                self._verify_model_available(
                                    active_model
                                )

                            continue

                        category = (
                            "llm_reasoning_budget_exhausted"
                            if empty_visible_content
                            else "llm_incomplete_response"
                        )

                        raise LLMProviderError(
                            (
                                "Groq exhausted the completion budget "
                                "before returning visible JSON."
                                if empty_visible_content
                                else
                                "Groq response was incomplete because "
                                "it reached the output token limit."
                            ),
                            category=category,
                        )

                    # ==================================================
                    # JSON PARSING
                    # ==================================================

                    parsed = self._parse_json_response(
                        content,
                        agent,
                        request_id,
                        response_details,
                        active_model,
                    )

                    logger.info(
                        "llm_usage",
                        extra={
                            "provider": "groq",
                            "agent": agent,
                            "model": active_model,
                            "estimated_input_tokens": estimated_input_tokens,
                            "requested_max_output_tokens": max_output_tokens,
                            "estimated_total_tokens": estimated_total_tokens,
                            "actual_prompt_tokens": getattr(
                                usage,
                                "prompt_tokens",
                                estimated_input_tokens,
                            ),
                            "actual_completion_tokens": getattr(
                                usage,
                                "completion_tokens",
                                None,
                            ),
                            "actual_total_tokens": getattr(
                                usage,
                                "total_tokens",
                                None,
                            ),
                            "finish_reason": getattr(
                                choice,
                                "finish_reason",
                                None,
                            ),
                            "visible_content_exists": bool(
                                isinstance(content, str)
                                and content.strip()
                            ),
                            "reasoning_content_exists": (
                                self._reasoning_content(message)
                                is not None
                            ),
                            "request_id": request_id,
                        },
                    )

                    return parsed

                # ======================================================
                # JSON / RESPONSE ERRORS
                # ======================================================

                except (
                    StructuredOutputParseError,
                    TypeError,
                    KeyError,
                    IndexError,
                ) as exc:

                    if reserved_tokens is not None:
                        self._token_budget.release_latest(
                            reserved_tokens
                        )
                        reserved_tokens = None

                    stage = getattr(
                        exc,
                        "stage",
                        "response_content",
                    )

                    event = {
                        "empty_response": "groq_empty_response",
                        "json_extraction": (
                            "groq_json_extraction_failed"
                        ),
                        "json_decode": "groq_json_decode_failed",
                    }.get(
                        stage,
                        "groq_json_parse_failed",
                    )

                    diagnostics = (
                        self._response_details(
                            content,
                            choice,
                            message,
                            request_id,
                            usage,
                            include_preview=True,
                        )
                        if choice is not None
                        else response_details
                    )

                    logger.warning(
                        event,
                        extra={
                            "provider": "groq",
                            "agent": agent,
                            "model": active_model,
                            "request_id": request_id,
                            "response_parsing_stage": stage,
                            **diagnostics,
                        },
                    )

                    if (
                        not incomplete_retried
                        and not fallback_used
                    ):
                        incomplete_retried = True
                        force_compact_input = True

                        active_prompt = (
                            f"{prompt}\n\n"
                            "Return the smallest valid JSON object now. "
                            "Use only required fields and no prose outside JSON."
                        )

                        continue

                    if self._activate_fallback(
                        agent,
                        primary_model,
                        active_model,
                        fallback_used,
                        "invalid_json",
                    ):
                        active_model = (
                            settings.groq_reasoning_fallback_model
                        )
                        fallback_used = True
                        transport_attempt = 0
                        force_compact_input = True

                        self._verify_model_available(
                            active_model
                        )

                        continue

                    raise LLMProviderError(
                        f"Groq returned malformed JSON output at {stage}.",
                        category="llm_invalid_response",
                    ) from exc

                # ======================================================
                # PROVIDER / RATE LIMIT ERRORS
                # ======================================================

                except LLMProviderError:
                    raise

                except Exception as exc:

                    # IMPORTANT:
                    # Failed requests must release their reservation.
                    if reserved_tokens is not None:
                        self._token_budget.release_latest(
                            reserved_tokens
                        )
                        reserved_tokens = None

                    (
                        category,
                        retryable,
                        retry_after,
                    ) = self._classify_error(exc)

                    self._log_error(
                        agent,
                        active_model,
                        transport_attempt + 1,
                        category,
                        exc,
                    )

                    # --------------------------------------------------
                    # Retry current model
                    # --------------------------------------------------

                    if (
                        retryable
                        and transport_attempt
                        < settings.max_llm_retries
                    ):
                        delay = self._retry_delay(
                            transport_attempt,
                            retry_after,
                        )

                        logger.warning(
                            "llm_retry",
                            extra={
                                "provider": "groq",
                                "agent": agent,
                                "model": active_model,
                                "attempt": transport_attempt + 1,
                                "retry_delay_seconds": round(
                                    delay,
                                    3,
                                ),
                                "error_category": category,
                            },
                        )

                        self._sleep(delay)

                        transport_attempt += 1

                        # Re-enter TPM gate before retry.
                        continue

                    # --------------------------------------------------
                    # Primary exhausted -> Qwen fallback
                    # --------------------------------------------------

                    if self._activate_fallback(
                        agent,
                        primary_model,
                        active_model,
                        fallback_used,
                        category,
                    ):
                        active_model = (
                            settings.groq_reasoning_fallback_model
                        )

                        fallback_used = True
                        transport_attempt = 0
                        incomplete_retried = False
                        force_compact_input = True

                        active_prompt = (
                            f"{prompt}\n\n"
                            "Return the smallest valid JSON object now. "
                            "Use only required fields and no prose outside JSON."
                        )

                        self._verify_model_available(
                            active_model
                        )

                        logger.warning(
                            "llm_model_fallback",
                            extra={
                                "provider": "groq",
                                "agent": agent,
                                "from_model": primary_model,
                                "to_model": active_model,
                                "reason": category,
                            },
                        )

                        continue

                    # --------------------------------------------------
                    # Final failure
                    # --------------------------------------------------

                    message_text = (
                        "Groq quota is exhausted."
                        if category == "llm_quota_exhausted"
                        else
                        "Groq rejected the requested JSON output."
                        if category == "llm_invalid_response"
                        else
                        "Groq provider is temporarily unavailable."
                    )

                    raise LLMProviderError(
                        message_text,
                        category=category,
                    ) from exc

    @staticmethod
    def _model_for(agent: str) -> str:
        if agent == "research_synthesizer":
            return settings.groq_final_model

        return settings.groq_research_model

    @staticmethod
    def _activate_fallback(
        agent: str,
        primary_model: str,
        active_model: str,
        fallback_used: bool,
        reason: str,
    ) -> bool:

        eligible = (
            agent != "research_synthesizer"
            and not fallback_used
            and active_model == primary_model
        )

        if eligible:
            logger.warning(
                "llm_fallback_triggered",
                extra={
                    "provider": "groq",
                    "agent": agent,
                    "primary_model": primary_model,
                    "fallback_model": (
                        settings.groq_reasoning_fallback_model
                    ),
                    "reason": reason,
                    "fallback_used": True,
                },
            )

        return eligible

    def _verify_model_available(
        self,
        model: str,
    ) -> None:
        """Verify that the configured Groq model is available."""

        models = getattr(
            self.client,
            "models",
            None,
        )

        if models is None:
            return

        try:
            models.retrieve(model)

        except Exception as exc:

            status = getattr(
                exc,
                "status_code",
                None,
            )

            if status in {403, 404}:
                raise LLMProviderError(
                    f"Configured Groq model '{model}' "
                    "is unavailable or not permitted for this project.",
                    category="llm_configuration_error",
                ) from exc

            raise LLMProviderError(
                f"Could not verify configured Groq model "
                f"'{model}' availability.",
                category="llm_configuration_error",
            ) from exc

    @staticmethod
    def _reasoning_options(
        model: str,
    ) -> dict[str, Any]:

        if model.startswith("openai/gpt-oss-"):
            return {
                "reasoning_effort": "low",
                "include_reasoning": False,
            }

        if model.startswith("qwen/"):
            return {
                "reasoning_effort": "none",
                "include_reasoning": False,
            }

        return {}

    @staticmethod
    def _enforce_output_budget(
        agent: str,
        requested: int,
    ) -> int:

        limits = {
            "news_analyst": settings.news_max_output_tokens,
            "market_analyst": settings.market_max_output_tokens,
            "document_rag_agent": settings.rag_max_output_tokens,
            "research_synthesizer": settings.final_max_output_tokens,
        }

        if agent not in limits:
            raise LLMProviderError(
                f"Unknown Groq agent budget: {agent}",
                category="llm_provider_error",
            )

        enforced = min(
            requested,
            limits[agent],
        )

        if requested != enforced:
            logger.warning(
                "llm_output_budget_clamped",
                extra={
                    "provider": "groq",
                    "agent": agent,
                    "requested_max_output_tokens": requested,
                    "enforced_max_output_tokens": enforced,
                },
            )

        return enforced

    def _wait_for_tpm(
        self,
        *,
        agent: str,
        model: str,
        estimated_input: int,
        requested_output: int,
        estimated_total: int,
        retry_number: int,
    ) -> int:

        safe_limit = settings.groq_safe_tpm_limit

        if estimated_total > safe_limit:
            raise LLMProviderError(
                "LLM request exceeds the safe TPM budget "
                "before it can be sent.",
                category="llm_provider_error",
            )

        while True:

            delay = self._token_budget.reserve_or_delay(
                estimated_total
            )

            # Impossible request.
            if delay < 0:
                raise LLMProviderError(
                    "LLM request exceeds the configured TPM limit.",
                    category="llm_provider_error",
                )

            # Capacity reserved.
            if delay == 0:
                return estimated_total

            available = self._token_budget.available()

            logger.warning(
                "llm_request_delayed",
                extra={
                    "provider": "groq",
                    "agent": agent,
                    "model": model,
                    "estimated_input_tokens": estimated_input,
                    "requested_max_output_tokens": requested_output,
                    "estimated_tokens": estimated_total,
                    "available_tpm": available,
                    "delay_seconds": round(delay, 3),
                    "reason": "tpm_budget",
                    "retry_number": retry_number,
                },
            )

            self._sleep(delay)

    @staticmethod
    def _messages(
        prompt: str,
        payload: dict,
        input_limit: int,
        *,
        force_compact: bool = False,
    ) -> tuple[
        list[dict[str, str]],
        int,
        bool,
    ]:

        payload_json = json.dumps(
            payload,
            default=str,
        )

        content = (
            f"{prompt}\n\n"
            f"Input data:\n{payload_json}"
        )

        estimated_tokens = _estimate_tokens(content)

        if (
            estimated_tokens <= input_limit
            and not force_compact
        ):
            return (
                [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                estimated_tokens,
                False,
            )

        compact_payload = _compact_payload(
            payload,
            aggressive=force_compact,
        )

        compact_json = json.dumps(
            compact_payload,
            default=str,
        )

        compact_content = (
            f"{prompt}\n\n"
            f"Input data:\n{compact_json}"
        )

        return (
            [
                {
                    "role": "user",
                    "content": compact_content,
                }
            ],
            _estimate_tokens(compact_content),
            True,
        )

    @staticmethod
    def _parse_json_response(
        content: Any,
        agent: str,
        request_id: str | None = None,
        response_details: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict:

        extracted = normalize_json_response(content)

        logger.info(
            "groq_json_parse_attempt",
            extra={
                "provider": "groq",
                "agent": agent,
                "model": (
                    model
                    or GroqProvider._model_for(agent)
                ),
                "request_id": request_id,
                "response_parsing_stage": (
                    "before_json_loads"
                ),
                **(response_details or {}),
            },
        )

        try:
            parsed = json.loads(extracted)

        except json.JSONDecodeError as exc:
            raise StructuredOutputParseError(
                "json_decode"
            ) from exc

        if not isinstance(parsed, dict):
            raise StructuredOutputParseError(
                "json_object"
            )

        logger.info(
            "llm_response_parsed",
            extra={
                "provider": "groq",
                "agent": agent,
                "model": (
                    model
                    or GroqProvider._model_for(agent)
                ),
                "request_id": request_id,
                "response_parsing_stage": "json_loaded",
            },
        )

        return parsed

    @staticmethod
    def _response_details(
        content: Any,
        choice: Any,
        message: Any,
        request_id: str | None,
        usage: Any,
        *,
        include_preview: bool = False,
    ) -> dict[str, Any]:

        preview = (
            content[:1000]
            if include_preview
            and isinstance(content, str)
            else None
        )

        return {
            "status": 200,
            "request_id": request_id,
            "finish_reason": getattr(
                choice,
                "finish_reason",
                None,
            ),
            "actual_prompt_tokens": getattr(
                usage,
                "prompt_tokens",
                None,
            ),
            "actual_completion_tokens": getattr(
                usage,
                "completion_tokens",
                None,
            ),
            "actual_total_tokens": getattr(
                usage,
                "total_tokens",
                None,
            ),
            "output_tokens": getattr(
                usage,
                "completion_tokens",
                None,
            ),
            "content_type": (
                type(content).__name__
                if content is not None
                else None
            ),
            "content_length": (
                len(content)
                if isinstance(content, str)
                else None
            ),
            "content_preview": preview,
            "content_empty": (
                not bool(content and content.strip())
                if isinstance(content, str)
                else content is None
            ),
            "reasoning_content_present": (
                GroqProvider._reasoning_content(message)
                is not None
            ),
            "content_has_code_fence": (
                "```" in content
                if isinstance(content, str)
                else False
            ),
            "tool_calls_present": bool(
                getattr(
                    message,
                    "tool_calls",
                    None,
                )
            ),
            "response_format": "json_object",
            "reasoning_format": None,
        }

    @staticmethod
    def _reasoning_content(
        message: Any,
    ) -> Any:

        return (
            getattr(message, "reasoning", None)
            or getattr(
                message,
                "reasoning_content",
                None,
            )
        )

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._embeddings.embed(texts)

    def _classify_error(
        self,
        exc: Exception,
    ) -> tuple[str, bool, str | None]:

        status = getattr(
            exc,
            "status_code",
            None,
        )

        body = getattr(
            exc,
            "body",
            None,
        )

        error_text = str(
            body or exc
        ).lower()

        headers = getattr(
            getattr(
                exc,
                "response",
                None,
            ),
            "headers",
            {},
        ) or {}

        error = (
            body.get("error", body)
            if isinstance(body, dict)
            else {}
        )

        retry_after = (
            headers.get("retry-after")
            or (
                str(error.get("retry_after"))
                if (
                    isinstance(error, dict)
                    and error.get("retry_after")
                    is not None
                )
                else None
            )
        )

        if (
            "insufficient_quota" in error_text
            or "quota" in error_text
            or "billing" in error_text
        ):
            return (
                "llm_quota_exhausted",
                False,
                retry_after,
            )

        if (
            "json_validate_failed" in error_text
            or "generated json does not match"
            in error_text
        ):
            return (
                "llm_invalid_response",
                False,
                retry_after,
            )

        if status == 429:
            return (
                "llm_rate_limit",
                True,
                retry_after,
            )

        if (
            isinstance(status, int)
            and 500 <= status <= 599
        ):
            return (
                "llm_provider_error",
                True,
                retry_after,
            )

        if exc.__class__.__name__ in {
            "APITimeoutError",
            "APIConnectionError",
        }:
            return (
                "llm_timeout",
                True,
                retry_after,
            )

        return (
            "llm_provider_error",
            False,
            retry_after,
        )

    def _retry_delay(
        self,
        attempt: int,
        retry_after: str | None,
    ) -> float:

        try:
            if (
                retry_after is not None
                and float(retry_after) >= 0
            ):
                # Small safety margin against timing races.
                return float(retry_after) + 0.25

        except (ValueError, TypeError):
            pass

        return (
            min(30.0, 2**attempt)
            + self._random_value()
        )

    def _log_error(
        self,
        agent: str,
        model: str,
        attempt: int,
        category: str,
        exc: Exception,
    ) -> None:

        response = getattr(
            exc,
            "response",
            None,
        )

        headers = getattr(
            response,
            "headers",
            {},
        ) or {}

        body = getattr(
            exc,
            "body",
            None,
        )

        error = (
            body.get("error", body)
            if isinstance(body, dict)
            else {}
        )

        logger.warning(
            "llm_request_failed",
            extra={
                "provider": "groq",
                "agent": agent,
                "model": model,
                "attempt": attempt,
                "status": getattr(
                    exc,
                    "status_code",
                    None,
                ),
                "request_id": headers.get(
                    "x-request-id"
                ),
                "groq_error_type": (
                    error.get("type")
                    if isinstance(error, dict)
                    else type(exc).__name__
                ),
                "groq_error_code": (
                    error.get("code")
                    if isinstance(error, dict)
                    else None
                ),
                "groq_error_message": (
                    error.get("message")
                    if isinstance(error, dict)
                    else str(exc)
                ),
                "error_category": category,
            },
        )


def _estimate_tokens(text: str) -> int:
    """
    Conservative approximation.

    We intentionally add a safety margin because the previous
    production logs showed actual prompt tokens substantially
    exceeding the raw len(text)/4 estimate.
    """
    base = max(
        1,
        (len(text) + 3) // 4,
    )

    return int(base * 1.25) + 8


def _compact_text(
    text: str,
    limit: int = 280,
) -> str:

    if len(text) <= limit:
        return text

    cut = text.rfind(
        " ",
        0,
        limit,
    )

    return text[
        : cut if cut > 0 else limit
    ].rstrip()


def _compact_payload(
    value: Any,
    *,
    aggressive: bool = False,
) -> Any:

    if isinstance(value, str):
        return _compact_text(
            value,
            120 if aggressive else 280,
        )

    if isinstance(value, list):
        return [
            _compact_payload(
                item,
                aggressive=aggressive,
            )
            for item in value[
                :3 if aggressive else 6
            ]
        ]

    if isinstance(value, dict):
        return {
            key: _compact_payload(
                item,
                aggressive=aggressive,
            )
            for key, item in value.items()
        }

    return value