"""Secret-safe asynchronous Groq client with bounded retry and Pydantic output."""

from __future__ import annotations

import json
import os
import re
from itertools import cycle
from threading import Lock
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from ..schemas import GeneratedSections
from ..config import load_project_environment
from .retries import DEFAULT_MAX_ATTEMPTS, retry_async


class ProviderError(RuntimeError):
    """Sanitized provider error suitable for local result logs."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GroqGenerator:
    """Generate validated report sections through Groq without logging prompts."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout_seconds: float = 120.0,
        client_factory: Any | None = None,
        max_structured_retries: int | None = None,
    ):
        load_project_environment()
        configured_structured_retries = (
            max_structured_retries
            if max_structured_retries is not None
            else int(os.environ.get("GROQ_MAX_STRUCTURED_RETRIES", "1"))
        )
        if configured_structured_retries < 0:
            raise ValueError("max_structured_retries cannot be negative.")
        self._max_structured_retries = configured_structured_retries
        self._owns_client = client is None
        self._keys: list[str] = []
        self._clients: list[Any] = []
        self._client_cycle: Any = None
        self._cycle_lock = Lock()
        if client is not None:
            self._clients = [client]
            self._keys = [api_key or ""]
        else:
            candidates = [api_key or os.environ.get("GROQ_API_KEY", "")]
            numbered = sorted(
                (key for key in os.environ if re.fullmatch(r"GROQ_API_KEY_\d+", key)),
                key=lambda name: int(name.rsplit("_", 1)[1]),
            )
            candidates.extend(os.environ[key] for key in numbered)
            self._keys = list(dict.fromkeys(value.strip() for value in candidates if value.strip()))
            if not self._keys:
                raise ValueError("Set GROQ_API_KEY or one or more GROQ_API_KEY_N values in .env or the environment.")
            if os.environ.get("USE_GROQ", "true").strip().lower() not in {"1", "true", "yes", "on"}:
                raise ValueError("USE_GROQ is disabled in the environment.")
            try:
                if client_factory is None:
                    from groq import AsyncGroq
                    client_factory = AsyncGroq
            except ImportError as error:
                raise RuntimeError("Install the base requirements to use Groq generation.") from error
            configured_attempts = int(os.environ.get("GROQ_MAX_HTTP_ATTEMPTS", max_attempts))
            configured_timeout = float(os.environ.get("GROQ_TIMEOUT_SECONDS", timeout_seconds))
            self._max_attempts = max(1, configured_attempts)
            self._clients = [
                client_factory(api_key=key, timeout=configured_timeout, max_retries=0)
                for key in self._keys
            ]
            self._client_cycle = cycle(self._clients)
            return
        self._max_attempts = max_attempts

    async def generate(self, payload: dict[str, Any]) -> GeneratedSections:
        """Send a request, validate its JSON object, and redact any exposed key."""
        with self._cycle_lock:
            client = next(self._client_cycle) if self._client_cycle is not None else self._clients[0]

        request_payload = dict(payload)
        request_payload["messages"] = [dict(message) for message in payload.get("messages", [])]
        repair_count = 0

        def repair_request(previous_content: str | None) -> None:
            messages = [dict(message) for message in payload.get("messages", [])]
            if previous_content:
                messages.append({"role": "assistant", "content": previous_content})
            messages.append({
                "role": "user",
                "content": (
                    "The previous completion failed structured-output validation. "
                    "Return exactly one valid JSON object matching the required schema, "
                    "with non-empty string values for findings and impression. "
                    "Do not include markdown, reasoning, section headers, or extra keys. "
                    "Use only CURRENT_CASE and preserve its template structure."
                ),
            })
            request_payload["messages"] = messages

        def provider_error_code(error: Exception) -> str | None:
            body = getattr(error, "body", None)
            if not isinstance(body, Mapping):
                return None
            detail = body.get("error")
            if not isinstance(detail, Mapping):
                return None
            code = detail.get("code")
            return code if isinstance(code, str) else None

        while True:
            async def call() -> Any:
                return await client.chat.completions.create(**request_payload)

            try:
                completion = await retry_async(call, max_attempts=self._max_attempts)
            except Exception as error:
                code = provider_error_code(error)
                status = getattr(error, "status_code", None)
                if status is None:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                if code in {"json_validate_failed", "json_schema_validation_failed"}:
                    if repair_count < self._max_structured_retries:
                        repair_count += 1
                        repair_request(previous_content=None)
                        continue
                    raise ProviderError(
                        f"Groq structured-output validation failed after {repair_count} repair retries "
                        f"({type(error).__name__}, HTTP {status}; provider message omitted).",
                        status_code=status if isinstance(status, int) else None,
                    ) from error
                status_detail = f", HTTP {status}" if status is not None else ""
                raise ProviderError(
                    f"Groq request failed ({type(error).__name__}{status_detail}; provider message omitted).",
                    status_code=status if isinstance(status, int) else None,
                ) from error

            choices = getattr(completion, "choices", None)
            content = (
                getattr(getattr(choices[0], "message", None), "content", None)
                if choices else None
            )
            try:
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Provider returned an empty response.")
                return GeneratedSections.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                if repair_count >= self._max_structured_retries:
                    raise ProviderError(
                        f"Invalid structured completion after {repair_count} repair retries: "
                        f"{type(error).__name__}"
                    ) from error
                repair_count += 1
                repair_request(content if isinstance(content, str) else None)

    async def aclose(self) -> None:
        """Close an internally created provider client, if supported."""
        if not self._owns_client:
            return
        for client in self._clients:
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result
