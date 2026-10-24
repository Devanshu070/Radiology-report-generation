"""Bounded retry policy for transient provider and rate-limit failures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import math
from typing import TypeVar


T = TypeVar("T")
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 30.0


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw = headers.get("retry-after", headers.get("Retry-After"))
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, delay) if math.isfinite(delay) else None


def is_retryable(error: Exception) -> bool:
    """Retry network errors, rate limits, and transient server failures only."""
    if isinstance(error, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    if type(error).__name__ in {
        "APIConnectionError", "APITimeoutError", "ConnectError", "NetworkError",
        "PoolTimeout", "ReadTimeout", "TimeoutException", "WriteTimeout",
    }:
        return True
    status = _status_code(error)
    return status == 429 or (status is not None and 500 <= status < 600)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> T:
    """Retry a transient call with bounded exponential backoff and Retry-After."""
    if max_attempts < 1 or base_delay_seconds < 0 or max_delay_seconds < 0:
        raise ValueError("Retry attempts must be positive and delays must be nonnegative.")
    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as error:
            if attempt + 1 >= max_attempts or not is_retryable(error):
                raise
            delay = _retry_after(error)
            if delay is None:
                delay = min(max_delay_seconds, base_delay_seconds * (2 ** attempt))
            else:
                delay = min(max_delay_seconds, delay)
            await sleep(delay)
    raise RuntimeError("Retry loop ended unexpectedly.")
