"""Per-tenant, two-layer execution-layer rate limiting for LongTrainer.

Provides:
- ``TokenBucket``: Thread-safe token-bucket with configurable rate/capacity.
- ``RateLimitConfig``: Pydantic model for rate limiting configuration.
- ``TenantRateLimiter``: Two-layer limiter (tenant ceiling + per-bot budget).
- ``LongTrainerRateLimitError``: Exception raised when a budget is exhausted.
- ``_current_tenant_id``: ``contextvars.ContextVar`` for tenant propagation.
"""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Context variable for tenant propagation ──────────────────────────────────

_current_tenant_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "longtrainer_tenant_id", default="default"
)


# ── Exception ────────────────────────────────────────────────────────────────

class LongTrainerRateLimitError(Exception):
    """Raised when a per-tenant or per-bot rate limit is exhausted.

    Attributes:
        retry_after: Seconds until the next token becomes available.
        resource: The resource type that was exhausted (e.g. ``llm_calls``).
        key: The rate-limit key that triggered the error (e.g. ``tenant:abc``).
    """

    def __init__(
        self,
        message: str,
        retry_after: float = 60.0,
        resource: str = "",
        key: str = "",
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.resource = resource
        self.key = key


# ── Configuration ────────────────────────────────────────────────────────────

class RateLimitConfig(BaseModel):
    """Per-resource rate limit configuration.

    Attributes:
        enabled: Master switch — when ``False``, no rate limiting is applied.
        llm_rpm: LLM calls per minute (tenant ceiling).
        embedding_rpm: Embedding calls per minute (tenant ceiling).
        tool_rpm: Tool executions per minute (tenant ceiling).
        ingestion_rpm: Document ingestion ops per minute (tenant ceiling).
        tenant_overrides: Per-tenant RPM overrides.
        bot_overrides: Per-bot RPM overrides.
    """

    enabled: bool = False
    llm_rpm: int = 60
    embedding_rpm: int = 120
    tool_rpm: int = 30
    ingestion_rpm: int = 10
    tenant_overrides: dict[str, dict[str, int]] = Field(default_factory=dict)
    bot_overrides: dict[str, dict[str, int]] = Field(default_factory=dict)

    @field_validator("llm_rpm", "embedding_rpm", "tool_rpm", "ingestion_rpm")
    @classmethod
    def _positive_rpm(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RPM values must be positive integers.")
        return v


# ── Token Bucket ─────────────────────────────────────────────────────────────

class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Args:
        rate: Tokens added per second (``rpm / 60``).
        capacity: Maximum burst size (== RPM).

    Raises:
        ValueError: If ``capacity`` is zero or negative.
    """

    def __init__(self, rate: float, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Token bucket capacity must be positive, got {capacity}.")
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> tuple[bool, float]:
        """Try to consume *tokens* from the bucket.

        Returns:
            ``(True, 0.0)`` if allowed, or
            ``(False, retry_after_seconds)`` if denied.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity, self._tokens + elapsed * self._rate
            )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True, 0.0

            deficit = tokens - self._tokens
            retry_after = deficit / self._rate if self._rate > 0 else 60.0
            return False, retry_after

    @property
    def tokens(self) -> float:
        """Current token count (approximate — not under lock)."""
        return self._tokens

    def update(self, rate: float, capacity: int) -> None:
        """Dynamically update the bucket's rate and capacity."""
        if capacity <= 0:
            raise ValueError(f"Token bucket capacity must be positive, got {capacity}.")
        with self._lock:
            # First, update tokens based on old rate
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity, self._tokens + elapsed * self._rate
            )
            self._last_refill = now
            
            # Now apply new rate and capacity
            self._rate = rate
            self._capacity = capacity
            self._tokens = min(self._tokens, self._capacity)


# ── Resource-name → config-field mapping ─────────────────────────────────────

_RESOURCE_FIELD_MAP: dict[str, str] = {
    "llm_calls": "llm_rpm",
    "embedding_calls": "embedding_rpm",
    "tool_calls": "tool_rpm",
    "ingestion_ops": "ingestion_rpm",
}

_DEFAULT_RPM = 60  # Fallback for unknown resource types


# ── Tenant Rate Limiter ──────────────────────────────────────────────────────

class TenantRateLimiter:
    """Two-layer rate limiter: tenant global ceiling + per-bot equal-share budget.

    Layer 1 – **Tenant ceiling**: Every tenant has a hard RPM cap per resource.
    Layer 2 – **Bot budget**: Each bot under a tenant gets an equal share
    (``tenant_rpm // num_bots``). Per-bot overrides take precedence.

    When either layer is exhausted, ``LongTrainerRateLimitError`` is raised.

    Args:
        config: A ``RateLimitConfig`` instance.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        # {f"tenant:{tid}:{resource}" -> TokenBucket}
        self._tenant_buckets: dict[str, TokenBucket] = {}
        # {f"bot:{bid}:{resource}" -> TokenBucket}
        self._bot_buckets: dict[str, TokenBucket] = {}
        # {tenant_id -> set(bot_id)}
        self._tenant_bots: dict[str, set[str]] = {}
        # Lock for bucket/registry creation
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def check_and_consume(
        self, resource: str, bot_id: str, tokens: int = 1
    ) -> tuple[bool, float]:
        """Two-layer rate limit check.

        Args:
            resource: Resource type (``llm_calls``, ``tool_calls``, etc.).
            bot_id: The bot's unique identifier.
            tokens: Number of tokens to consume.

        Returns:
            ``(True, 0.0)`` if allowed.

        Raises:
            LongTrainerRateLimitError: If either layer is exhausted.
        """
        if not self._config.enabled:
            return True, 0.0

        try:
            tenant_id = _current_tenant_id.get()

            # Track bot → tenant mapping
            with self._lock:
                if tenant_id not in self._tenant_bots:
                    self._tenant_bots[tenant_id] = set()
                self._tenant_bots[tenant_id].add(bot_id)

            # Layer 1: Tenant ceiling (skip for 'default' tenant in SDK mode)
            if tenant_id != "default":
                tenant_rpm = self._get_tenant_rpm(resource, tenant_id)
                tenant_bucket = self._get_bucket(
                    f"tenant:{tenant_id}:{resource}", tenant_rpm
                )
                allowed, retry_after = tenant_bucket.consume(tokens)
                if not allowed:
                    raise LongTrainerRateLimitError(
                        f"Tenant '{tenant_id}' rate limit exceeded for {resource}. "
                        f"Retry after {retry_after:.1f}s.",
                        retry_after=retry_after,
                        resource=resource,
                        key=f"tenant:{tenant_id}",
                    )

            # Layer 2: Per-bot budget
            bot_rpm = self._get_bot_rpm(resource, bot_id, tenant_id)
            bot_bucket = self._get_bot_bucket(
                f"bot:{bot_id}:{resource}", bot_rpm
            )
            allowed, retry_after = bot_bucket.consume(tokens)
            if not allowed:
                raise LongTrainerRateLimitError(
                    f"Bot '{bot_id}' rate limit exceeded for {resource}. "
                    f"Retry after {retry_after:.1f}s.",
                    retry_after=retry_after,
                    resource=resource,
                    key=f"bot:{bot_id}",
                )

            return True, 0.0

        except LongTrainerRateLimitError:
            raise  # Re-raise rate limit errors
        except Exception as e:
            # Fail-open: rate limiting should never block legitimate traffic
            import logging
            logging.getLogger(__name__).warning(
                "Rate limiter check failed (allowing request): %s", e
            )
            return True, 0.0

    def get_usage(self, bot_id: str) -> dict:
        """Return current usage stats for a bot.

        Returns:
            Dict mapping resource names to remaining tokens.
        """
        usage: dict[str, dict] = {}
        for resource in _RESOURCE_FIELD_MAP:
            bot_key = f"bot:{bot_id}:{resource}"
            bucket = self._bot_buckets.get(bot_key)
            usage[resource] = {
                "remaining": int(bucket.tokens) if bucket else -1,
            }
        return usage

    def reset(self, key: str) -> None:
        """Reset all buckets matching a key prefix.

        Args:
            key: Prefix to match (e.g. ``"tenant:abc"`` or ``"bot:bot-123"``).
        """
        with self._lock:
            for k in list(self._tenant_buckets.keys()):
                if k.startswith(key):
                    del self._tenant_buckets[k]
            for k in list(self._bot_buckets.keys()):
                if k.startswith(key):
                    del self._bot_buckets[k]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_tenant_rpm(self, resource: str, tenant_id: str) -> int:
        """Get effective RPM for a tenant + resource, including overrides."""
        field = _RESOURCE_FIELD_MAP.get(resource)
        if not field:
            return _DEFAULT_RPM

        # Check tenant overrides first
        overrides = self._config.tenant_overrides.get(tenant_id, {})
        if field in overrides:
            return overrides[field]

        return getattr(self._config, field, _DEFAULT_RPM)

    def _get_bot_rpm(self, resource: str, bot_id: str, tenant_id: str) -> int:
        """Get effective RPM for a bot, considering overrides and equal share."""
        field = _RESOURCE_FIELD_MAP.get(resource)
        if not field:
            return _DEFAULT_RPM

        # Check per-bot overrides first
        overrides = self._config.bot_overrides.get(bot_id, {})
        if field in overrides:
            return overrides[field]

        # Equal share: tenant_rpm / num_bots
        tenant_rpm = self._get_tenant_rpm(resource, tenant_id)
        num_bots = len(self._tenant_bots.get(tenant_id, set())) or 1
        return max(1, tenant_rpm // num_bots)

    def _get_bucket(self, key: str, rpm: int) -> TokenBucket:
        """Get or create a tenant-level TokenBucket."""
        with self._lock:
            if key not in self._tenant_buckets:
                rate = rpm / 60.0
                self._tenant_buckets[key] = TokenBucket(rate=rate, capacity=rpm)
            else:
                self._tenant_buckets[key].update(rate=rpm / 60.0, capacity=rpm)
        return self._tenant_buckets[key]

    def _get_bot_bucket(self, key: str, rpm: int) -> TokenBucket:
        """Get or create a bot-level TokenBucket."""
        with self._lock:
            if key not in self._bot_buckets:
                rate = rpm / 60.0
                self._bot_buckets[key] = TokenBucket(rate=rate, capacity=rpm)
            else:
                self._bot_buckets[key].update(rate=rpm / 60.0, capacity=rpm)
        return self._bot_buckets[key]
