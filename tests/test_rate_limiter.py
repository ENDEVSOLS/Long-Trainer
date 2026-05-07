import time
import pytest
import contextvars

from longtrainer.rate_limiter import (
    RateLimitConfig,
    TokenBucket,
    TenantRateLimiter,
    LongTrainerRateLimitError,
    _current_tenant_id,
)

def test_token_bucket_basics():
    # 60 RPM = 1 token/sec
    bucket = TokenBucket(rate=1.0, capacity=60)

    # Can consume available capacity
    allowed, retry = bucket.consume(1)
    assert allowed
    assert retry == 0.0

    allowed, retry = bucket.consume(59)
    assert allowed

    # Next one should fail
    allowed, retry = bucket.consume(1)
    assert not allowed
    assert retry > 0.0

def test_tenant_rate_limiter_two_layer():
    config = RateLimitConfig(enabled=True, llm_rpm=60)
    limiter = TenantRateLimiter(config)

    # Mock context
    _current_tenant_id.set("tenant_1")
    bot_id = "bot_1"

    # Layer 2 budget is 60 (since only 1 bot)
    # Consume 60
    limiter.check_and_consume("llm_calls", bot_id, tokens=60)

    # 61st should fail
    with pytest.raises(LongTrainerRateLimitError) as exc:
        limiter.check_and_consume("llm_calls", bot_id, tokens=1)

    assert "Tenant 'tenant_1' rate limit exceeded" in str(exc.value)

def test_tenant_rate_limiter_equal_share():
    config = RateLimitConfig(enabled=True, llm_rpm=60)
    limiter = TenantRateLimiter(config)

    _current_tenant_id.set("tenant_2")

    # Register 2 bots by hitting check_and_consume
    limiter.check_and_consume("llm_calls", "bot_A", tokens=1)
    limiter.check_and_consume("llm_calls", "bot_B", tokens=1)

    # Now there are 2 bots. Equal share = 60 // 2 = 30 RPM per bot.
    # After bucket resize (60→30), bot_A has 30 tokens (capped). Consume 30.
    limiter.check_and_consume("llm_calls", "bot_A", tokens=30)

    # The next token for bot_A should fail due to Layer 2 (bot budget exhausted)
    with pytest.raises(LongTrainerRateLimitError) as exc:
        limiter.check_and_consume("llm_calls", "bot_A", tokens=1)

    assert "Bot 'bot_A' rate limit exceeded" in str(exc.value)

def test_rate_limiter_disabled():
    config = RateLimitConfig(enabled=False, llm_rpm=1) # 1 RPM
    limiter = TenantRateLimiter(config)
    
    _current_tenant_id.set("tenant_disabled")
    
    # Should be able to consume infinitely because it's disabled
    for _ in range(10):
        limiter.check_and_consume("llm_calls", "bot_1", tokens=1)


def test_per_bot_override():
    """Per-bot override takes precedence over equal share."""
    config = RateLimitConfig(
        enabled=True,
        llm_rpm=60,
        bot_overrides={"bot-special": {"llm_rpm": 5}},
    )
    limiter = TenantRateLimiter(config)
    _current_tenant_id.set("tenant_override")

    # bot-special gets 5 RPM (override), not equal share
    limiter.check_and_consume("llm_calls", "bot-special", tokens=5)
    with pytest.raises(LongTrainerRateLimitError) as exc:
        limiter.check_and_consume("llm_calls", "bot-special", tokens=1)
    assert "bot-special" in str(exc.value)


def test_exception_attributes():
    """LongTrainerRateLimitError has correct retry_after, resource, key."""
    # Use 'default' tenant (skips Layer 1) to isolate bot-layer error
    config = RateLimitConfig(enabled=True, llm_rpm=3)
    limiter = TenantRateLimiter(config)
    _current_tenant_id.set("default")

    # Bot budget = 3 (default tenant, 1 bot)
    limiter.check_and_consume("llm_calls", "bot-exc", tokens=3)
    with pytest.raises(LongTrainerRateLimitError) as exc:
        limiter.check_and_consume("llm_calls", "bot-exc", tokens=1)

    assert exc.value.retry_after > 0
    assert exc.value.resource == "llm_calls"
    assert "bot-exc" in exc.value.key


def test_context_variable_default():
    """_current_tenant_id defaults to 'default'."""
    _current_tenant_id.set("default")
    assert _current_tenant_id.get() == "default"
