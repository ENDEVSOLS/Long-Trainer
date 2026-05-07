# Execution-Layer Rate Limiting

LongTrainer 1.3.0 introduces a robust, **two-layer execution rate limiting system** using a token-bucket algorithm. This system prevents noisy-neighbor degradation in multi-tenant deployments and ensures fair resource distribution.

## Two-Layer Enforcement

1. **Layer 1: Tenant Ceiling**: Every tenant has a hard RPM (Requests Per Minute) cap per resource. This prevents a single malicious or runaway tenant from exhausting the global infrastructure (LLM API keys, vector DB limits, etc.).
2. **Layer 2: Bot Equal Share**: Inside a single tenant, the global budget is equally divided among their active bots (`tenant_rpm / active_bots`). This prevents a single noisy bot from starving a tenant's other production bots.

## Configuration

Rate limiting is **disabled by default** to ensure backward compatibility. You can enable it in your `longtrainer.yaml`:

```yaml
rate_limiting:
  enabled: true
  llm_rpm: 60           # Max LLM generation calls per minute
  embedding_rpm: 120    # Max embedding calls per minute
  tool_rpm: 30          # Max tool executions per minute
  ingestion_rpm: 10     # Max document chunking/ingestion ops per minute
```

### In-Code Overrides

You can programmatically override limits for specific tenants or bots using the SDK:

```python
from longtrainer.rate_limiter import RateLimitConfig

config = RateLimitConfig(
    enabled=True,
    llm_rpm=60,
    tenant_overrides={
        "enterprise_tenant_1": {"llm_rpm": 500}
    },
    bot_overrides={
        "critical_bot_A": {"llm_rpm": 100}
    }
)
```

## Error Handling

### API Mode
When using the FastAPI server (`longtrainer serve`), rate-limited requests are caught by a global exception handler. The server returns a `429 Too Many Requests` with a `Retry-After` header indicating exactly when the bucket will have tokens again.

```json
HTTP/1.1 429 Too Many Requests
Retry-After: 15

{
  "detail": "Bot 'bot-123' rate limit exceeded for llm_calls. Retry after 14.5s."
}
```

### CLI Mode
When using the interactive terminal (`longtrainer chat`), rate limits are automatically caught. Instead of crashing, the CLI displays a user-friendly countdown progress bar and auto-retries your message when the budget is restored.
