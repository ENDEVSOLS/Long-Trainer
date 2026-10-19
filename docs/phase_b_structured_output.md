# Phase B structured-output repair

Implemented on `refactor/structured-output-module`, based on the completed local Phase A branch. The starting checkout was clean. Expected changes were limited to the new helper, `bot.py`, structured/boundary tests, the temporary Ruff exception, changelog, and verification documentation.

The existing structured bot method embedded validation and retry logic; its absent reusable module caused Phase A to skip 17 tests. Extraction preserves the four public dictionary keys (`status`, `data`, `raw_llm_output`, `error`), final failure text and second raw output, markdown-fence tolerance, and arbitrary schema-valid JSON values. Successful responses append only the original query and stringified parsed result to history. Failed responses leave history unchanged. System messages remain first; historical system messages are filtered, and retry feedback is appended to a copied list. There is exactly one retry for invalid JSON/schema output, and no retry for transport or unexpected errors.

`StructuredResponse` is internal and is serialized at the bot boundary. An explicitly supplied LangChain config now reaches both LLM attempts. Logging for extracted code and structured retrieval failures uses the structured module logger with concise validation feedback. Other bot logging is unchanged.

`compute_schema_hash` uses SHA-256 over recursively key-sorted compact JSON. The existing `MongoStorage.save_schema`, `get_current_schema`, and `list_schema_versions` methods have no production callers and are retained unchanged. Hashing remains available for explicit consumers; chat requests do not gain implicit schema persistence, bot-pointer updates, or migration requirements.

Verification used the existing isolated Python 3.12 environment at `/tmp/longtrainer-phase-a-venv`:

- `pytest tests/test_09_phase3.py -q`: **17 passed**, including all four existing vision tests.
- `pytest tests/ -q -ra`: **87 passed, 1 skipped**, two existing LangChain deprecation warnings. The skipped case requires separately provisioned MongoDB integration services.
- `ruff check .`: **all checks passed**. The temporary E731 exception is removed.
- `git diff --check`: passed.

Commands above used the environment's absolute executable paths. The sandboxed full-suite run stalled in FastAPI TestClient and was interrupted; the successful full suite ran outside the sandbox, consistent with Phase A's environment limitation. GitHub Actions and Python 3.10/3.11 were not run locally.

New regressions verify first/retry success, final invalid output, first/retry transport failure, retrieval failure, exact dictionary values, successful-only history updates, message ordering, unchanged first-attempt messages, config forwarding, fenced and array JSON, schema-enabled chat persistence, and API schema forwarding/response wrapping. Existing tests verify deterministic hashes and concise parse/validation errors.

All Phase B local acceptance checks are satisfied. No new dependencies, storage writes, protected runtime changes, or later-phase agent features are introduced. Remote CI, human review, and PR publication remain pending.
