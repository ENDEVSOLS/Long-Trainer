# Phase C implementation and verification

> Historical verification record: Python 3.10 results and reproduction commands below predate the minimum-version change on 7 October 2026. Current installation requires Python 3.11+, and CI tests Python 3.11/3.12.

Phase C1 and C2 add four named purposes on branch `feat/agent-types`. Work started with the uncommitted Phase B changes intact. No commit, push, or remote PR was created.

Before implementation, inspection confirmed that legacy tool lists accumulate per-bot tools, global tools are merged separately for new and lazy chats, and internal document/chat-training rebuilds call legacy `create_bot()` without configuration. The installed community loader supports `wikipedia` and `arxiv`; Tavily, Python REPL and Yahoo Finance News require the existing factories. Yahoo Finance News defers its yfinance import until execution, so the new adapter validates that dependency during construction.

## Implemented decisions

- The Pydantic `AgentTypeConfig` and separate registry hold purpose, prompt, immutable tool identifiers and agent/RAG mode. Four built-ins are registered; applications can register custom configurations.
- `agent_type` is appended to `create_bot()`; invalid names raise outside legacy error swallowing. Required tool/runtime failures also raise clearly for named bots.
- A supplied type determines mode. Explicit prompts replace type prompts. `tools=None` chooses defaults; explicit lists, including empty lists, replace defaults. Named bots exclude implicit global and pre-existing tools.
- Bot records save optional type metadata, effective prompt/mode and string identifiers through `update_bot`, without storage signature changes, credentials or tool-object serialization. Explicit empty lists survive reload. Legacy records load without migration.
- New and lazy text chats use matching configuration. Internal updates, chat training and CLI ingestion retain named selections. Named rebuilds clear cached chains for history replay. Direct no-type calls keep legacy behavior.
- Customer support remains RAG. A small purpose-specific retriever annotates copied documents with source identifiers. The public trainer rejects web/upload augmentation on sync and async text chat, and rejects vision responses for support. No protected retrieval, vision or chat implementation was edited. Grounding/escalation is instructed by the prompt; model compliance is not independently enforced.
- Research, coding and financial extras declare their tool dependencies. Optional tools are loaded only when requested. Coding/financial Python executes in-process. Financial coverage is news and calculations, not a price feed.
- SQL is absent and explicitly deferred until its database policy, dialect, credentials/reconnection and runtime design receive review. Phase D API/CLI/YAML selection is not implemented.

The user guide is [agent_types.md](agent_types.md). Changes are limited to `agent_types/`, `trainer.py`, the package export, one CLI ingestion call, dependency extras, offline tests and documentation/changelog. Phase B's existing working-tree changes remain present.

## Validation

The full offline suite runs on the existing Python 3.12.14 development environment using `/tmp/longtrainer-phase-a-venv/bin/pytest tests/ -q -ra`, outside the sandbox because FastAPI TestClient stalls inside it. **Final result: 116 passed, 1 skipped, 8 deprecation warnings in 36.26 seconds.** The skip is the separately provisioned MongoDB integration test. All 29 new Phase C cases passed.

The 29 new cases cover registry configuration/lookup, four default definitions, explicit tools/prompts/empty overrides, global exclusion, persistence/reload, unknown/missing dependencies, new/lazy agent construction, real LangGraph responses with a local fake model, support RAG and augmentation policy, source annotation, updates/chat training/CLI ingestion, and rate-limiter/tracer forwarding. No external APIs or paid models are used. A subprocess blocks LangGraph, LangChain Experimental, Tavily, arXiv and yfinance imports and verifies package import plus a real local plain-RAG response.

`ruff check .` and `git diff --check` pass. Existing deprecation warnings remain, including LangGraph's `create_react_agent` deprecation; runtime migration is outside this phase.

## Completed follow-up verification — 5 October 2026

All four requested local verification gates now pass:

| Gate | Observed result |
| --- | --- |
| Fresh base installation with network | New Python 3.12.14 environment; 195 packages installed, dependency consistency check passed. Package import and a local plain-RAG response passed with all new optional imports blocked. |
| Temporary MongoDB integration | MongoDB 7 container on loopback port 27019; retained MongoDB/in-memory Qdrant test: **1 passed, 116 deselected** in 5.36s. |
| Named support persistence integration | Real MongoDB plus persisted FAISS: type, explicit empty tools and custom prompt restored; lazy chat history replay, source annotations and internal rebuild passed. Fake model/embeddings; temporary directory cleaned. |
| Python 3.10.21 full suite | **116 passed, 1 skipped**, 9 warnings in 40.87s. The skipped MongoDB case passed separately above. |
| Python 3.11.16 full suite | **116 passed, 1 skipped**, 8 warnings in 43.39s. The skipped MongoDB case passed separately above. |
| Research extra in its own fresh environment | 197 packages installed; real Tavily, Wikipedia and arXiv tools constructed; local LangGraph response passed. |
| Coding extra in its own fresh environment | 197 packages installed; real Python REPL and Tavily tools constructed; local LangGraph response passed. |
| Financial extra in its own fresh environment | 201 packages installed; real Yahoo Finance News and Python REPL tools constructed; local LangGraph response passed. |
| Customer support in the fresh base environment | Plain-RAG response using the shipped support prompt passed; no purpose extra required. |
| Dependency consistency | `uv pip check` passed in the base, all three individual purpose environments, and both Python test environments. |
| Final lint and patch integrity | `ruff check .` and `git diff --check` passed. |

The base and individual purpose environments have no inherited system site packages. A combined-purpose development environment was also installed successfully, but the individual environments above are the evidence for each extra's completeness.

Python 3.10/3.11 resolved LangChain 1.4.3, Core 1.6.6, Community 0.4.2, LangGraph 1.2.12 and pytest 9.1.1. Their Unstructured versions were respectively 0.18.32 and 0.27.10. Both received the existing CI parser prerequisite `en_core_web_sm` 3.8.0. No implementation fixes were needed. The initial Python 3.10 download timed out and passed on retry with reduced download concurrency.

### Executed commands and repeatable checks

Commands ran from `Long-Trainer/` except the standalone smoke checks, which ran from `/tmp` to avoid relying on the repository working directory. All uv commands used `UV_CACHE_DIR=/tmp/longtrainer-uv-cache`; downloaded Python runtimes used `UV_PYTHON_INSTALL_DIR=/tmp/longtrainer-python`.

```bash
uv python install 3.10 3.11
uv venv --python /tmp/longtrainer-phase-a-venv/bin/python /tmp/longtrainer-c-base-fresh
uv pip install --python /tmp/longtrainer-c-base-fresh/bin/python -e . \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# For each purpose: research, coding, financial (substitute the actual name).
uv venv --python /tmp/longtrainer-phase-a-venv/bin/python /tmp/longtrainer-c-extra-PURPOSE
uv pip install --python /tmp/longtrainer-c-extra-PURPOSE/bin/python -e '.[PURPOSE]' \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# Separate fresh environments for 3.10 and 3.11.
uv venv --python 3.10 /tmp/longtrainer-c-py310
uv venv --python 3.11 /tmp/longtrainer-c-py311
# Run for each interpreter; substitute its absolute path.
uv pip install --python PYTHON -e '.[agent,dev,cli,api]' \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python PYTHON \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
/tmp/longtrainer-c-py310/bin/pytest tests/ -q -ra
/tmp/longtrainer-c-py311/bin/pytest tests/ -q -ra

# Run against the dedicated temporary container, then remove that container.
docker run --detach --name longtrainer-phase-c-verification-mongo \
  --publish 127.0.0.1:27019:27017 docker.io/library/mongo:7
LONGTRAINER_TEST_MONGO_URI='mongodb://127.0.0.1:27019/?serverSelectionTimeoutMS=5000' \
  /tmp/longtrainer-phase-a-venv/bin/pytest tests/ -q -ra -m integration --run-integration
```

Every interpreter path above was executed. `uv pip check --python PYTHON` was executed for all six individual validation environments. CPU PyTorch wheels avoid GPU downloads without changing project dependencies. Network downloads and FastAPI/real MongoDB test runs used execution outside the filesystem sandbox.

Reusable constructor/base and named-Mongo checks are retained as `testing-folder/verify_phase_c_installation.py` and `testing-folder/verify_phase_c_mongo.py`. Their original versions were executed from `/tmp/longtrainer-c-install-smoke.py` and `/tmp/longtrainer-c-mongo-support.py`; the retained versions wrap execution in `main()` and allow the Mongo URI environment variable. Example invocation:

```bash
/tmp/longtrainer-c-base-fresh/bin/python testing-folder/verify_phase_c_installation.py base
/tmp/longtrainer-c-base-fresh/bin/python testing-folder/verify_phase_c_installation.py customer_support
/tmp/longtrainer-c-extra-research/bin/python testing-folder/verify_phase_c_installation.py research
/tmp/longtrainer-c-extra-coding/bin/python testing-folder/verify_phase_c_installation.py coding
/tmp/longtrainer-c-extra-financial/bin/python testing-folder/verify_phase_c_installation.py financial
```

Installation, suite, integration and smoke logs remain under `/tmp/longtrainer-c-*.log`; temporary logs/environments are local artifacts, not permanent release evidence. The created MongoDB container is removed after verification. Application data from the named-bot smoke check is deleted by its cleanup block.

This completes the four requested local verification gates and Phase C's local test/install/lint acceptance checks. Tool checks construct real tool classes using a placeholder Tavily key and verify an agent response through a local fake model; they do not make live search, finance or paid model requests. Prompt adherence and external service behavior are therefore not asserted. SQL remains deferred. Remote CI, human review and release publication remain separate pending steps.
