# Changelog

All notable changes to this project will be documented in this file.

## [1.3.1] — 2026-05-07

### ✨ New Features
- **Tenant Rate Limiting**: Two-layer execution rate limiting system using a token-bucket algorithm. Layer 1 enforces per-tenant RPM ceilings, Layer 2 enforces equal-share budgets across bots within a tenant. Fully configurable via `longtrainer.yaml` or `RateLimitConfig`.
- **Per-Bot Overrides**: Fine-grained rate limit overrides for specific bots via `bot_overrides` configuration.
- **CLI Rate Limit Recovery**: The `longtrainer chat` and `longtrainer add-doc` commands now auto-retry with a progress bar when rate limits are hit.
- **API 429 Responses**: FastAPI server returns proper `429 Too Many Requests` with `Retry-After` headers when rate limits are exceeded.

### 🔧 Improvements
- **Interactive Demo**: Added `demos/longtrainer_demo.py` — a complete RAG workflow demo with live Rich progress tracking. Fully env-var driven (supports OpenAI, Gemini, Ollama, any vector store) with zero hardcoded config.
- **CLI Cleanup**: Reverted CLI from Rich TUI panels to clean `click.echo/secho` output. Removed `rich>=13.0` from core dependencies.
- **Removed VHS/TUI assets**: Cleaned up `.tape` files, `Makefile`, and demo GIFs from the repository.

### 📖 Documentation
- New `rate_limiting.md` guide covering two-layer enforcement, YAML configuration, API error handling, and CLI auto-retry.
- Updated README with embedded demo screenshot and streamlined Quick Start section.

## [1.3.0] — 2026-05-01

### ✨ New Features
- **LongTracer Integration**: Native, optional support for LongTracer hallucination detection and observability.
  - Automatically captures spans, latencies, and token counts for LLM calls and retriever queries.
  - Provides claim verification via `CitationVerifier` using hybrid STS and NLI for RAG, Vision, and Structured bot responses.
  - Opt-in via `enable_tracer=True` during `LongTrainer` initialization.
  - Available via the `[tracer]` extra (`pip install longtrainer[tracer]`).

## [1.2.3] — 2026-04-29

### 🚀 Performance & Scalability
- **Lazy Loading Chat Histories:** Significantly optimized startup time and RAM consumption. `load_bot()` no longer eagerly loads all chat histories into RAM. Instead, chat histories are lazy-loaded on-demand from MongoDB via new `_ensure_chat_loaded()` and `_ensure_vision_chat_loaded()` methods.
- **Qdrant Initialization Fix:** Fixed a bug in `vectorstores.py` where passing `:memory:` to Qdrant would fail due to improper URL parsing. In-memory fallback is now properly handled via the `location` parameter.

## [1.2.2] — 2026-04-11

### 🐛 Bug Fixes & Perf Improvements
- **Async Document Ingestion:** Implemented `aadd_document_from_path` and `aadd_document_from_link` with parallel ingestion via `ThreadPoolExecutor` and `asyncio.gather()`. Batch document ingestion is now non-blocking and significantly faster.
- **Retrieval Confidence Scores:** Added `invoke_vectorstore_with_scores()` which returns the raw FAISS L2 distance and automatically injects a normalized `retrieval_score` (1.0 = perfect match) into document metadata.
- **KeyError Bug:** Fixed a crash during `update_chatbot()` caused by an obsolete `faiss_path` dictionary reference.
- **Documentation:** Added a new comprehensive implementation roadmap `/testing-folder/LONGTRAINER_ROADMAP.md` covering Phase 3 through Phase 10.


## [1.1.0] — 2026-02-21

### ✨ New Features
- **Zero-Code CLI**: Introduced `longtrainer` command-line interface. Use `longtrainer init` for interactive project scaffolding (generates `longtrainer.yaml`) and `longtrainer serve` to start the REST API.
- **FastAPI REST Server**: Shipped a comprehensive built-in API server (`longtrainer.api:app`) with 16 endpoints covering bot management, document ingestion, sync/streaming chat, vision chat, and vector search.
- **Lazy Initialization**: Trainer instance now defers LLM connection initialization until the first API call, allowing the server to start successfully (`/health` check passes) without an `OPENAI_API_KEY`.
- **API Documentation**: Auto-generated Swagger UI available at `/docs` when serving the API.

### 📦 Dependencies
- Added `click>=8.0` and `pyyaml>=6.0` to core dependencies for CLI and configuration management.
- Added `[cli]` optional dependency group (`click`, `pyyaml`).
- Automatically installs `fastapi` and `uvicorn` when using the `[api]` extra.

## [1.0.1] — 2026-02-21

### Improved

- **PyPI SEO**: Added 20 search keywords and expanded classifiers (3 → 14) for better discoverability
- **PyPI description**: Now includes key terms (LangChain, FAISS, MongoDB, tool calling, agent mode)
- **PyPI sidebar**: Added Bug Tracker and Changelog URLs

### Added

- **README badges**: GitHub Stars, CI status, Python versions, Open Collective sponsors count
- **Sponsor section**: "Support the Project 💖" with Open Collective donate button
- **FUNDING.yml**: Enables the 💖 Sponsor button on GitHub repo
- **Sponsor nav link**: Quick access to sponsorship from README header

## [1.0.0] — 2026-02-18

### ⚠️ Breaking Changes

- **Removed** `ConversationalRetrievalChain` — replaced with LCEL-based `RAGBot` and LangGraph-based `AgentBot`
- **Removed** `setup.py` and `requirements.txt` — migrated to `pyproject.toml` (UV/pip compatible)
- **Removed** `langchain.memory.ConversationTokenBufferMemory` — replaced with `InMemoryChatMessageHistory`
- **Removed** `EnsembleRetriever` / `MultiQueryRetriever` from deprecated `langchain_classic` — replaced with custom `MultiQueryEnsembleRetriever`
- **Changed** `get_response()` return type — now returns `(answer, sources)` tuple instead of `(answer, sources, web_sources)`

### ✨ New Features

- **Dual Mode Architecture**: RAG mode (default, LCEL chain) + Agent mode (LangGraph, tool calling)
- **Streaming Responses**: `get_response(stream=True)` yields tokens, `aget_response()` for async streaming
- **Custom Tool Calling**: `add_tool()`, `remove_tool()`, `list_tools()` — register any `@tool` decorated function
- **Built-in Tools**: `web_search` (DuckDuckGo) and `document_reader` (multi-format text extraction)
- **Per-Bot Configuration**: Custom LLM, embeddings, retriever config, and prompt per bot via `create_bot()`
- **Tool Registry**: `ToolRegistry` class for managing tools globally or per-bot

### 🔧 Improvements

- All imports updated to latest LangChain 2026 standards (`langchain_core`, `langchain_community`, `langchain_text_splitters`)
- Full type hints across all modules
- Comprehensive docstrings on all public methods
- `pyproject.toml` with `hatchling` build system
- `langgraph` is an **optional dependency** — `pip install longtrainer[agent]`
- Optional `[api]` and `[dev]` dependency groups

### 🧪 Testing & CI

- 4 offline test suites (28 checks): imports, loaders, tool registry, bot architecture
- 3 integration test suites: RAG pipeline, agent mode, encryption + web search
- GitHub Actions CI: flake8 lint + offline tests on Python 3.10, 3.11, 3.12

### 📖 Documentation

- Complete MkDocs documentation rewrite for 1.0.0
- New pages: Agent Mode & Tools, Migration Guide (0.3.4 → 1.0.0)
- Updated all existing pages with 1.0.0 API and examples
- Grouped navigation: Getting Started, Guides, Integrations

### 📦 Dependencies

- `langchain>=0.3.14`, `langchain-core>=0.3.30`, `langchain-community>=0.3.14`
- `langchain-openai>=0.3.4`, `langchain-text-splitters>=0.3.0`
- `langgraph>=0.3.10` (optional, for agent mode)
- Python `>=3.10` required

---

## [0.3.4] — 2024-12-17 (Previous Release)

- Final pre-1.0 release
- ConversationalRetrievalChain-based architecture
- Basic web search, vision chat, and document ingestion
