# Changelog

All notable changes to this project will be documented in this file.

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
