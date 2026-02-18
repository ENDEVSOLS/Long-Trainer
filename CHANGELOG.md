# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] — 2026-02-17

### ⚠️ Breaking Changes

- **Removed** `ConversationalRetrievalChain` — replaced with LCEL-based `RAGBot` and LangGraph-based `AgentBot`
- **Removed** `setup.py` and `requirements.txt` — migrated to `pyproject.toml` (UV/pip compatible)
- **Removed** `langchain.memory.ConversationTokenBufferMemory` — replaced with `InMemoryChatMessageHistory`
- **Removed** `EnsembleRetriever` / `MultiQueryRetriever` from deprecated `langchain_classic` — replaced with custom `MultiQueryEnsembleRetriever`
- **Changed** `get_response()` return type — now returns `(answer, sources)` tuple

### ✨ New Features

- **Dual Mode Architecture**: RAG mode (default, LCEL chain) + Agent mode (LangGraph, tool calling)
- **Streaming Responses**: `get_response(stream=True)` yields tokens, `aget_response()` for async
- **Custom Tool Calling**: `add_tool()`, `remove_tool()`, `list_tools()` — register any `@tool` decorated function
- **Built-in Tools**: `web_search` (DuckDuckGo) and `document_reader` (multi-format)
- **Per-Bot Configuration**: Custom LLM, embeddings, retriever config, and prompt per bot via `create_bot()`
- **Tool Registry**: `ToolRegistry` class for managing tools globally or per-bot

### 🔧 Improvements

- All imports updated to latest LangChain 2026 standards (`langchain_core`, `langchain_community`, `langchain_text_splitters`)
- Full type hints across all modules
- Comprehensive docstrings on all public methods
- `pyproject.toml` with `hatchling` build system
- `langgraph` is an **optional dependency** — `pip install longtrainer[agent]`
- Optional `[api]` and `[dev]` dependency groups

### 📦 Dependencies

- `langchain>=0.3.14`, `langchain-core>=0.3.30`, `langchain-community>=0.3.14`
- `langchain-openai>=0.3.4`, `langchain-text-splitters>=0.3.0`
- `langgraph>=0.3.10` (optional, for agent mode)
- Python `>=3.10` required

## [0.3.3] — 2024-01-XX (Previous Release)

- Final V1 release
- ConversationalRetrievalChain-based architecture
- Basic web search, vision chat, and document ingestion
