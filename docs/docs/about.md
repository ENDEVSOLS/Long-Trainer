# About LongTrainer

LongTrainer is a production-ready RAG framework built on LangChain, designed for managing multiple bots with isolated, context-aware chat sessions. It handles the infrastructure that every production RAG system needs — so you can focus on your application logic.

## Features

### Core

- ✅ **Dual Mode:** RAG (LCEL chain) for document Q&A, Agent (LangGraph) for tool calling
- ✅ **Streaming Responses:** Sync (`stream=True`) and async (`aget_response()`) streaming
- ✅ **Custom Tool Calling:** Register any LangChain `@tool` — built-in web search, document reader, or your own
- ✅ **Multi-Bot Management:** Isolated bots with independent sessions, data, and configurations
- ✅ **Persistent Memory:** MongoDB-backed chat history, fully restorable across restarts
- ✅ **Chat Encryption:** Fernet encryption for stored conversations
- ✅ **Per-Bot Customization:** Independent LLM, embeddings, retrieval config, and prompt templates per bot

### Document Ingestion

- ✅ **PDF, DOCX, CSV, HTML, Markdown, TXT** — auto-detected by extension
- ✅ **URLs, YouTube, Wikipedia** — via `add_document_from_link()` / `add_document_from_query()`
- ✅ **Any format** via `use_unstructured=True` (PowerPoint, images, etc.)

### RAG Pipeline

- ✅ **FAISS Vector Store** — fast similarity search with batched indexing
- ✅ **Multi-Query Ensemble Retrieval** — generates alternative queries for better recall
- ✅ **Self-Improving:** `train_chats()` feeds past Q&A back into the knowledge base

### Vision

- ✅ **GPT-4 Vision Support** — image understanding with context-aware responses
- ✅ **Vision Chat Sessions** — separate vision chat histories with MongoDB persistence

## Supported LLMs and Embeddings

LongTrainer works with any LangChain-compatible model:

- ✅ OpenAI (default)
- ✅ Anthropic
- ✅ Google VertexAI / Gemini
- ✅ AWS Bedrock
- ✅ HuggingFace
- ✅ Groq
- ✅ Together AI
- ✅ Ollama (local models)
- ✅ Any `BaseChatModel` implementation

## Use Cases

- **Enterprise Solutions:** Multi-tenant customer support with isolated bots per department
- **Educational Platforms:** AI tutors that maintain context across sessions
- **Healthcare Applications:** Context-aware patient interaction with encrypted chat storage
- **Research Tools:** Agent-powered assistants with web search and custom analysis tools
- **Knowledge Bases:** Self-improving document Q&A systems
