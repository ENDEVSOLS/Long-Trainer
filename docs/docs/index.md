<p align="center">
  <img src="https://github.com/ENDEVSOLS/Long-Trainer/blob/master/assets/longtrainer.png?raw=true" alt="LongTrainer Logo">
</p>

<h1 align="center">LongTrainer 1.3.1 — Production-Ready RAG Framework</h1>

<p align="center">
  <strong>Multi-tenant bots, streaming, tools, and persistent memory — all batteries included.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/longtrainer/">
    <img src="https://img.shields.io/pypi/v/longtrainer" alt="PyPI Version">
  </a>
  <a href="https://pepy.tech/project/longtrainer">
    <img src="https://static.pepy.tech/badge/longtrainer" alt="Total Downloads">
  </a>
  <a href="https://pepy.tech/project/longtrainer">
    <img src="https://static.pepy.tech/badge/longtrainer/month" alt="Monthly Downloads">
  </a>
</p>
<hr />

# Welcome to LongTrainer 1.3.1

LongTrainer is a **production-ready RAG framework** that turns your documents into intelligent, multi-tenant chatbots with minimal code. Built on top of LangChain, it handles multi-bot isolation, persistent MongoDB memory, FAISS vector search, streaming responses, custom tool calling, chat encryption, and vision support.

## Quick Start

Install LongTrainer and start building in minutes:

```bash
pip install longtrainer
```

### RAG Mode (Default)

```python
from longtrainer.trainer import LongTrainer
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

trainer = LongTrainer(mongo_endpoint="mongodb://localhost:27017/")
bot_id = trainer.initialize_bot_id()

trainer.add_document_from_path("data.pdf", bot_id)
trainer.create_bot(bot_id)

chat_id = trainer.new_chat(bot_id)
answer, sources = trainer.get_response("What is this about?", bot_id, chat_id)
print(answer)
```

### Agent Mode (With Tools)

```python
from longtrainer.tools import web_search
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

trainer.add_tool(web_search, bot_id)
trainer.add_tool(calculate, bot_id)

trainer.create_bot(bot_id, agent_mode=True)
chat_id = trainer.new_chat(bot_id)
answer, _ = trainer.get_response("What is 42 * 17?", bot_id, chat_id)
```

### Zero-Code CLI & API Server
Manage bots, chat, and run a production API directly from your terminal—no Python required.

#### A. Interactive Terminal Chat

![CLI Quick Start](assets/01-cli-quickstart.gif)

```bash
# 1. Initialize a new project and generate longtrainer.yaml
longtrainer init

# 2. Create a new bot
longtrainer bot create --prompt "You are a helpful assistant."

# 3. Add a document (PDF, link, etc.)
longtrainer add-doc <bot_id> /path/to/document.pdf

# 4. Start chatting!
longtrainer chat <bot_id>
```

#### B. Interactive Streaming Chat

![Interactive Chat](assets/02-interactive-chat.gif)

```bash
longtrainer chat <bot_id>
```

#### C. FastAPI REST Server

![API Server](assets/03-api-server.gif)

Start a production-ready API server backed by your LongTrainer bots:
```bash
longtrainer serve
```

Visit `http://localhost:8000/docs` to see the auto-generated Swagger UI.

## What's New in 1.2.2

- **Dynamic Tool Engine:** Inject any LangChain tool by string name — zero custom code (`"wikipedia"`, `"arxiv"`, `"tavily_search_results_json"`)
- **Enterprise Vector DBs:** 9 providers — FAISS, Pinecone, Chroma, Qdrant, PGVector, MongoDB Atlas, Milvus, Weaviate, Elasticsearch
- **Enterprise Document Loaders:** AWS S3, Google Drive, Confluence, GitHub, JSON, Notion, and dynamic loader injection
- **Dynamic Model Factory:** OpenAI, Anthropic, Google, AWS Bedrock, HuggingFace, Groq, Together, Ollama
- **CLI & API Enhancements:** `--tools` flag and API `tools` parameter for zero-code tool injection
- **Tool Persistence:** Tool selections saved to MongoDB and restored on bot reload

## What's New in 1.3.1

- **LongTracer Integration:** Native, optional observability and hallucination detection for all response paths (RAG, Agent, Vision, Structured)
- **CitationVerifier:** Hybrid STS + NLI claim verification to detect hallucinated responses
- **Lightweight Tracing:** `tracer_verify=False` for spans-only observability without the ~500MB NLI model download
- **Graceful Degradation:** Zero overhead when disabled, no crash when `longtracer` is not installed
- **Lazy Loading:** Chat histories are lazy-loaded on demand from MongoDB, eliminating startup latency

## What's New in 1.0.0

- **Dual Mode:** RAG (LCEL) for simple Q&A, Agent (LangGraph) for tool calling
- **Streaming Responses:** Sync and async out of the box
- **Custom Tool Calling:** `add_tool()` with any LangChain `@tool`
- **Per-Bot Customization:** Independent LLM, embeddings, and retrieval config per bot
- **Chat Encryption:** Fernet encryption for stored conversations

Upgrading from 0.3.4? See the [Migration Guide](migration.md).

## Documentation

| Guide | Description |
|---|---|
| [Installation](installation.md) | Install LongTrainer and system dependencies |
| [Creating an Instance](creating_instance.md) | Configure the LongTrainer class |
| [Creating and Using a Bot](creating_using_bot.md) | Bot lifecycle: create, load, chat |
| [Agent Mode & Tools](agent_mode.md) | Tool calling, streaming, agent configuration |
| [CLI & API Server](cli_api.md) | Zero-code CLI and REST API |
| [Rate Limiting](rate_limiting.md) | Two-layer execution rate limit configuration |
| [Chat Management](chat_management.md) | Sessions, history, training on chats |
| [Supported Formats](supported_formats.md) | Document types and ingestion methods |
| [Integrating LLMs](integrating_llms.md) | Use any LangChain-compatible LLM |
| [Integrating Embeddings](integrating_embeddings.md) | Custom embedding models |
| [Observability & Tracing](tracer.md) | LongTracer hallucination detection & spans |
| [Updating Bots](updating_bots.md) | Add documents and reconfigure bots |
| [Deleting Bots](deleting_bot.md) | Remove bots and associated data |
| [Migration 0.3.4 → 1.0.0](migration.md) | Breaking changes and upgrade path |

## Part of the Long Suite

LongTrainer is part of the **Long Suite** — a collection of tools for building, testing, and monitoring production RAG systems.

| Project | Description |
|---|---|
| **[LongParser](https://github.com/ENDEVSOLS/LongParser)** | Document ingestion and chunking |
| **[LongTrainer](https://github.com/ENDEVSOLS/Long-Trainer)** | RAG chatbot framework *(you are here)* |
| **[LongTracer](https://github.com/ENDEVSOLS/LongTracer)** | Hallucination detection and tracing |
| **[LongProbe](https://github.com/ENDEVSOLS/LongProbe)** | Retrieval regression testing |
