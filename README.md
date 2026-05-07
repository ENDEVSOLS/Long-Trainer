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
  <a href="https://github.com/ENDEVSOLS/Long-Trainer/stargazers">
    <img src="https://img.shields.io/github/stars/ENDEVSOLS/Long-Trainer?style=flat" alt="GitHub Stars">
  </a>
  <a href="https://github.com/ENDEVSOLS/Long-Trainer/actions/workflows/ci.yml">
    <img src="https://github.com/ENDEVSOLS/Long-Trainer/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/longtrainer" alt="Python Versions">
  <a href="https://github.com/ENDEVSOLS/Long-Trainer/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/ENDEVSOLS/Long-Trainer" alt="License">
  </a>
  <a href="https://opencollective.com/longtrainer">
    <img src="https://img.shields.io/opencollective/all/longtrainer?label=sponsors" alt="Open Collective">
  </a>
</p>

<p align="center">
  <a href="https://endevsols.github.io/Long-Trainer/">Documentation</a> •
  <a href="#quick-start-">Quick Start</a> •
  <a href="#features-">Features</a> •
  <a href="#migration-from-034">Migration from 0.3.4</a> •
  <a href="#support-the-project-">Sponsor</a>
</p>

---

> **"5 lines to production RAG"** — Multi-tenant bots, streaming, tools, and persistent memory.

---

## What is LongTrainer?

LongTrainer is a **production-ready RAG framework** that turns your documents into intelligent, multi-tenant chatbots — with **5 lines of code**.

Built on top of LangChain, LongTrainer handles the hard parts that every production RAG system needs: **multi-bot isolation, persistent MongoDB memory, FAISS vector search, streaming responses, custom tool calling, chat encryption, and vision support** — so you don't have to wire them together yourself.

### Why LongTrainer over raw LangChain / LlamaIndex?

| Problem | LangChain / LlamaIndex | LongTrainer |
|---|---|---|
| Multi-bot management | DIY — manage state per bot | Built-in: `initialize_bot_id()` → isolated bots |
| Persistent chat memory | Wire MongoDB/Redis yourself | Built-in: MongoDB-backed, encrypted, restorable |
| Document ingestion | Assemble loaders + splitters | One-liner: `add_document_from_path(path, bot_id)` |
| Streaming responses | Implement `astream` yourself | `get_response(stream=True)` yields chunks |
| Custom tool calling | Define tools, build agent | `add_tool(my_tool)` — plug and play |
| Web search augmentation | Find and integrate search | Built-in toggle: `web_search=True` |
| Vision chat | Complex multi-modal setup | `get_vision_response()` — pass images |
| Self-improving from chats | Not a concept | `train_chats()` feeds Q&A back into KB |
| Encryption at rest | DIY | `encrypt_chats=True` — Fernet out of the box |

---

## Installation

```bash
pip install longtrainer
```

**With agent/tool-calling support (optional):**

```bash
pip install longtrainer[agent]
```

**With observability & hallucination detection (optional):**

```bash
pip install longtrainer[tracer]
```

### System Dependencies

<details>
<summary><strong>Linux (Ubuntu/Debian)</strong></summary>

```bash
sudo apt install libmagic-dev poppler-utils tesseract-ocr qpdf libreoffice pandoc
```
</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install libmagic poppler tesseract qpdf libreoffice pandoc
```
</details>

---

## Quick Start 🚀

### 🎬 Complete Workflow Demo

<p align="center">
  <img src="https://github.com/ENDEVSOLS/Long-Trainer/blob/master/assets/longtrainer-demo.gif?raw=true" alt="LongTrainer Demo" width="700">
</p>

> Full RAG workflow: initialize → create bot → ingest documents → vector search → Q&A — all with live progress tracking. Run it yourself: `python demos/longtrainer_demo.py`

---

### 1. Zero-Code CLI & API Server

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

#### FastAPI REST Server

Start a production-ready API server backed by your LongTrainer bots:
```bash
longtrainer serve
```

This starts a FastAPI server on `http://localhost:8000` with **18 REST endpoints**, including:
- `/health`
- `/bots` (CRUD)
- `/bots/{id}/documents/path` (Ingest files)
- `/bots/{id}/chats/{chat_id}` (Chat and Streaming)

Visit `http://localhost:8000/docs` to see the auto-generated Swagger UI.

### 2. Python SDK — Default RAG Mode

```python
from longtrainer.trainer import LongTrainer
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# Initialize
trainer = LongTrainer(mongo_endpoint="mongodb://localhost:27017/")
bot_id = trainer.initialize_bot_id()

# Add documents (PDF, DOCX, CSV, HTML, MD, TXT, URLs, YouTube, Wikipedia)
trainer.add_document_from_path("path/to/your/data.pdf", bot_id)

# Create bot and start chatting
trainer.create_bot(bot_id)
chat_id = trainer.new_chat(bot_id)

# Get response
answer, sources = trainer.get_response("What is this document about?", bot_id, chat_id)
print(answer)
```

### Streaming Responses

```python
# Stream tokens in real-time
for chunk in trainer.get_response("Summarize the key points", bot_id, chat_id, stream=True):
    print(chunk, end="", flush=True)
```

### Async Streaming

```python
async for chunk in trainer.aget_response("Explain the methodology", bot_id, chat_id):
    print(chunk, end="", flush=True)
```

### AgentBot automatically routes questions to tools like web search when necessary.

### 🌟 NEW: Dynamic ZERO CODE Tools
LongTrainer V2 now integrates LangChain's massive dynamic tool ecosystem **natively**:
```python
trainer.create_bot(
    "agent-id", 
    agent_mode=True, 
    tools=["tavily_search_results_json", "wikipedia", "arxiv", "PythonREPLTool", "yahoo_finance_news"]
)
```

LongTrainer will dynamically import and initialize ANY string-based tool from `langchain.agents.load_tools` natively on the backend!

You may still register custom tools globally or per-bot explicitly:
```python
from langchain.tools import tool

@tool
def get_weather(location: str):
```

### Agent Mode — With Custom Tools

```python
from longtrainer.tools import web_search
from langchain_core.tools import tool

# Add built-in web search tool
trainer.add_tool(web_search, bot_id)

# Add your own custom tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

trainer.add_tool(calculate, bot_id)

# Create bot in agent mode
trainer.create_bot(bot_id, agent_mode=True)
chat_id = trainer.new_chat(bot_id)

response, _ = trainer.get_response("What is 42 * 17?", bot_id, chat_id)
print(response)
```

### Vision Chat

```python
vision_id = trainer.new_vision_chat(bot_id)
response, sources = trainer.get_vision_response(
    "Describe what you see in this image",
    image_paths=["photo.jpg"],
    bot_id=bot_id,
    vision_chat_id=vision_id,
)
print(response)
```

### Per-Bot Customization

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Each bot can have its own LLM, embeddings, and retrieval config
trainer.create_bot(
    bot_id,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    num_k=5,                    # retrieve 5 docs per query
    prompt_template="You are a helpful legal assistant. {context}",
    agent_mode=True,            # enable tool calling
    tools=[web_search],
)
```

---

## Features ✨

### Core
- ✅ **Dual Mode:** RAG (LCEL chain) for simple Q&A, Agent (LangGraph) for tool calling
- ✅ **Streaming Responses:** Sync and async streaming out of the box
- ✅ **Custom Tool Calling:** Add any LangChain `@tool` — web search, document reader, or your own
- ✅ **Multi-Bot Management:** Isolated bots with independent sessions, data, and configs
- ✅ **Persistent Memory:** MongoDB-backed chat history, fully restorable
- ✅ **Chat Encryption:** Fernet encryption for stored conversations
- ✅ **Observability & Tracing:** Native integration with LongTracer for logging spans and hallucination detection (`pip install longtrainer[tracer]`)

### Document Ingestion
- ✅ **Standard Formats:** PDF, DOCX, CSV, HTML, Markdown, TXT
- ✅ **Web & Crawling:** `add_document_from_link()`, `add_document_from_query()`, `add_document_from_crawl()`
- ✅ **Cloud & Enterprise:** S3 (`add_document_from_aws_s3`), Google Drive (`add_document_from_google_drive`), Confluence (`add_document_from_confluence`)
- ✅ **Structued Data:** Local Directory (`add_document_from_directory`), JSON & JQ (`add_document_from_json`), GitHub Repo (`add_document_from_github`)
- ✅ **Dynamic Integrations:** Inject ANY LangChain document loader class dynamically via `add_document_from_dynamic_loader()`

### RAG Pipeline & Vector DBs
- ✅ **Vector Databases:** FAISS, Pinecone, Chroma, Qdrant, **PGVector, MongoDB Atlas, Milvus, Elasticsearch, Weaviate**
- ✅ **Multi-Query Ensemble Retrieval:** Generates alternative queries for better recall
- ✅ **Self-Improving Memory:** `train_chats()` feeds past Q&A back into the knowledge base

### Customization
- ✅ **Per-bot LLM** — use different models for different bots
- ✅ **Per-bot Embeddings** — custom embedding models per bot
- ✅ **Per-bot Retrieval Config** — custom `num_k`, `chunk_size`, `chunk_overlap`
- ✅ **Custom Prompt Templates** — full control over system prompts
- ✅ **Vision Chat** — GPT-4 Vision support with image understanding

### Works with All LangChain-Compatible LLMs

- ✅ OpenAI (default)
- ✅ Anthropic
- ✅ Google VertexAI / Gemini
- ✅ AWS Bedrock
- ✅ HuggingFace
- ✅ Groq
- ✅ Together AI
- ✅ Ollama (local models)
- ✅ Any `BaseChatModel` implementation

---

## API Reference

### `LongTrainer` — Main Class

```python
trainer = LongTrainer(
    mongo_endpoint="mongodb://localhost:27017/",
    llm=None,                # default: ChatOpenAI(model="gpt-4o-2024-08-06")
    embedding_model=None,    # default: OpenAIEmbeddings()
    prompt_template=None,    # custom system prompt
    max_token_limit=32000,   # conversation memory limit
    num_k=3,                 # docs to retrieve per query
    chunk_size=2048,         # text splitter chunk size
    chunk_overlap=200,       # text splitter overlap
    ensemble=False,          # enable multi-query ensemble retrieval
    encrypt_chats=False,     # enable Fernet encryption
    encryption_key=None,     # custom encryption key (auto-generated if None)
    enable_tracer=False,     # enable LongTracer observability
    tracer_backend="mongo",    # tracer backend ('mongo', 'sqlite', 'memory')
    tracer_verify=True,      # run CitationVerifier (hallucination detection)
    tracer_verbose=False,    # print tracer spans to console
    tracer_threshold=0.5,    # confidence threshold for tracer (0-1)
)
```

### Key Methods

| Method | Description |
|---|---|
| `initialize_bot_id()` | Create a new bot, returns `bot_id` |
| `create_bot(bot_id, ...)` | Build the bot from loaded documents |
| `load_bot(bot_id)` | Restore an existing bot from MongoDB + FAISS |
| `new_chat(bot_id)` | Start a new chat session, returns `chat_id` |
| `get_response(query, bot_id, chat_id, stream=False)` | Get response (or stream) |
| `aget_response(query, bot_id, chat_id)` | Async streaming response |
| `add_document_from_path(path, bot_id)` | Ingest a file |
| `add_document_from_link(links, bot_id)` | Ingest URLs / YouTube links |
| `add_tool(tool, bot_id)` | Register a tool for a bot |
| `remove_tool(tool_name, bot_id)` | Remove a tool |
| `list_tools(bot_id)` | List registered tools |
| `train_chats(bot_id)` | Self-improve from chat history |
| `new_vision_chat(bot_id)` | Start a vision chat session |
| `get_vision_response(query, images, bot_id, vision_id)` | Vision response |

---

## Migration from 0.3.4

LongTrainer 1.0.0 is a major upgrade with breaking changes:

| 0.3.4 | 1.0.0 |
|---|---|
| `ConversationalRetrievalChain` | LCEL chain (`RAGBot`) or LangGraph agent (`AgentBot`) |
| `requirements.txt` + `setup.py` | `pyproject.toml` (UV/pip compatible) |
| No streaming | `stream=True` or `aget_response()` |
| No tool calling | `add_tool()` + `agent_mode=True` |
| `langchain.memory` | `langchain_core.chat_history` |
| Fixed LLM for all bots | Per-bot LLM, embeddings, and config |

**Upgrade path:**
```bash
pip install --upgrade longtrainer
```

The core API (`initialize_bot_id`, `create_bot`, `new_chat`, `get_response`) remains the same — existing code should work with minimal changes. The main difference is `get_response()` now returns `(answer, sources)` instead of `(answer, sources, web_sources)`.

---


## Part of the Long Suite

LongTrainer is part of the **Long Suite** — a collection of tools for building, testing, and monitoring production RAG systems.

| Project | Description |
|---|---|
| **[LongParser](https://github.com/ENDEVSOLS/LongParser)** | Document ingestion and chunking |
| **[LongTrainer](https://github.com/ENDEVSOLS/Long-Trainer)** | RAG chatbot framework *(you are here)* |
| **[LongTracer](https://github.com/ENDEVSOLS/LongTracer)** | Hallucination detection and tracing |
| **[LongProbe](https://github.com/ENDEVSOLS/LongProbe)** | Retrieval regression testing |

---

## Citation

```
@misc{longtrainer,
  author = {Endevsols},
  title = {LongTrainer: Production-Ready RAG Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ENDEVSOLS/Long-Trainer}},
}
```

## License

[MIT License](LICENSE)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
