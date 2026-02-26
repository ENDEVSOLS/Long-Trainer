## Migration Guide: 1.0.0 → 1.2.0

LongTrainer 1.2.0 is a **backward-compatible** feature release. No breaking changes — all existing code continues to work as-is.

### Upgrade

```bash
pip install --upgrade longtrainer
```

### What's New (No Migration Required)

#### Dynamic Tool Injection

You can now pass tools as strings instead of importing and instantiating them manually:

```python
# 1.0.0 — manual tool registration
from longtrainer.tools import web_search
trainer.add_tool(web_search, bot_id)
trainer.create_bot(bot_id, agent_mode=True)

# 1.2.0 — zero-code string injection (ALSO works)
trainer.create_bot(bot_id, agent_mode=True, tools=["wikipedia", "arxiv", "tavily_search_results_json"])
```

Tool selections are automatically saved to MongoDB and restored on `load_bot()`.

#### Enterprise Vector Databases

You can now use any of 9 vector store providers instead of only FAISS:

```python
# 1.0.0 — FAISS only (still the default)
trainer = LongTrainer()

# 1.2.0 — choose your provider
trainer = LongTrainer(
    vector_store_provider="qdrant",  # or pinecone, chroma, pgvector, mongodb, milvus, weaviate, elasticsearch
    vector_store_kwargs={"url": "http://localhost:6333"}
)
```

#### Enterprise Document Loaders

New helper methods for loading from cloud and enterprise platforms:

| Method | Source |
|---|---|
| `load_directory()` | Local folders with glob patterns |
| `load_json()` | JSON files with jq schema |
| `load_confluence()` | Atlassian Confluence |
| `load_github()` | GitHub repositories |
| `load_s3()` | AWS S3 buckets |
| `load_google_drive()` | Google Drive folders |
| `load_dynamic_loader()` | Any LangChain loader by class name |

#### Dynamic Model Factory

New providers supported out of the box:

```python
trainer = LongTrainer(
    llm_provider="anthropic",        # or google, bedrock, huggingface, groq, together, ollama
    default_llm="claude-3-5-sonnet-20241022",
)
```

#### CLI & API

```bash
# New --tools flag
longtrainer bot create --agent --tools "wikipedia,arxiv"
```

```json
// API: POST /bots/{bot_id}/build
{ "agent_mode": true, "tools": ["wikipedia", "arxiv"] }
```

### Dependency Changes (1.0.0 → 1.2.0)

| New Optional Dependency | Used For |
|---|---|
| `langchain_community` | Dynamic tools & loaders |
| `langchain_experimental` | Python REPL tool |
| Provider packages (e.g. `langchain_pinecone`) | Enterprise vector stores |

---

## Migration Guide: 0.3.4 → 1.0.0

LongTrainer 1.0.0 is a major rewrite with significant improvements and some breaking changes.

### Upgrade

```bash
pip install --upgrade longtrainer
```

For agent mode support:

```bash
pip install --upgrade longtrainer[agent]
```

### Breaking Changes

| Area | 0.3.4 | 1.0.0 |
|---|---|---|
| **Internal chain** | `ConversationalRetrievalChain` | LCEL chain (`RAGBot`) or LangGraph agent (`AgentBot`) |
| **Packaging** | `requirements.txt` + `setup.py` | `pyproject.toml` (compatible with pip, UV, and Poetry) |
| **Memory** | `langchain.memory.ConversationTokenBufferMemory` | `langchain_core.chat_history.InMemoryChatMessageHistory` |
| **Response format** | `get_response()` returns `(answer, sources, web_sources)` | Returns `(answer, web_sources)` |
| **LLM default** | `gpt-4-turbo` | `gpt-4o-2024-08-06` |

### What Stays the Same

The core API surface is unchanged:

```python
# These all work exactly as before
trainer = LongTrainer(mongo_endpoint="mongodb://localhost:27017/")
bot_id = trainer.initialize_bot_id()
trainer.add_document_from_path("data.pdf", bot_id)
trainer.create_bot(bot_id)
chat_id = trainer.new_chat(bot_id)
trainer.load_bot(bot_id)
trainer.update_chatbot(paths, bot_id)
trainer.delete_chatbot(bot_id)
trainer.list_chats(bot_id)
trainer.get_chat_by_id(chat_id)
trainer.train_chats(bot_id)
```

### Code Changes Required

#### Response Unpacking

```python
# 0.3.4
answer, sources, web_sources = trainer.get_response(query, bot_id, chat_id)

# 1.0.0
answer, web_sources = trainer.get_response(query, bot_id, chat_id)
```

#### Imports

```python
# 0.3.4 — these still work
from longtrainer.trainer import LongTrainer

# 1.0.0 — new top-level imports available
from longtrainer import LongTrainer, ToolRegistry, web_search
```

### New Features in 1.0.0

These are new capabilities that don't require migration — just start using them:

#### Streaming

```python
# Sync streaming
for chunk in trainer.get_response(query, bot_id, chat_id, stream=True):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in trainer.aget_response(query, bot_id, chat_id):
    print(chunk, end="", flush=True)
```

#### Agent Mode with Tools

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
```

#### Per-Bot Customization

```python
trainer.create_bot(
    bot_id,
    llm=ChatOpenAI(model="gpt-4o-mini"),
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    num_k=5,
    prompt_template="Custom prompt. {context}",
)
```

### Dependency Changes

| 0.3.4 Dependency | 1.0.0 Replacement |
|---|---|
| `langchain` (monolithic) | `langchain_core`, `langchain_community`, `langchain_openai` |
| `setup.py` | `pyproject.toml` |
| `requirements.txt` | `pyproject.toml` `[project.dependencies]` |
| N/A | `langgraph` (optional, for agent mode) |
