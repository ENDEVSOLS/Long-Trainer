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
