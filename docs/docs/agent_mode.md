## Agent Mode & Tool System

LongTrainer 1.0.0 supports two modes of operation:

- **RAG Mode** (default): Standard retrieval-augmented generation using an LCEL chain
- **Agent Mode**: LangGraph-powered agent with tool calling capabilities

### Enabling Agent Mode

```python
trainer.create_bot(bot_id, agent_mode=True)
```

When `agent_mode=True`, the bot uses LangGraph's `create_react_agent` instead of a simple RAG chain. This allows the bot to decide when to call tools and how to combine results.

!!! note
    Agent mode requires the `langgraph` package. Install it with:
    ```bash
    pip install longtrainer[agent]
    ```

### Registering Tools

#### Built-in Tools

LongTrainer ships with two built-in tools:

```python
from longtrainer.tools import web_search, document_reader, get_builtin_tools

# Register individually
trainer.add_tool(web_search, bot_id)

# Or get all built-in tools
for tool in get_builtin_tools():
    trainer.add_tool(tool, bot_id)
```

| Tool | Description |
|---|---|
| `web_search` | Search the web using DuckDuckGo, returns top 5 results |
| `document_reader` | Extract text from files (PDF, DOCX, TXT, CSV, HTML, Markdown) |

#### Custom Tools

Create custom tools using LangChain's `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your weather API logic here
    return f"Weather in {city}: 72°F, Sunny"

trainer.add_tool(calculate, bot_id)
trainer.add_tool(get_weather, bot_id)
```

!!! tip
    Write clear docstrings for your tools — the agent uses them to decide when and how to call each tool.

### Dynamic ZERO-CODE Tools (NEW in V2)

LongTrainer V2 natively integrates LangChain's massive dynamic tool ecosystem. You can inject standard tools using just their string name from `langchain.agents.load_tools`!

```python
# Pass strings directly to `create_bot`
trainer.create_bot(
    "agent-id", 
    agent_mode=True, 
    tools=[
        "tavily_search_results_json", 
        "wikipedia", 
        "arxiv", 
        "PythonREPLTool", 
        "yahoo_finance_news"
    ]
)
```

LongTrainer will automatically:
1. Trap the String
2. Import the correct LangChain package internally
3. Instantiate the Tool 
4. Feed it into `bot["tools"].register(t)` 

*These injected tools are fully saved to the MongoDB configuration and will restore perfectly upon restarts!*

### Global vs Bot-Specific Tools

Tools can be registered globally (available to all bots) or per-bot:

```python
# Global — available to every bot
trainer.add_tool(web_search)

# Bot-specific — only for this bot
trainer.add_tool(calculate, bot_id)
```

### Managing Tools

```python
# List all tools for a bot (includes global tools)
tool_names = trainer.list_tools(bot_id)
print(tool_names)  # ['web_search', 'calculate']

# Remove a tool
trainer.remove_tool("calculate", bot_id)
```

### Using the Tool Registry Directly

For advanced use cases, you can work with the `ToolRegistry` class:

```python
from longtrainer.tools import ToolRegistry

registry = ToolRegistry()
registry.register(calculate)
registry.register(web_search)

# Get a tool by name
calc = registry.get("calculate")

# List all tool names
print(registry.list_tool_names())

# Unregister a tool
registry.unregister("calculate")
```

### Streaming with Agent Mode

Agent mode fully supports streaming:

```python
trainer.create_bot(bot_id, agent_mode=True)
chat_id = trainer.new_chat(bot_id)

# Sync streaming
for chunk in trainer.get_response("Search for Python news", bot_id, chat_id, stream=True):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in trainer.aget_response("What is 42 * 17?", bot_id, chat_id):
    print(chunk, end="", flush=True)
```

### RAG Mode vs Agent Mode

| Feature | RAG Mode | Agent Mode |
|---|---|---|
| Architecture | LCEL chain | LangGraph ReAct agent |
| Tool calling | ❌ | ✅ |
| Document retrieval | ✅ (automatic) | ✅ (via context) |
| Web search | Via `web_search=True` param | Via `web_search` tool |
| Streaming | ✅ | ✅ |
| Best for | Simple document Q&A | Complex tasks requiring tools |

### Complete Example

```python
from longtrainer.trainer import LongTrainer
from longtrainer.tools import web_search
from langchain_core.tools import tool

trainer = LongTrainer()
bot_id = trainer.initialize_bot_id()

# Add knowledge base
trainer.add_document_from_path("company_docs.pdf", bot_id)

# Create custom tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Register tools and create agent bot
trainer.add_tool(web_search, bot_id)
trainer.add_tool(calculate, bot_id)
trainer.create_bot(bot_id, agent_mode=True)

# Chat with tools
chat_id = trainer.new_chat(bot_id)
answer, _ = trainer.get_response(
    "Search the web for today's weather and calculate 32°F to Celsius",
    bot_id, chat_id,
)
print(answer)
```
