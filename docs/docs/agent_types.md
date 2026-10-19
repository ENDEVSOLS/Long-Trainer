# Named agent types

Select a purpose when building a bot from its ingested documents:

```python
from longtrainer import LongTrainer, AgentTypeRegistry

trainer = LongTrainer()  # Configure the existing model, embeddings and MongoDB first.
bot_id = trainer.initialize_bot_id()
trainer.create_bot(bot_id, agent_type="research")
chat_id = trainer.new_chat(bot_id)
answer, sources = trainer.get_response("Compare recent findings", bot_id, chat_id)
print(AgentTypeRegistry.names())
```

| Type | Mode | Default tool identifiers | Installation and configuration |
| --- | --- | --- | --- |
| `research` | Agent | `tavily`, `wikipedia`, `arxiv` | `pip install 'longtrainer[research]'`; set `TAVILY_API_KEY` |
| `coding` | Agent | `python_repl`, `tavily` | `pip install 'longtrainer[coding]'`; set `TAVILY_API_KEY` |
| `financial` | Agent | `yahoo_finance_news`, `python_repl` | `pip install 'longtrainer[financial]'`; no additional default tool key |
| `customer_support` | RAG | None; document retrieval | Base installation and normal RAG configuration |

All agent types require a tool-calling model. The purpose extras include LangGraph. Wikipedia is already a base dependency. Research adds Tavily and arXiv; coding adds Tavily and LangChain Experimental; financial adds yfinance and LangChain Experimental. Base RAG imports do not initialize or import these optional integrations. Missing default tools, dependencies or keys cause an actionable construction error instead of silently constructing an empty agent.

Research prompts request cross-checking and Markdown citations. Coding prompts request ordered work, tests and error recovery. Financial prompts prohibit invented figures, request timestamps and sources, and require “This is informational only and is not financial advice.” Yahoo Finance News provides company news, not a market-price feed. Published news may omit timestamps; the assistant must acknowledge missing data.

Python REPL executes in the application process with its permissions. Use coding and financial agents only with trusted workloads in a suitably isolated deployment. This library does not sandbox Python execution.

Customer support uses knowledge-base text documents, a polite escalation prompt, and no external default tools. Retrieved copies include `[Document N]` identifiers and available source metadata for citations. Web search and uploaded-file augmentation are rejected in both synchronous and asynchronous text chat; vision responses are disabled for this type. Add authoritative material through normal document ingestion. Instructions request grounded answers and escalation when unsupported; model adherence and citation correctness are not independently verified. An explicit prompt override replaces these instructions.

## Overrides and restoration

The type determines mode even if `agent_mode` is also supplied. `prompt_template=None` uses the purpose prompt; any explicit prompt replaces it. RAG prompts should contain `{context}`. `tools=None` uses purpose defaults; an explicit list replaces them, including `[]`. For named bots neither previous per-bot tools nor global tools are implicitly included. Runtime tools added explicitly through `register_tool(..., bot_id=...)` can participate afterwards. Customer support remains RAG even with a tool override; RAG does not execute agent tools.

```python
trainer.create_bot(bot_id, agent_type="coding", tools=[])  # No default Python/search tools.
trainer.create_bot(bot_id, agent_type="research", tools=["wikipedia"],
                   prompt_template="Research carefully and cite your sources.")
```

Without `agent_type`, existing positional calls, default RAG, `agent_mode`, prompt handling and global/per-bot tool behavior remain supported. A direct `create_bot(bot_id)` retains legacy creation semantics; it does not infer a previously selected purpose.

Bot records store the purpose, effective prompt/mode, string tool identifiers and whether tools were explicitly overridden. No tool credentials or tool objects are serialized. `load_bot`, internal document-update/chat-training rebuilds, and CLI document ingestion retain named configuration. Tool objects remain usable in the current process and during rebuilds, but must be re-registered after reload; their names are not assumed to be loader identifiers. Custom registry configurations must be registered again before loading records that reference them. New chats and lazily restored chats select the same runtime. Rebuilding a named bot clears cached text chains so stored history is replayed with the new configuration.

`AgentTypeRegistry.register(AgentTypeConfig(...))` supports application-defined purposes; import `AgentTypeConfig` from `longtrainer.agent_types`. Register configurations before creating or loading their bots. Unknown names raise `ValueError` listing valid names.

SQL is deferred: read-only enforcement, dialect support, credential/reconnection handling and runtime design have not passed the required review. Only four built-in names are registered. API and CLI purpose selection are available as described below. YAML contains infrastructure settings only; per-bot purpose, prompt and tools are supplied explicitly through Python, API or CLI, not inferred from YAML.

## CLI and API selection

Install the chosen purpose extra and configure the normal model, embeddings and MongoDB prerequisites before building. For research, set `TAVILY_API_KEY` and use a tool-calling model. `longtrainer build` without an ID initializes and builds a new empty bot and prints its ID:

```bash
pip install 'longtrainer[research,api,cli]'
longtrainer init
longtrainer build --agent-type research
longtrainer bot create --agent-type customer_support
longtrainer build BOT_ID --agent-type coding --tools ""
```

An optional `BOT_ID` builds an existing bot after loading its configuration. With no purpose/mode/tool override it preserves the stored named configuration; `--prompt` overrides the stored prompt. A new explicit purpose chooses that purpose's defaults. An explicit `--tools` list replaces defaults; `--tools ""` selects no tools. For an existing named bot a tool-only override retains its stored purpose and prompt. `--agent` selects the legacy agent path if `--agent-type` is absent. Existing `bot create`, `--agent`, `--tools`, and ingestion commands remain supported. Invalid purposes fail before creating or loading a bot and list supported names.

Infrastructure YAML sets MongoDB/model/embedding/chunking settings. Keys such as `agent_type` and per-bot tools/prompts in YAML are not supported and do not select a purpose. This exclusion avoids hidden precedence: explicit CLI/API/Python parameters determine purpose settings; existing bot records supply configuration for a plain CLI rebuild.

Use the existing HTTP build route after initializing an ID with `POST /bots` and optionally uploading documents:

```http
POST /bots/BOT_ID/build
Content-Type: application/json

{"agent_type": "research", "tools": ["wikipedia"]}
```

`agent_type` is optional. Existing payloads keep their legacy behavior. Unknown string names produce HTTP **400** with supported names; malformed non-string fields use ordinary request validation. Omit `tools` or use JSON `null` for type defaults, and use `[]` to replace them with no tools. Explicit `prompt_template` replaces the purpose prompt. `agent_mode` remains accepted; a supplied purpose determines mode. HTTP builds without `agent_type` keep the legacy direct-creation behavior rather than inferring a stored purpose.
