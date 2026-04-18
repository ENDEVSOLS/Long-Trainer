# LongTrainer — Complete New Roadmap 🔥
## From v1.2.0 → v2.0.0 (The Beast)

> **Research Base:** LangChain, LangGraph (Supervisor + Swarm), LlamaIndex, MCP Protocol (Model Context Protocol), LangSmith — all researched via Context7 on April 7–9, 2026.
> **Codebase Audit:** Full live audit of `trainer.py`, `bot.py`, `chat.py`, `tools.py`, `vectorstores.py`, `retrieval.py` conducted on April 7, 2026.

---

## 📊 Current State Audit (v1.2.0)

### ✅ What IS Complete and Production-Ready

| Feature | File | Status |
|---|---|---|
| Multi-bot isolation | `trainer.py` | ✅ Complete |
| LCEL RAG chain (non-agent mode) | `bot.py → RAGBot` | ✅ Complete |
| ReAct Agent via LangGraph | `bot.py → AgentBot` | ✅ Complete |
| MongoDB persistence | `storage.py` | ✅ Complete |
| Fernet chat encryption | `storage.py` | ✅ Complete |
| Sync + Async streaming responses | `chat.py` | ✅ Complete |
| 9 Vector store providers | `vectorstores.py` | ✅ Complete |
| 12+ Document loaders | `loaders.py`, `documents.py` | ✅ Complete |
| Dynamic tool injection by string | `tools.py` | ✅ Complete |
| Multi-Query + Ensemble retriever | `retrieval.py` | ✅ Complete |
| Vision chat (GPT-4V) | `vision_bot.py` | ✅ Complete |
| Web search built-in | `chat.py` | ✅ Complete |
| CLI + FastAPI server (16+ endpoints) | `cli.py`, `api.py` | ✅ Complete |
| Train bot on own chat history | `trainer.py` | ✅ Complete |
| Model Factory (6+ providers) | `models.py` | ✅ Complete |

### ❌ Known Bugs to Fix Immediately

| Bug | Location | Description |
|---|---|---|
| `update_chatbot()` references wrong key | `trainer.py:L611` | Uses `bot["faiss_path"]` — should be `bot["db_path"]` |
| No async document ingestion | `documents.py` | `add_document_from_path()` is sync — blocks the server thread |
| No retrieval confidence scores | `retrieval.py` | `invoke_vectorstore()` returns docs without similarity scores |
| Serial document loading | `documents.py` | 100+ docs load serially — no parallel ingestion pipeline |

---

---

## Phase 3 — v1.3.0: Named Custom Agent Types
> **Priority: HIGHEST — Start Here**
> **Estimated time: 2–3 weeks**

### The Problem
Right now `agent_mode=True` gives a generic ReAct agent with no tuning for the use case.
An experienced engineer comparing LongTrainer to LangChain sees "just one agent type."
Named agent types give pre-configured architectures — tuned system prompt, default tools,
and the correct LangGraph graph topology for each domain.

### 5 Agent Types to Build

#### 3.1 — `ResearchAgent`
Multi-source information gathering. Produces structured reports with citations.

```python
trainer.create_bot(bot_id, agent_type="research")
```

- **Default tools:** `tavily_search_results_json`, `wikipedia`, `arxiv`
- **System prompt:** Optimized for cross-referencing sources, avoiding hallucination, producing structured markdown reports
- **LangGraph topology:** Parallel tool execution → synthesis → response

#### 3.2 — `SQLAgent`
Full multi-step SQL reasoning agent with query validation before execution.

```python
trainer.create_bot(bot_id, agent_type="sql", db_uri="postgresql://...")
```

- **Default tools:** `sql_db_list_tables`, `sql_db_schema`, `sql_db_query`, `sql_db_query_checker`
- **LangGraph topology:** list_tables → get_schema → generate_query → check_query → execute
- **Safety:** DDL statements (INSERT/UPDATE/DELETE/DROP) blocked by default

#### 3.3 — `CodingAgent`
Software engineering agent. Writes code, executes it in a sandbox, iterates on errors.

```python
trainer.create_bot(bot_id, agent_type="coding")
```

- **Default tools:** `PythonREPLTool`, `tavily_search_results_json`
- **System prompt:** Step-by-step reasoning, test-after-write discipline, error recovery loops

#### 3.4 — `FinancialAgent`
Live market data + numerical analysis. Accuracy-first system prompt.

```python
trainer.create_bot(bot_id, agent_type="financial")
```

- **Default tools:** `YahooFinanceNewsTool`, `PythonREPLTool` (for math)
- **System prompt:** No hallucinated prices, always includes data timestamps and disclaimers

#### 3.5 — `CustomerSupportAgent`
Grounded support agent. Cannot go outside the document corpus.

```python
trainer.create_bot(bot_id, agent_type="customer_support")
```

- **Topology:** LCEL (not open-ended ReAct) — answers stay document-grounded
- **System prompt:** Polite, escalation-aware, always cites source documents

### Implementation Architecture

```
longtrainer/
└── agent_types/
    ├── __init__.py          # exports AgentTypeRegistry
    ├── base.py              # AgentTypeConfig dataclass
    ├── research.py          # ResearchAgent
    ├── sql.py               # SQLAgent
    ├── coding.py            # CodingAgent
    ├── financial.py         # FinancialAgent
    └── customer_support.py  # CustomerSupportAgent
```

Each file exports:
- `DEFAULT_TOOLS: list[str]` — tool names for dynamic loading
- `SYSTEM_PROMPT: str` — tuned system prompt
- `build_graph(llm, tools, retriever) -> CompiledGraph` — optional custom topology

**API Change (100% backward compatible):**
```python
# Old (still works)
trainer.create_bot(bot_id, agent_mode=True, tools=["wikipedia"])

# New
trainer.create_bot(bot_id, agent_type="research")  # uses research defaults
trainer.create_bot(bot_id, agent_type="sql", db_uri="...")
trainer.create_bot(bot_id, agent_type="coding", tools=["PythonREPLTool"])  # override tools
```

---

---

## Phase 4 — v1.4.0: Multi-Agent Systems
> **Priority: HIGH**
> **Estimated time: 3–4 weeks**
> **Dependencies:** Phase 3 must be complete

### The Problem
Single-agent bots are 2023 thinking. Production 2026 systems route user requests across
fleets of specialized agents. LangChain has `langgraph-supervisor`. LlamaIndex has
`AgentWorkflow`. LongTrainer has neither.

### 4.1 — Supervisor Pattern

A central "supervisor bot" intelligently routes requests to the best specialist sub-bot.

```python
# Register specialist bots
trainer.create_bot(research_bot_id, agent_type="research")
trainer.create_bot(sql_bot_id, agent_type="sql", db_uri="...")
trainer.create_bot(coding_bot_id, agent_type="coding")

# Create supervisor
trainer.create_supervisor(
    supervisor_id,
    managed_bots=[research_bot_id, sql_bot_id, coding_bot_id],
    prompt="You are a team supervisor. Route requests to the right specialist."
)

# Single API call — supervisor handles routing internally
answer = trainer.get_response("Write a Python script to query our sales DB...",
                               supervisor_id, chat_id)
# Internally: supervisor → coding_bot (writes script) + sql_bot (writes query)
```

### 4.2 — Swarm Pattern

Peer-to-peer handoffs. Agents dynamically pass control to each other.

```python
trainer.create_swarm(
    swarm_id,
    bot_ids=[research_bot_id, coding_bot_id, financial_bot_id],
    default_bot=research_bot_id
)
# research_bot researches → hands off to financial_bot → hands off to coding_bot
```

### 4.3 — Nested Hierarchies

Supervisors can manage other supervisors for enterprise-scale deployments.

```python
research_team = trainer.create_supervisor(
    "research_team",
    managed_bots=[research_bot_id, sql_bot_id]
)
eng_team = trainer.create_supervisor(
    "eng_team",
    managed_bots=[coding_bot_id]
)
enterprise_cto_bot = trainer.create_supervisor(
    "cto_bot",
    managed_bots=[research_team, eng_team]
)
```

### Custom Handoff Tools

```python
trainer.create_swarm(
    swarm_id,
    bot_ids=[...],
    handoff_with_context=True  # passes task_description on handoff
)
```

### Implementation

- Wrap `langgraph-supervisor-py` and `langgraph-swarm-py`
- Add `trainer.create_supervisor()` and `trainer.create_swarm()` to `trainer.py`
- Store supervisor/swarm topology in MongoDB alongside regular bot configs
- Share `MongoStorage` checkpointer across all managed bots for consistent memory

---

---

## Phase 5 — v1.5.0: Production Agent Upgrades
> **Priority: HIGH**
> **Estimated time: 3 weeks**

### 5.1 — LangGraph Checkpointer (Replaces Manual History Replay)

The current `load_bot()` replays chat history manually. This is fragile and slow.
The modern pattern is a **LangGraph checkpointer** — state is persisted automatically.

```python
# Old way (current — fragile)
trainer.load_bot(bot_id)  # replays all history from MongoDB

# New way — checkpointer handles everything
trainer.create_bot(bot_id, checkpointer="mongodb")  # or "sqlite", "postgres", "memory"
# thread_id maps to chat_id — state is automatically persisted and restored
```

**Supported backends:**
- `"memory"` — `InMemorySaver` (dev/testing)
- `"sqlite"` — `SqliteSaver` (local production)
- `"mongodb"` — Custom `MongoSaver` wrapping existing `MongoStorage`
- `"postgres"` — `PostgresSaver` from `langgraph-checkpoint-postgres`

### 5.2 — Structured Output Enforcement

```python
from pydantic import BaseModel

class SupportTicket(BaseModel):
    priority: str          # "low", "medium", "high", "critical"
    category: str          # "billing", "technical", "product"
    summary: str
    action_items: list[str]
    escalate: bool

trainer.create_bot(bot_id, agent_type="customer_support", output_schema=SupportTicket)

# get_response() now returns a typed Pydantic object, not a raw string
ticket = trainer.get_response("My payment failed...", bot_id, chat_id)
print(ticket.priority)    # "high"
print(ticket.escalate)    # True
```

Internally uses LangChain's `.with_structured_output()`.

### 5.3 — Human-in-the-Loop (HITL) Checkpoints

Pause agent mid-run, send to a human for approval, then resume.
Critical for legal, medical, and financial deployments.

```python
trainer.create_bot(bot_id, agent_type="financial", hitl=True,
                   hitl_callback=my_approval_function)

# When agent tries to execute a trade, it pauses and calls my_approval_function
# If approved → continues. If rejected → explains why it was rejected.
```

### 5.4 — Parallel Tool Execution

Current: tools execute sequentially. For research agents with 3+ tools, this is slow.
New: tool calls dispatched in parallel via LangGraph's `Send()` API.

```python
trainer.create_bot(bot_id, agent_type="research", parallel_tools=True)
# Wikipedia, Arxiv, Tavily all run simultaneously → merged and synthesized
```

---

---

## Phase 6 — v1.6.0: Native Observability (LongTracer Integration)
> **Priority: HIGH**
> **Estimated time: 2 weeks**
> **Dependencies:** LongTracer (`pip install longtracer`) already published

### The Problem (from the image in our discussion)

The comparison table shows LangChain and LlamaIndex both have LangSmith observability
(even though it's external). LongTrainer has `×` — nothing. This makes LongTrainer look
like a toy to enterprise engineers who need production monitoring.

### 6.1 — Automatic Hallucination Verification

Wire LongTracer's `CitationVerifier` into every `get_response()` call.

```python
trainer.create_bot(
    bot_id,
    enable_tracing=True,
    tracing_backend="sqlite"   # or "mongodb", "memory"
)

# get_response() now returns a rich tuple
answer, sources, trust_score, trace = trainer.get_response("...", bot_id, chat_id)
print(answer)              # "The Eiffel Tower is in Paris."
print(trust_score)         # 0.95  (0.0 = hallucination, 1.0 = fully grounded)
print(trace.hallucination_count)  # 0
print(trace.claims)        # [{claim: "...", verdict: "SUPPORTED", score: 0.95}]
```

**Optional extra pip dependency:**
```bash
pip install "longtrainer[tracing]"   # installs longtracer as optional dep
```

### 6.2 — Local Observability Dashboard

```bash
longtrainer dashboard --port 8001
# Opens: http://localhost:8001
```

Dashboard shows:
- All recent agent runs per bot
- Token count, latency, trust score per run
- Per-claim breakdown (which claims were hallucinated vs. supported)
- Tool call sequences and timing
- Project filter (by bot_id)
- Export trace to HTML or JSON

### 6.3 — LangSmith Cloud Integration (Optional)

```python
import os
os.environ["LANGSMITH_API_KEY"] = "ls__..."
os.environ["LANGSMITH_PROJECT"] = "my-longtrainer-project"

trainer = LongTrainer(..., enable_langsmith=True)
# Every run is automatically traced to LangSmith cloud
```

### 6.4 — Hooks and Callbacks

```python
trainer.create_bot(
    bot_id,
    on_response=lambda r: slack_notify(r),
    on_tool_call=lambda t: log_audit(t),
    on_hallucination=lambda h: alert_if_below(h.trust_score, threshold=0.7)
)
```

---

---

## Phase 7 — v1.7.0: Advanced Memory Architecture
> **Priority: MEDIUM-HIGH**
> **Estimated time: 3 weeks**

### The Problem (from the image)

The comparison table shows both LangChain and LlamaIndex have
`Semantic/Episodic/Procedural memory` ✅. LongTrainer has `×`.

Current LongTrainer memory = simple conversation buffer (short-term only).
2026 production agents need cross-session, typed memory.

### Three Memory Types (Based on Cognitive Science + LangGraph Docs)

#### 7.1 — Semantic Memory (Remembers Facts About Users)

```python
trainer.create_bot(bot_id, memory_type="semantic")

# Session 1: user says "My name is Mudassir, I work in fintech"
# Session 2 (new conversation): agent already knows name + industry
# Stored as structured key-value namespace per user_id in MongoDB
```

#### 7.2 — Episodic Memory (Remembers Past Successful Interactions)

```python
trainer.create_bot(bot_id, memory_type="episodic")

# Agent retrieves past Q&A pairs similar to current query
# Uses them as few-shot examples in-context
# Gets better at your specific use case over time
# Retrieval via vector similarity on past chat embeddings
```

#### 7.3 — Procedural Memory (Self-Improving System Prompt)

```python
trainer.create_bot(bot_id, memory_type="procedural", refine_on_feedback=True)

# User gives thumbs-down → agent's system prompt is refined
# Background async task — no downtime, no restart needed
# Example: repeatedly fails at JSON formatting → prompt auto-adds "always respond in JSON"
```

#### 7.4 — Composite Memory

```python
trainer.create_bot(bot_id, memory_type=["semantic", "episodic"])
# Mix and match memory types
```

---

---

## Phase 8 — v1.8.0: Production Hardening + Evaluation
> **Priority: MEDIUM**
> **Estimated time: 3 weeks**

### The Problem (from the image)

`Evaluation & regression testing` — LangChain has it via LangSmith ✅.
LlamaIndex has it ✅. LongTrainer has `×`.

### 8.1 — Built-in Evaluation Suite

```python
results = trainer.evaluate_bot(
    bot_id,
    test_cases=[
        {
            "question": "What is the return policy?",
            "expected_keywords": ["30 days", "receipt"],
            "expected_sources": ["policy.pdf"]
        },
        {
            "question": "How do I cancel my subscription?",
            "expected_keywords": ["account settings", "billing"],
        },
    ]
)

print(results.pass_rate)         # 0.85  (85% of tests passed)
print(results.avg_trust_score)   # 0.91
print(results.latency_p95_ms)    # 1240
print(results.per_case)          # [{pass: True, got: "...", latency_ms: 890}, ...]
```

### 8.2 — Rate Limiting & Cost Controls

```python
trainer.create_bot(
    bot_id,
    max_tokens_per_day=100_000,
    max_requests_per_minute=30
)
# Raises LongTrainerQuotaError when limits hit
# Token usage tracked per bot_id in MongoDB
# Reset daily at midnight UTC
```

### 8.3 — Webhook & Event System

```python
trainer.create_bot(
    bot_id,
    on_response=my_slack_alert,
    on_tool_call=my_audit_log,
    on_quota_exceeded=my_billing_handler,
    on_hallucination=lambda h: alert_if_below(h.trust_score, 0.7)
)
```

### 8.4 — Cross-Bot Retrieval

```python
# Query multiple bots' vector stores simultaneously and merge results
docs = trainer.cross_bot_search(
    query="What is the refund policy?",
    bot_ids=[policy_bot, faq_bot, legal_bot]
)
```

---

---

## Phase 9 — v1.9.0: MCP (Model Context Protocol) Integration
> **Priority: MEDIUM (after agents are solid)**
> **Estimated time: 3–4 weeks**

### What MCP Is (Plain English)

MCP = a "USB-C standard for AI tools." Any AI app that "speaks MCP"
(Claude Desktop, Cursor, VS Code Copilot, your own app) can connect to any
MCP server and use its tools with zero custom integration code.

There are now **hundreds of community MCP servers**: GitHub, Gmail, Notion, Slack,
databases, file systems, web browsers, etc.

### Part A — MCP Client (LongTrainer connects to MCP servers)

Give your bot access to ANY MCP-compatible service with zero tool definition code:

```python
trainer.create_bot(
    bot_id,
    agent_mode=True,
    mcp_servers=[
        "https://github.example/mcp",    # GitHub MCP server
        "https://notion.example/mcp",    # Notion MCP server
        "http://localhost:3001/mcp",     # Your own custom MCP server
    ]
)
# LongTrainer auto-fetches all tools from every MCP server
# Injects them into the agent's tool registry
# Your bot can now read GitHub issues, write Notion pages — automatically
```

### Part B — MCP Host (LongTrainer IS the MCP server)

This is the killer feature. LongTrainer exposes itself as an MCP server,
making your private knowledge base available inside Claude Desktop, Cursor, VS Code.

```bash
longtrainer server mcp-start --port 3001
```

Exposed MCP tools:
- `query_bot(bot_id, question)` — query any LongTrainer bot
- `add_document(path, bot_id)` — add a document to any bot
- `list_bots()` — list all registered bots

Now users can ask Claude Desktop: *"Search our company knowledge base for the refund policy"*
and it calls your LongTrainer MCP server automatically.

### Part C — mcp-use Library Integration

`mcp-use` is the Python library for connecting LangChain agents to multiple MCP servers.
LongTrainer wraps it behind the `mcp_servers=[]` kwarg cleanly:

```python
trainer.create_bot(
    bot_id,
    mcp_servers=["https://slack.example/mcp"],
    mcp_client="langchain"   # or "llamaindex"
)
```

---

---

## Phase 10 — v2.0.0: Full Suite Integration + Grand Launch 🚀
> **Priority: Final milestone**
> **Estimated time: 2 weeks**
> **Dependencies:** All Phases 3–9 complete

### Native LongParser Integration

```python
# LongParser's enterprise pipeline (HITL + OCR + HybridChunker) becomes
# available directly inside add_document_from_path()
trainer.add_document_from_path(
    "earnings_report.pdf",
    bot_id,
    use_longparser=True,
    hitl_review=True,        # pause for human approval before embedding
    ocr_backend="pix2tex",   # LaTeX OCR for equations/formulas
    chunk_strategy="hybrid"  # token-aware hierarchical chunking
)

# HITL review flow
job_id = trainer.add_document_from_path("doc.pdf", bot_id, use_longparser=True, hitl_review=True)
trainer.review_document(job_id, action="approve_all")  # or "reject", chunk_ids=[3, 7]
trainer.finalize_document(job_id, bot_id)
```

**Optional extra pip dependency:**
```bash
pip install "longtrainer[parser]"   # installs longparser as optional dep
```

### The v2.0.0 Marketing Pitch

> **"The only open-source framework that handles the complete lifecycle
> of a production RAG agent: Parse → Chunk → Embed → Index → Route →
> Agent → Verify → Trace → Monitor — all in one `pip install longtrainer`."**

**vs. LangChain:** "We give the high-level abstractions LangChain doesn't provide."
**vs. LlamaIndex:** "We add hallucination detection and HITL parsing LlamaIndex doesn't have."
**vs. LangSmith:** "Our observability is open-source, local-first, and free."
**vs. OpenAI Assistants:** "We're provider-agnostic and you own 100% of your data."

---

---

## 📊 Complete Feature Gap Table

| Feature | LangChain | LlamaIndex | LongTrainer Now | LongTrainer v2.0 |
|---|---|---|---|---|
| RAG pipelines | ✅ | ✅ | ✅ | ✅ |
| 9 Vector DB providers | ✅ | ✅ | ✅ | ✅ |
| 12+ Document loaders | ✅ | ✅ | ✅ | ✅ |
| CLI + REST API server | ❌ raw | ❌ raw | ✅ | ✅ |
| Dynamic tool injection | Partial | Partial | ✅ | ✅ |
| Vision chat | ✅ | ✅ | ✅ | ✅ |
| Fernet encryption | ❌ | ❌ | ✅ | ✅ |
| Named agent types | ✅ | ✅ | ❌ | ✅ Phase 3 |
| Multi-agent Supervisor | ✅ | ✅ | ❌ | ✅ Phase 4 |
| Multi-agent Swarm | ✅ | ✅ | ❌ | ✅ Phase 4 |
| LangGraph checkpointer | ✅ | Partial | ❌ | ✅ Phase 5 |
| Structured output | ✅ | ✅ | ❌ | ✅ Phase 5 |
| HITL agent pauses | ✅ | ✅ | ❌ | ✅ Phase 5 |
| Parallel tool execution | ✅ | ✅ | ❌ | ✅ Phase 5 |
| LangSmith-style observability | ❌ external | ❌ external | ❌ | ✅ Phase 6 |
| Hallucination detection | ❌ | ❌ | ❌ (LongTracer separate) | ✅ Phase 6 |
| Semantic memory | ✅ | ✅ | ❌ | ✅ Phase 7 |
| Episodic memory | ✅ | ✅ | ❌ | ✅ Phase 7 |
| Procedural memory | ✅ | ✅ | ❌ | ✅ Phase 7 |
| Evaluation & regression testing | ✅ LangSmith | ✅ | ❌ | ✅ Phase 8 |
| Rate limiting & cost controls | ❌ | ❌ | ❌ | ✅ Phase 8 |
| MCP client integration | Partial | ✅ | ❌ | ✅ Phase 9 |
| MCP host (expose as server) | ❌ | ❌ | ❌ | ✅ Phase 9 |
| HITL document parsing | ❌ | Partial | ❌ (LongParser separate) | ✅ Phase 10 |

---

## 📅 Version Timeline

| Version | Phase | Key Feature | Estimated ETA |
|---|---|---|---|
| **v1.3.0** | Named Agent Types | `agent_type="research/sql/coding/financial/support"` | 2–3 weeks |
| **v1.4.0** | Multi-Agent Systems | Supervisor + Swarm + Nested hierarchies | +3–4 weeks |
| **v1.5.0** | Production Agent Upgrades | Checkpointers, structured output, HITL, parallel tools | +3 weeks |
| **v1.6.0** | Native Observability | LongTracer integration + local dashboard + LangSmith | +2 weeks |
| **v1.7.0** | Advanced Memory | Semantic/Episodic/Procedural memory types | +3 weeks |
| **v1.8.0** | Production Hardening | Evaluation suite, rate limiting, webhooks | +3 weeks |
| **v1.9.0** | MCP Integration | MCP client + MCP host + mcp-use wrapper | +3–4 weeks |
| **v2.0.0** | Grand Launch | LongParser native integration + full suite | +2 weeks |

**Total estimated time to v2.0.0: ~20–24 weeks (5–6 months)**

---

## 🎯 Immediate Actions (Before Phase 3 Code Work)

1. **Fix `update_chatbot()` bug** — `bot["faiss_path"]` → `bot["db_path"]` in `trainer.py:L611`
2. **Add async `add_document_from_path()`** — needed before parallel doc ingestion
3. **Add retrieval confidence scores** — `invoke_vectorstore()` should return `(doc, score)` tuples
4. **Write tests for Phase 3 agent types** — `tests/test_09_phase3.py` already exists as a scaffold, expand it
5. **Tag v1.2.0 as stable** on PyPI and GitHub before starting Phase 3

---

## 🏗️ Implementation File Map

```
longtrainer/
├── trainer.py                  ← Add create_supervisor(), create_swarm(), evaluate_bot()
├── bot.py                      ← Add StructuredOutputBot, HITLBot
├── agent_types/                ← NEW MODULE (Phase 3)
│   ├── __init__.py
│   ├── base.py
│   ├── research.py
│   ├── sql.py
│   ├── coding.py
│   ├── financial.py
│   └── customer_support.py
├── multi_agent/                ← NEW MODULE (Phase 4)
│   ├── __init__.py
│   ├── supervisor.py
│   └── swarm.py
├── checkpointers/              ← NEW MODULE (Phase 5)
│   ├── __init__.py
│   ├── mongo_saver.py
│   └── sqlite_saver.py
├── memory/                     ← NEW MODULE (Phase 7)
│   ├── __init__.py
│   ├── semantic.py
│   ├── episodic.py
│   └── procedural.py
├── observability/              ← NEW MODULE (Phase 6)
│   ├── __init__.py
│   ├── tracer_bridge.py        ← wires LongTracer into get_response()
│   └── dashboard.py            ← FastAPI local dashboard server
├── evaluation/                 ← NEW MODULE (Phase 8)
│   ├── __init__.py
│   └── harness.py
└── mcp/                        ← NEW MODULE (Phase 9)
    ├── __init__.py
    ├── client.py               ← MCP client (connect to external servers)
    └── host.py                 ← MCP host (expose LongTrainer as MCP server)
```

---

*Document created: April 9, 2026*
*Version: 1.0*
*Author: LongTrainer Team — ENDEVSOLS*
