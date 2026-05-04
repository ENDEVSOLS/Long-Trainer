# Observability & Hallucination Detection

LongTrainer v1.3.0 introduces native integration with [LongTracer](https://github.com/ENDEVSOLS/LongTracer) — an observability and hallucination detection SDK for RAG pipelines.

When enabled, LongTracer automatically captures performance spans, latencies, and token counts for every LLM call and retriever query. Optionally, it can run **CitationVerifier** — a hybrid STS + NLI claim verification engine — to detect hallucinated responses in real time.

---

## Installation

LongTracer is an optional dependency. Install it with:

```bash
pip install longtrainer[tracer]
```

This installs `longtracer` with its LangChain, LangGraph, and MongoDB adapters.

!!! note
    LongTracer requires a running MongoDB instance. By default, it reuses the same MongoDB endpoint you pass to LongTrainer and stores trace data in a separate `longtracer` database.

---

## Quick Start

Enable tracing with a single flag:

```python
from longtrainer import LongTrainer

trainer = LongTrainer(
    mongo_endpoint="mongodb://localhost:27017/",
    enable_tracer=True,
)
```

That's it. All RAG, Agent, Vision, and Structured output responses will now be traced automatically.

---

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_tracer` | `bool` | `False` | Enable LongTracer integration |
| `tracer_backend` | `str` | `"mongo"` | Trace storage backend (`"mongo"`, `"sqlite"`, `"memory"`) |
| `tracer_verbose` | `bool` | `False` | Print per-span summaries to console |
| `tracer_verify` | `bool` | `True` | Run CitationVerifier for hallucination detection |
| `tracer_threshold` | `float` | `0.5` | Confidence threshold for hallucination flagging (0.0–1.0) |

### Full Example

```python
trainer = LongTrainer(
    mongo_endpoint="mongodb://localhost:27017/",
    enable_tracer=True,
    tracer_backend="mongo",
    tracer_verbose=True,       # Print spans to console
    tracer_verify=True,        # Enable hallucination detection
    tracer_threshold=0.5,      # Flag claims below 50% confidence
)
```

---

## How It Works

LongTracer instruments all five response paths in LongTrainer:

### RAG Responses (`get_response`)

Automatically captures:

- **Retrieval span** — documents retrieved, latency
- **LLM span** — prompt, model, response, latency
- **Claim verification** — splits the response into individual claims and checks each against retrieved source documents using NLI

### Agent Responses (`get_response` with `agent_mode=True`)

Uses the `LongTracerAgentHandler` which automatically:

- Creates a root trace for the entire agent invocation
- Captures each tool call as a child span
- Logs the final response

### Streaming Responses (`get_response(stream=True)` / `aget_response`)

Traces are captured the same way as standard responses. The trace root is closed in a `finally` block after the stream completes.

### Vision Responses (`get_vision_response`)

Since vision pipelines don't use LCEL chains, tracing is handled **post-hoc**:

- A **retrieval span** captures the context documents
- An **LLM span** captures the vision model response
- A **grounding span** runs CitationVerifier against the retrieved sources (if `tracer_verify=True`)

### Structured Output (`get_response(schema={...})`)

Since `invoke_structured` bypasses the LCEL chain, manual span wrapping captures:

- The structured output call as a single span
- The result status and a preview of the response

---

## Lightweight Tracing (Without NLI)

If you want observability (spans, latencies, token counts) **without** the ~500MB NLI model download, disable verification:

```python
trainer = LongTrainer(
    enable_tracer=True,
    tracer_verify=False,  # Spans only — no NLI model download
)
```

!!! warning
    When `tracer_verify=False`, the CitationVerifier is skipped for Vision and Structured output paths. For RAG and Agent paths, the LongTracer callback handler manages verification internally — the `tracer_verify` flag does not override it.

---

## Graceful Degradation

LongTracer is a fully optional dependency. The system handles all failure scenarios gracefully:

| Scenario | Behavior |
|---|---|
| `enable_tracer=False` (default) | Zero `longtracer` imports — zero overhead |
| `enable_tracer=True` + `longtracer` not installed | Prints a warning, continues normally |
| `enable_tracer=True` + runtime error in tracer | Catches exception, logs warning, continues |
| Tracer crash mid-response | `try/except` in `finally` blocks ensures response is still returned |

---

## Trace Storage

Traces are stored in a separate MongoDB database called `longtracer` (not `longtrainer_db`). LongTracer automatically sets the `MONGODB_URI` environment variable from your `mongo_endpoint`.

You can query traces directly:

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["longtracer"]

# Find all traces for a specific bot
traces = db.runs.find({"inputs.bot_id": "bot-xxx"})
for trace in traces:
    print(trace["inputs"]["chat_type"], trace["outputs"])
```

---

## Architecture Notes

- **Single Project:** All traces go to the LongTracer "default" project. Bot and chat identification is done via metadata fields (`bot_id`, `chat_id`, `chat_type`) on each trace.
- **Singleton Init:** `LongTracer.init()` is called exactly once at `LongTrainer.__init__()` time. Subsequent calls are no-ops.
- **Private API Usage:** The integration uses `tracer._safe_update_run()` (a private method) to inject metadata into agent traces. This is wrapped in `try/except` to prevent crashes if the internal API changes.
