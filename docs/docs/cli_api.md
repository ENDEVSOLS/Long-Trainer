# Zero-Code CLI & API Server

LongTrainer 1.1.0 introduces a built-in CLI and FastAPI REST server. This transforms LongTrainer from a Python library into a standalone, language-agnostic document Q&A service.

You can now chat with your documents from Javascript, Go, mobile apps, or no-code tools without writing any Python backend code.

---

## 1. The CLI (`longtrainer`)

Once installed via `pip install longtrainer[api,cli]`, you get access to the global `longtrainer` command.

### `longtrainer init`

Scaffolds a new LongTrainer project interactively.

```bash
longtrainer init
```

This will prompt you for your MongoDB URI, LLM provider, embedding model, and chunking preferences. It generates a `longtrainer.yaml` configuration file in your current directory.

You can also run it non-interactively:

```bash
longtrainer init --mongo "mongodb://localhost:27017/" --llm-provider openai --model-name gpt-4o --output my_config.yaml
```

### `longtrainer serve`

Starts the built-in FastAPI server using `uvicorn`.

```bash
longtrainer serve
```

By default, it reads `longtrainer.yaml` from your current directory and starts the server on `http://0.0.0.0:8000`.

Options:
- `-c, --config PATH`: Path to a specific yaml config.
- `--host TEXT`: Override the host.
- `-p, --port INTEGER`: Override the port.
- `--reload`: Enable auto-reload for development.

---

## 2. The REST API

When you run `longtrainer serve`, it exposes a full REST API with 16 endpoints. 

You can view the interactive **Swagger UI Documentation** by navigating to:
👉 `http://localhost:8000/docs`

### Lazy Initialization (No API Key needed at startup)

The server starts up instantly. It does **not** connect to the LLM or check your `OPENAI_API_KEY` until the first API request that actually requires it is made. This means `/health` will always work, even if your environment variables aren't set up yet.

### Key API Endpoints

Below is a quick reference of the available endpoints. All API requests process JSON and return JSON.

#### Health
* `GET /health` — Check if the server is running.

#### Bot Management
* `POST /bots` — Initialize a new bot and return a `bot_id`.
* `POST /bots/{bot_id}/build` — Build the bot (spin up the FAISS index).
* `POST /bots/{bot_id}/load` — Load an existing bot from MongoDB into memory.
* `DELETE /bots/{bot_id}` — Wipe a bot and all its data.

#### Document Ingestion
* `POST /bots/{bot_id}/documents/path` — Ingest a local file path (`{"path": "/tmp/data.pdf"}`).
* `POST /bots/{bot_id}/documents/link` — Ingest URLs (`{"links": ["https://example.com"]}`).
* `POST /bots/{bot_id}/documents/query` — Ingest from a Wikipedia search (`{"search_query": "Python"}`).

#### Chat & Sessions
* `POST /bots/{bot_id}/chats` — Create a new chat session, returns a `chat_id`.
* `GET /bots/{bot_id}/chats` — List all chat session IDs for this bot.
* `POST /bots/{bot_id}/chats/{chat_id}` — Send a message to the bot.
  ```json
  {
    "query": "What does the document say?",
    "stream": false,
    "web_search": false
  }
  ```
  *Note: If `"stream": true` is passed, the endpoint returns an SSE (Server-Sent Events) streaming response.*
* `GET /chats/{chat_id}` — Retrieve the full conversation history.

#### Vision Chat
* `POST /bots/{bot_id}/vision-chats` — Create a new vision chat session.
* `POST /bots/{bot_id}/vision-chats/{vision_chat_id}` — Ask a question about attached images.

#### Utilities
* `POST /bots/{bot_id}/vectorstore` — Directly search the FAISS index without querying the LLM.
* `PUT /bots/{bot_id}/prompt` — Update the system prompt for a bot.
* `POST /bots/{bot_id}/train-chats` — Trigger self-improvement (train the bot on its past Q&A).
