"""LongTrainer REST API — FastAPI server.

Start with:
    longtrainer serve
    # or directly:
    uvicorn longtrainer.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import uuid

import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# P3-7: Rate limiting — graceful fallback if slowapi is not installed
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False

from longtrainer import __version__

# ─── Global trainer instance ─────────────────────────────────────────────────

_trainer = None


def _get_trainer():
    """Get or lazily create the LongTrainer instance from config."""
    global _trainer
    if _trainer is not None:
        return _trainer

    config_path = os.environ.get("LONGTRAINER_CONFIG", "longtrainer.yaml")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    from longtrainer.trainer import LongTrainer

    _trainer = LongTrainer(
        mongo_endpoint=cfg.get("mongo_endpoint", "mongodb://localhost:27017/"),
        llm_provider=cfg.get("llm", {}).get("provider", "openai"),
        default_llm=cfg.get("llm", {}).get("model_name", "gpt-4o-2024-08-06"),
        embedding_provider=cfg.get("embedding", {}).get("provider", "openai"),
        embedding_model_name=cfg.get("embedding", {}).get("model_name", "text-embedding-3-small"),
        vector_store_provider=cfg.get("vector_store", {}).get("provider", "faiss"),
        vector_store_kwargs=cfg.get("vector_store", {}).get("kwargs", {}),
        chunk_size=cfg.get("chunking", {}).get("chunk_size", 2048),
        chunk_overlap=cfg.get("chunking", {}).get("chunk_overlap", 200),
        encrypt_chats=cfg.get("encrypt_chats", False),
    )
    return _trainer


def _ensure_bot_loaded(trainer, bot_id: str) -> None:
    """Load bot from MongoDB if not already in local cache.

    MongoDB is the source of truth. self.bot_data is a warm per-process cache.
    On any miss → load from Mongo, never the reverse.
    """
    if bot_id not in trainer.bot_data:
        trainer.load_bot(bot_id)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """App lifespan — trainer is lazy-initialized on first API call."""
    yield


# ─── H2: Tenant-Aware API Key Authentication ─────────────────────────────────

_AUTH_ENABLED = bool(os.environ.get("LONGTRAINER_API_KEY"))


async def _authenticate(request: Request, x_api_key: Optional[str] = Header(default=None)):
    """FastAPI dependency: validate API key and extract tenant_id.

    When LONGTRAINER_API_KEY env var is set, auth is required.
    If an `api_keys` collection exists in MongoDB with {key, tenant_id},
    the tenant_id is extracted and injected into request.state.
    Otherwise falls back to a default tenant.

    When LONGTRAINER_API_KEY is NOT set, auth is disabled (backwards-compatible).
    """
    if not _AUTH_ENABLED:
        request.state.tenant_id = "default"
        return

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    # Check MongoDB api_keys collection for tenant mapping
    trainer = _get_trainer()
    api_key_doc = trainer.db["api_keys"].find_one({"key": x_api_key})

    if api_key_doc:
        request.state.tenant_id = api_key_doc.get("tenant_id", "default")
        return

    # Fallback: check against the global env var key
    if x_api_key == os.environ.get("LONGTRAINER_API_KEY"):
        request.state.tenant_id = "default"
        return

    raise HTTPException(status_code=403, detail="Invalid API key.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LongTrainer API",
    version=__version__,
    description="Production-Ready RAG Framework REST API",
    lifespan=lifespan,
    dependencies=[Depends(_authenticate)],
)

# P3-7: Rate limiting — in-memory by default, Redis via config
if _SLOWAPI_AVAILABLE:
    _rate_limit_storage = os.environ.get("LONGTRAINER_RATE_LIMIT_STORAGE", "memory://")
    limiter = Limiter(key_func=get_remote_address, storage_uri=_rate_limit_storage)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    import logging as _log
    _log.getLogger(__name__).warning(
        "slowapi not installed — rate limiting disabled. Install with: pip install slowapi"
    )
    # No-op limiter so @limiter.limit() decorators don't crash
    class _NoOpLimiter:
        def limit(self, *a, **kw):
            def decorator(func):
                return func
            return decorator
    limiter = _NoOpLimiter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────

class CreateBotRequest(BaseModel):
    prompt_template: Optional[str] = None
    agent_mode: bool = False
    tools: Optional[list[str]] = None


class DocumentPathRequest(BaseModel):
    path: str
    use_unstructured: bool = False


class DocumentLinkRequest(BaseModel):
    links: list[str]


class DocumentQueryRequest(BaseModel):
    search_query: str


class ChatRequest(BaseModel):
    query: str
    stream: bool = False
    web_search: bool = False
    uploaded_files: Optional[list[dict]] = None


class VisionChatRequest(BaseModel):
    query: str
    image_paths: list[str]
    web_search: bool = False
    uploaded_files: Optional[list[dict]] = None


class PromptTemplateRequest(BaseModel):
    prompt_template: str


class VectorSearchRequest(BaseModel):
    query: str




# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


# ─── Bot Lifecycle ────────────────────────────────────────────────────────────

@app.post("/bots")
async def create_bot_id():
    """Initialize a new bot and return its ID."""
    trainer = _get_trainer()
    bot_id = trainer.initialize_bot_id()
    if not bot_id:
        raise HTTPException(status_code=500, detail="Failed to create bot.")
    return {"bot_id": bot_id}


@app.post("/bots/{bot_id}/build")
async def build_bot(bot_id: str, req: CreateBotRequest):
    """Build a bot from its loaded documents."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        trainer.create_bot(
            bot_id=bot_id,
            prompt_template=req.prompt_template,
            agent_mode=req.agent_mode,
            tools=req.tools,
        )
        return {"status": "ok", "bot_id": bot_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/load")
async def load_bot(bot_id: str):
    """Load an existing bot from MongoDB and FAISS."""
    trainer = _get_trainer()
    try:
        trainer.load_bot(bot_id)
        return {"status": "ok", "bot_id": bot_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/bots/{bot_id}")
async def delete_bot(bot_id: str):
    """Delete a bot and all associated data."""
    trainer = _get_trainer()
    try:
        trainer.delete_chatbot(bot_id)
        return {"status": "deleted", "bot_id": bot_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))




# ─── Documents ────────────────────────────────────────────────────────────────

@app.post("/bots/{bot_id}/documents/path")
async def add_document_path(bot_id: str, req: DocumentPathRequest, background_tasks: BackgroundTasks):
    """Add a document from a local file path (async with job tracking).

    H4: Returns immediately with a job_id. Use GET /jobs/{job_id} to poll status.
    """
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)

    job_id = str(uuid.uuid4())
    trainer._storage.create_job(job_id, bot_id, "document_ingest")

    def _ingest(j_id: str, b_id: str, path: str, use_unstructured: bool):
        try:
            trainer._storage.update_job_status(j_id, "processing")
            trainer.add_document_from_path(path, b_id, use_unstructured)
            trainer._storage.update_job_status(j_id, "success")
        except Exception as exc:
            trainer._storage.update_job_status(j_id, "failed", error=str(exc))

    background_tasks.add_task(_ingest, job_id, bot_id, req.path, req.use_unstructured)
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of an async job."""
    trainer = _get_trainer()
    job = trainer._storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    job.pop("_id", None)
    return job


@app.post("/bots/{bot_id}/documents/link")
async def add_document_link(bot_id: str, req: DocumentLinkRequest):
    """Add documents from web links."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        trainer.add_document_from_link(req.links, bot_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/documents/query")
async def add_document_query(bot_id: str, req: DocumentQueryRequest):
    """Add documents from a Wikipedia search."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        trainer.add_document_from_query(req.search_query, bot_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Chats ────────────────────────────────────────────────────────────────────

@app.post("/bots/{bot_id}/chats")
async def new_chat(bot_id: str):
    """Create a new chat session."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        chat_id = trainer.new_chat(bot_id)
        if not chat_id:
            raise HTTPException(status_code=500, detail="Failed to create chat.")
        return {"chat_id": chat_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/bots/{bot_id}/chats")
async def list_chats(bot_id: str):
    """List all chat session IDs for a bot."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    return trainer.list_chats(bot_id)


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str, order: str = "newest"):
    """Get full chat history for a session."""
    trainer = _get_trainer()
    data = trainer.get_chat_by_id(chat_id, order)
    if data is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    # Remove MongoDB _id for JSON serialization
    for item in data:
        item.pop("_id", None)
    return {"messages": data}


@app.post("/bots/{bot_id}/chats/{chat_id}")
async def chat(bot_id: str, chat_id: str, req: ChatRequest):
    """Send a message and get a response."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)

    if req.stream:
        async def stream_gen():
            async for chunk in trainer.aget_response(
                query=req.query,
                bot_id=bot_id,
                chat_id=chat_id,
                uploaded_files=req.uploaded_files,
                web_search=req.web_search,
            ):
                yield chunk

        return StreamingResponse(stream_gen(), media_type="text/plain")

    try:
        answer, web_sources = trainer.get_response(
            query=req.query,
            bot_id=bot_id,
            chat_id=chat_id,
            stream=False,
            uploaded_files=req.uploaded_files,
            web_search=req.web_search,
        )
        return {"answer": answer, "web_sources": web_sources}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Vision Chats ─────────────────────────────────────────────────────────────

@app.post("/bots/{bot_id}/vision-chats")
async def new_vision_chat(bot_id: str):
    """Create a new vision chat session."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        vision_chat_id = trainer.new_vision_chat(bot_id)
        if not vision_chat_id:
            raise HTTPException(status_code=500, detail="Failed to create vision chat.")
        return {"vision_chat_id": vision_chat_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/bots/{bot_id}/vision-chats/{vision_chat_id}")
async def vision_chat(bot_id: str, vision_chat_id: str, req: VisionChatRequest):
    """Send a vision query with images and get a response."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        response, web_sources = trainer.get_vision_response(
            query=req.query,
            image_paths=req.image_paths,
            bot_id=bot_id,
            vision_chat_id=vision_chat_id,
            uploaded_files=req.uploaded_files,
            web_search=req.web_search,
        )
        return {"response": response, "web_sources": web_sources}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Utilities ────────────────────────────────────────────────────────────────

@app.put("/bots/{bot_id}/prompt")
async def set_prompt(bot_id: str, req: PromptTemplateRequest):
    """Update the system prompt for a bot."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        trainer.set_custom_prompt_template(bot_id, req.prompt_template)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/vectorstore")
async def search_vectorstore(bot_id: str, req: VectorSearchRequest):
    """Search the vector store directly."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        docs = trainer.invoke_vectorstore(bot_id, req.query)
        return {"documents": [{"content": d.page_content, "metadata": d.metadata} for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/train-chats")
async def train_on_chats(bot_id: str):
    """Train the bot on its unprocessed chat history."""
    trainer = _get_trainer()
    _ensure_bot_loaded(trainer, bot_id)
    try:
        result = trainer.train_chats(bot_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


