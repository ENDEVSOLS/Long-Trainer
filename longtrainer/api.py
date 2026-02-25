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

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


@asynccontextmanager
async def lifespan(application: FastAPI):
    """App lifespan — trainer is lazy-initialized on first API call."""
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LongTrainer API",
    version=__version__,
    description="Production-Ready RAG Framework REST API",
    lifespan=lifespan,
)

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
async def add_document_path(bot_id: str, req: DocumentPathRequest):
    """Add a document from a local file path."""
    trainer = _get_trainer()
    try:
        trainer.add_document_from_path(req.path, bot_id, req.use_unstructured)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/documents/link")
async def add_document_link(bot_id: str, req: DocumentLinkRequest):
    """Add documents from web links."""
    trainer = _get_trainer()
    try:
        trainer.add_document_from_link(req.links, bot_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/documents/query")
async def add_document_query(bot_id: str, req: DocumentQueryRequest):
    """Add documents from a Wikipedia search."""
    trainer = _get_trainer()
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
    try:
        trainer.set_custom_prompt_template(bot_id, req.prompt_template)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/vectorstore")
async def search_vectorstore(bot_id: str, req: VectorSearchRequest):
    """Search the vector store directly."""
    trainer = _get_trainer()
    try:
        docs = trainer.invoke_vectorstore(bot_id, req.query)
        return {"documents": [{"content": d.page_content, "metadata": d.metadata} for d in docs]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bots/{bot_id}/train-chats")
async def train_on_chats(bot_id: str):
    """Train the bot on its unprocessed chat history."""
    trainer = _get_trainer()
    try:
        result = trainer.train_chats(bot_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
