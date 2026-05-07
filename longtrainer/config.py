"""Configuration models for LongTrainer.

Provides validated configuration with sensible defaults using Pydantic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


_DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent assistant named LongTrainer. "
    "Your purpose is to answer all kind of queries and interact with the user "
    "in a helpful and conversational manner.\n"
    "{context}\n"
    "Use the following information to respond to the user's question. "
    "If the answer is unknown, admit it rather than fabricating a response. "
    "Avoid unnecessary details or irrelevant explanations.\n"
    "Responses should be direct, professional, and focused solely on the user's query."
)


class LongTrainerConfig(BaseModel):
    """Global configuration for a LongTrainer instance.

    Attributes:
        mongo_endpoint: MongoDB connection string.
        prompt_template: Default system prompt template for all bots.
        max_token_limit: Token buffer limit for conversation memory.
        num_k: Number of documents to retrieve per query.
        chunk_size: Text splitter chunk size.
        chunk_overlap: Text splitter overlap size.
        ensemble: Enable ensemble retriever (FAISS + MultiQuery).
        encrypt_chats: Enable Fernet encryption for stored chats.
        encryption_key: Fernet key bytes (auto-generated if not provided).
        enable_tracer: Enable LongTracer integration for tracing and verification.
        tracer_backend: LongTracer backend ("mongo", "sqlite", "memory").
        tracer_verbose: Print per-span summaries to console.
        tracer_verify: Run CitationVerifier for hallucination detection.
            False = spans only, no NLI model download (~500MB).
        tracer_threshold: Hallucination detection threshold (0.0–1.0).
    """

    mongo_endpoint: str = "mongodb://localhost:27017/"
    llm_provider: str = "openai"
    default_llm: str = "gpt-4o-2024-08-06"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    vector_store_provider: str = "faiss"
    vector_store_kwargs: dict = Field(default_factory=dict)
    prompt_template: str = _DEFAULT_SYSTEM_PROMPT
    max_token_limit: int = 32000
    num_k: int = 3
    chunk_size: int = 2048
    chunk_overlap: int = 200
    ensemble: bool = False
    encrypt_chats: bool = False
    encryption_key: Optional[bytes] = None

    # LongTracer integration
    enable_tracer: bool = False
    tracer_backend: str = "mongo"
    tracer_verbose: bool = False
    tracer_verify: bool = True
    tracer_threshold: float = 0.5

    # Rate limiting
    rate_limit_enabled: bool = False
    rate_limit_llm_rpm: int = 60
    rate_limit_embedding_rpm: int = 120
    rate_limit_tool_rpm: int = 30
    rate_limit_ingestion_rpm: int = 10

    model_config = {"arbitrary_types_allowed": True}
