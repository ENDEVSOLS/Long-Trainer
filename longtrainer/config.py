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
    """

    mongo_endpoint: str = "mongodb://localhost:27017/"
    prompt_template: str = _DEFAULT_SYSTEM_PROMPT
    max_token_limit: int = 32000
    num_k: int = 3
    chunk_size: int = 2048
    chunk_overlap: int = 200
    ensemble: bool = False
    encrypt_chats: bool = False
    encryption_key: Optional[bytes] = None

    model_config = {"arbitrary_types_allowed": True}
