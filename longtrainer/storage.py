"""MongoDB storage and chat encryption for LongTrainer.

Manages all database interactions: bot persistence, chat history,
document storage, and optional Fernet encryption.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from cryptography.fernet import Fernet
from pymongo import MongoClient

from longtrainer.config import LongTrainerConfig


class MongoStorage:
    """Manages MongoDB connections and all database operations.

    Args:
        config: A LongTrainerConfig instance with connection settings.
    """

    def __init__(self, config: LongTrainerConfig) -> None:
        self.client = MongoClient(
            config.mongo_endpoint,
            maxPoolSize=200,        # Double default (100) for concurrent tasks + API load
            minPoolSize=20,         # Keep connections warm to avoid cold-start latency
            maxIdleTimeMS=60_000,   # Drop stale connections after 60s
        )

        # H1: Fail fast if MongoDB is unreachable
        try:
            self.client.admin.command("ping")
        except Exception as e:
            raise ConnectionError(
                f"MongoDB unreachable at {config.mongo_endpoint}: {e}"
            ) from e

        self.db = self.client["longtrainer_db"]

        # Collections
        self.bots = self.db["bots"]
        self.chats = self.db["chats"]
        self.vision_chats = self.db["vision_chats"]
        self.documents_collection = self.db["documents"]

        # Ensure compound indexes exist for all hot query paths
        self._ensure_indexes()

        # Encryption
        self.encrypt_chats = config.encrypt_chats
        self._fernet: Optional[Fernet] = None
        self.encryption_key: Optional[bytes] = None
        if config.encrypt_chats:
            self.encryption_key = config.encryption_key or Fernet.generate_key()
            self._fernet = Fernet(self.encryption_key)

    # ─── Index Initialization ─────────────────────────────────────────────────

    def _ensure_indexes(self) -> None:
        """Create compound indexes for all hot query paths.

        Called once at startup. MongoDB create_index is idempotent — safe to
        call on every restart. background=True avoids blocking reads/writes
        on large existing collections.

        Unique indexes use partialFilterExpression to skip documents where
        the key field doesn't exist — prevents DuplicateKeyError on legacy
        data that was inserted before the field was introduced.
        """
        from pymongo.errors import OperationFailure

        # documents: (bot_id, indexed) — used by H3 delta ingestion,
        # P3-2 live count, and P3-3 embedding model lock guard
        self.documents_collection.create_index(
            [("bot_id", 1), ("indexed", 1)], background=True
        )
        # documents: (bot_id, doc_hash) — used by H3 deduplication
        # partialFilterExpression: only index docs where doc_hash exists
        # (legacy docs without doc_hash are skipped — no DuplicateKeyError)
        try:
            self.documents_collection.create_index(
                [("bot_id", 1), ("doc_hash", 1)],
                unique=True,
                background=True,
                partialFilterExpression={"doc_hash": {"$type": "string"}},
            )
        except OperationFailure:
            # If old index with different options exists, drop and recreate
            try:
                self.documents_collection.drop_index("bot_id_1_doc_hash_1")
            except Exception:
                pass  # Ignore drop failure, might not exist under this name

            try:
                self.documents_collection.create_index(
                    [("bot_id", 1), ("doc_hash", 1)],
                    unique=True,
                    background=True,
                    partialFilterExpression={"doc_hash": {"$type": "string"}},
                )
            except Exception as e:
                print(f"[WARN] Could not create doc_hash unique index: {e}")

        # chats: (bot_id, chat_id) — used by all chat history queries
        self.chats.create_index([("bot_id", 1), ("chat_id", 1)], background=True)
        # jobs: (bot_id, status) — used by H4 job polling and P3-3 lock guard
        self.db["jobs"].create_index([("bot_id", 1), ("status", 1)], background=True)


    # ─── Encryption Helpers ───────────────────────────────────────────────────

    def encrypt_data(self, data: str) -> str:
        """Encrypt a string using Fernet."""
        try:
            if self._fernet is None:
                return data
            return self._fernet.encrypt(data.encode()).decode()
        except Exception as e:
            print(f"[ERROR] Encryption error: {e}")
            return data

    def decrypt_data(self, data: str) -> str:
        """Decrypt a Fernet-encrypted string."""
        try:
            if self._fernet is None:
                return data
            return self._fernet.decrypt(data.encode()).decode()
        except Exception as e:
            raise ValueError(
                f"[DECRYPT FAIL] Could not decrypt data. Key mismatch or corrupted ciphertext: {e}"
            ) from e

    # ─── Bot Persistence ──────────────────────────────────────────────────────

    def save_bot(self, bot_id: str, db_path: str) -> None:
        """Insert a new bot record."""
        self.bots.insert_one({"bot_id": bot_id, "db_path": db_path})

    def find_bot(self, bot_id: str) -> Optional[dict]:
        """Find a bot config by ID."""
        return self.bots.find_one({"bot_id": bot_id})

    def update_bot(self, bot_id: str, updates: dict) -> None:
        """Update bot config fields."""
        self.bots.update_one({"bot_id": bot_id}, {"$set": updates})

    def delete_bot(self, bot_id: str) -> None:
        """Delete bot and all associated data."""
        self.chats.delete_many({"bot_id": bot_id})
        self.vision_chats.delete_many({"bot_id": bot_id})
        self.documents_collection.delete_many({"bot_id": bot_id})
        self.bots.delete_one({"bot_id": bot_id})

    # ─── P3-1: Schema Registry ────────────────────────────────────────────────

    def save_schema(self, bot_id: str, schema_json: dict, schema_hash: str) -> None:
        """P3-1: Save a schema version to the dedicated bot_schemas collection.

        Upserts — if the hash already exists for this bot, no duplicate is created.
        Updates the bot document to point to this hash as the current schema.
        """
        self.bot_schemas.update_one(
            {"bot_id": bot_id, "schema_version_hash": schema_hash},
            {"$setOnInsert": {
                "bot_id": bot_id,
                "schema_version_hash": schema_hash,
                "schema_json": schema_json,
                "created_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        # Update pointer on the bot document
        self.bots.update_one(
            {"bot_id": bot_id},
            {"$set": {"current_schema_version_hash": schema_hash}},
        )

    def get_current_schema(self, bot_id: str) -> Optional[dict]:
        """P3-1: Retrieve the current active schema for a bot."""
        bot = self.bots.find_one({"bot_id": bot_id})
        if not bot or not bot.get("current_schema_version_hash"):
            return None
        return self.bot_schemas.find_one({
            "bot_id": bot_id,
            "schema_version_hash": bot["current_schema_version_hash"],
        })

    def list_schema_versions(self, bot_id: str) -> list[dict]:
        """P3-1: List all schema versions for a bot (newest first)."""
        cursor = self.bot_schemas.find(
            {"bot_id": bot_id},
            {"_id": 0, "schema_version_hash": 1, "created_at": 1},
        ).sort("created_at", -1)
        return list(cursor)

    # ─── Document Storage ─────────────────────────────────────────────────────

    def save_document(self, bot_id: str, serialized_doc: dict) -> None:
        """Store a serialized document for a bot.

        H3: Includes content hash and indexed flag for delta ingestion.
        """
        import hashlib
        content = serialized_doc.get("page_content", "")
        doc_hash = hashlib.sha256(content.encode()).hexdigest()

        # Skip if exact duplicate already stored for this bot
        existing = self.documents_collection.find_one({
            "bot_id": bot_id,
            "doc_hash": doc_hash,
        })
        if existing:
            return

        self.documents_collection.insert_one({
            "bot_id": bot_id,
            "document": serialized_doc,
            "doc_hash": doc_hash,
            "indexed": False,
        })

    def find_documents(self, bot_id: str) -> list[dict]:
        """Retrieve all raw document records for a bot."""
        return list(self.documents_collection.find({"bot_id": bot_id}))

    def find_unindexed_documents(self, bot_id: str) -> list[dict]:
        """H3: Find documents not yet embedded into the vector store."""
        return list(self.documents_collection.find({
            "bot_id": bot_id,
            "indexed": {"$ne": True},
        }))

    def mark_documents_indexed(self, bot_id: str) -> None:
        """H3: Mark all documents for a bot as indexed."""
        self.documents_collection.update_many(
            {"bot_id": bot_id, "indexed": {"$ne": True}},
            {"$set": {"indexed": True, "indexed_at": datetime.now(timezone.utc)}},
        )

    def count_documents(self, bot_id: str) -> int:
        """Count stored documents for a bot."""
        return self.documents_collection.count_documents({"bot_id": bot_id})


    # ─── H4: Job Tracking ─────────────────────────────────────────────────────

    def create_job(self, job_id: str, bot_id: str, job_type: str = "document_ingest") -> None:
        """H4: Create a job record for async task tracking."""
        self.db["jobs"].insert_one({
            "job_id": job_id,
            "bot_id": bot_id,
            "type": job_type,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "error": None,
        })

    def update_job_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        """H4: Update a job's status."""
        update = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if error:
            update["error"] = error
        self.db["jobs"].update_one({"job_id": job_id}, {"$set": update})

    def get_job(self, job_id: str) -> Optional[dict]:
        """H4: Retrieve a job record."""
        return self.db["jobs"].find_one({"job_id": job_id})

    # ─── Chat Storage ─────────────────────────────────────────────────────────

    def store_chat(
        self,
        bot_id: str,
        chat_id: str,
        query: str,
        answer: str,
        web_source: Optional[list[str]] = None,
        uploaded_files: Optional[list[dict]] = None,
    ) -> None:
        """Store a chat exchange in MongoDB with optional encryption."""
        try:
            current_time = datetime.now(timezone.utc)

            if self.encrypt_chats:
                enc_query = self.encrypt_data(query)
                enc_answer = self.encrypt_data(answer)
                enc_sources = (
                    [self.encrypt_data(s) for s in web_source]
                    if web_source
                    else []
                )
            else:
                enc_query, enc_answer = query, answer
                enc_sources = web_source or []

            self.chats.insert_one({
                "bot_id": bot_id,
                "chat_id": chat_id,
                "timestamp": current_time,
                "question": enc_query,
                "answer": enc_answer,
                "web_sources": enc_sources,
                "uploaded_files": uploaded_files,
                "trained": False,
            })
        except Exception as e:
            print(f"[ERROR] Error storing chat: {e}")

    def store_vision_chat(
        self,
        bot_id: str,
        vision_chat_id: str,
        image_paths: list[str],
        query: str,
        response: str,
        web_source: Optional[list[str]] = None,
        uploaded_files: Optional[list[dict]] = None,
    ) -> None:
        """Store a vision chat exchange in MongoDB with optional encryption."""
        try:
            current_time = datetime.now(timezone.utc)

            if self.encrypt_chats:
                enc_query = self.encrypt_data(query)
                enc_response = self.encrypt_data(response)
                enc_sources = (
                    [self.encrypt_data(s) for s in web_source]
                    if web_source
                    else []
                )
            else:
                enc_query, enc_response = query, response
                enc_sources = web_source or []

            self.vision_chats.insert_one({
                "bot_id": bot_id,
                "vision_chat_id": vision_chat_id,
                "timestamp": current_time,
                "image_path": ",".join(image_paths),
                "question": enc_query,
                "response": enc_response,
                "web_sources": enc_sources,
                "uploaded_files": uploaded_files,
                "trained": False,
            })
        except Exception as e:
            print(f"[ERROR] Error storing vision chat: {e}")

    # ─── Chat Retrieval ───────────────────────────────────────────────────────

    def list_chats(self, bot_id: str) -> dict:
        """List all chat and vision chat IDs for a bot."""
        try:
            return {
                "chat_ids": list(self.chats.distinct("chat_id", {"bot_id": bot_id})),
                "vision_chat_ids": list(
                    self.vision_chats.distinct("vision_chat_id", {"bot_id": bot_id})
                ),
            }
        except Exception as e:
            print(f"[ERROR] Error listing chats: {e}")
            return {"chat_ids": [], "vision_chat_ids": []}

    def get_chat_by_id(self, chat_id: str, order: str = "newest") -> Optional[list[dict]]:
        """Retrieve full chat history for a session."""
        try:
            sort_order = -1 if order == "newest" else 1
            chat_data = list(self.chats.find({"chat_id": chat_id}).sort("timestamp", sort_order))
            if not chat_data:
                return None

            if self.encrypt_chats:
                for chat in chat_data:
                    chat["question"] = self.decrypt_data(chat["question"])
                    chat["answer"] = self.decrypt_data(chat["answer"])
                    if chat.get("web_sources"):
                        chat["web_sources"] = [
                            self.decrypt_data(s) for s in chat["web_sources"]
                        ]
            return chat_data
        except Exception as e:
            print(f"[ERROR] Error getting chat {chat_id}: {e}")
            return None

    def get_vision_chat_by_id(
        self, vision_chat_id: str, order: str = "newest"
    ) -> Optional[list[dict]]:
        """Retrieve full vision chat history."""
        try:
            sort_order = -1 if order == "newest" else 1
            data = list(
                self.vision_chats.find({"vision_chat_id": vision_chat_id}).sort(
                    "timestamp", sort_order
                )
            )
            if not data:
                return None

            if self.encrypt_chats:
                for chat in data:
                    chat["question"] = self.decrypt_data(chat["question"])
                    chat["response"] = self.decrypt_data(chat["response"])
                    if chat.get("web_sources"):
                        chat["web_sources"] = [
                            self.decrypt_data(s) for s in chat["web_sources"]
                        ]
            return data
        except Exception as e:
            print(f"[ERROR] Error getting vision chat {vision_chat_id}: {e}")
            return None

    def find_untrained_chats(self, bot_id: str) -> list[dict]:
        """Find chats not yet used for training."""
        return list(self.chats.find({"bot_id": bot_id, "trained": {"$ne": True}}))

    def mark_chats_trained(self, chat_ids: list) -> None:
        """Mark chats as trained."""
        self.chats.update_many(
            {"_id": {"$in": chat_ids}}, {"$set": {"trained": True}}
        )

    # ─── Export ────────────────────────────────────────────────────────────────

    def export_chats_to_csv(self, dataframe: pd.DataFrame, bot_id: str) -> str:
        """Export a DataFrame of chats to a CSV file."""
        try:
            csv_folder = f"./data-{bot_id}"
            os.makedirs(csv_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            csv_path = os.path.join(csv_folder, f"{bot_id}_data_{timestamp}.csv")
            dataframe.to_csv(csv_path, index=False)
            return csv_path
        except Exception as e:
            print(f"[ERROR] Error exporting chats: {e}")
            return ""
