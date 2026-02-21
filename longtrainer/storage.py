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
        self.client = MongoClient(config.mongo_endpoint)
        self.db = self.client["longtrainer_db"]

        # Collections
        self.bots = self.db["bots"]
        self.chats = self.db["chats"]
        self.vision_chats = self.db["vision_chats"]
        self.documents_collection = self.db["documents"]

        # Encryption
        self.encrypt_chats = config.encrypt_chats
        self._fernet: Optional[Fernet] = None
        self.encryption_key: Optional[bytes] = None
        if config.encrypt_chats:
            self.encryption_key = config.encryption_key or Fernet.generate_key()
            self._fernet = Fernet(self.encryption_key)

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
            print(f"[ERROR] Decryption error: {e}")
            return data

    # ─── Bot Persistence ──────────────────────────────────────────────────────

    def save_bot(self, bot_id: str, faiss_path: str) -> None:
        """Insert a new bot record."""
        self.bots.insert_one({"bot_id": bot_id, "faiss_path": faiss_path})

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

    # ─── Document Storage ─────────────────────────────────────────────────────

    def save_document(self, bot_id: str, serialized_doc: dict) -> None:
        """Store a serialized document for a bot."""
        self.documents_collection.insert_one({
            "bot_id": bot_id,
            "document": serialized_doc,
        })

    def find_documents(self, bot_id: str) -> list[dict]:
        """Retrieve all raw document records for a bot."""
        return list(self.documents_collection.find({"bot_id": bot_id}))

    def count_documents(self, bot_id: str) -> int:
        """Count stored documents for a bot."""
        return self.documents_collection.count_documents({"bot_id": bot_id})

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
