"""Document ingestion pipeline for LongTrainer.

Handles loading, storing, and retrieving documents from various sources.
"""

from __future__ import annotations

import gc
from typing import Optional

from longtrainer.loaders import DocumentLoader
from longtrainer.storage import MongoStorage
from longtrainer.utils import deserialize_document, serialize_document


class DocumentManager:
    """Manages document ingestion from files, links, and queries.

    Args:
        storage: A MongoStorage instance for database operations.
        document_loader: A DocumentLoader instance (optional, creates default).
    """

    def __init__(
        self,
        storage: MongoStorage,
        document_loader: Optional[DocumentLoader] = None,
    ) -> None:
        self.storage = storage
        self.document_loader = document_loader or DocumentLoader()

    def get_documents(self, bot_id: str) -> list:
        """Retrieve deserialized documents from MongoDB for a bot.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            List of deserialized Document objects.
        """
        try:
            return [
                deserialize_document(doc["document"])
                for doc in self.storage.find_documents(bot_id)
            ]
        except Exception as e:
            print(f"[ERROR] Error loading documents for bot {bot_id}: {e}")
            return []

    def add_document_from_path(
        self, path: str, bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Load and store documents from a file path.

        Args:
            path: Path to the document file.
            bot_id: The bot's unique identifier.
            use_unstructured: Use UnstructuredLoader for any file type.
        """
        try:
            if use_unstructured:
                documents = self.document_loader.load_unstructured(path)
            else:
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                loaders = {
                    "csv": self.document_loader.load_csv,
                    "docx": self.document_loader.load_doc,
                    "pdf": self.document_loader.load_pdf,
                    "md": self.document_loader.load_markdown,
                    "markdown": self.document_loader.load_markdown,
                    "txt": self.document_loader.load_markdown,
                    "html": self.document_loader.load_text_from_html,
                    "htm": self.document_loader.load_text_from_html,
                }
                loader_fn = loaders.get(ext)
                if not loader_fn:
                    raise ValueError(f"Unsupported file type: {ext}")
                documents = loader_fn(path)

            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))

            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from path: {e}")

    def add_document_from_link(self, links: list[str], bot_id: str) -> None:
        """Load and store documents from web links.

        Args:
            links: List of URLs or YouTube links.
            bot_id: The bot's unique identifier.
        """
        try:
            for link in links:
                if "youtube.com" in link.lower() or "youtu.be" in link.lower():
                    documents = self.document_loader.load_youtube_video(link)
                else:
                    documents = self.document_loader.load_urls([link])

                for doc in documents:
                    self.storage.save_document(bot_id, serialize_document(doc))
                del documents
                gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from link: {e}")

    def add_document_from_query(self, search_query: str, bot_id: str) -> None:
        """Load and store documents from a Wikipedia search.

        Args:
            search_query: Wikipedia search query.
            bot_id: The bot's unique identifier.
        """
        try:
            documents = self.document_loader.wikipedia_query(search_query)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from query: {e}")

    def pass_documents(self, documents: list, bot_id: str) -> None:
        """Store pre-loaded LangChain documents.

        Args:
            documents: List of Document objects.
            bot_id: The bot's unique identifier.
        """
        try:
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding documents: {e}")

    def count_documents(self, bot_id: str) -> int:
        """Count stored documents for a bot."""
        return self.storage.count_documents(bot_id)
