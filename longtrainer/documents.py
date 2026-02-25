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

    def add_document_from_github(self, repo_url: str, bot_id: str, branch: str = "main", access_token: Optional[str] = None) -> None:
        """Load and store documents from a GitHub repository.

        Args:
            repo_url: URL or 'owner/repo' string.
            bot_id: The bot's unique identifier.
            branch: Repository branch to load.
            access_token: GitHub Personal Access Token.
        """
        try:
            documents = self.document_loader.load_github_repo(repo_url, branch, access_token)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from GitHub: {e}")

    def add_document_from_notion(self, path: str, bot_id: str) -> None:
        """Load and store documents from an exported Notion directory.

        Args:
            path: Path to the unzipped Notion export directory.
            bot_id: The bot's unique identifier.
        """
        try:
            documents = self.document_loader.load_notion_directory(path)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from Notion: {e}")

    def add_document_from_crawl(self, url: str, bot_id: str, max_depth: int = 2) -> None:
        """Deep crawl a website and store documents.

        Args:
            url: The root URL to crawl.
            bot_id: The bot's unique identifier.
            max_depth: Maximum recursion depth for crawl.
        """
        try:
            documents = self.document_loader.crawl_website(url, max_depth)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from crawl: {e}")

    def add_document_from_directory(self, path: str, bot_id: str, glob: str = "**/*") -> None:
        """Load documents recursively from a local directory."""
        try:
            documents = self.document_loader.load_directory(path, glob)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from directory: {e}")

    def add_document_from_json(self, path: str, bot_id: str, jq_schema: str = ".") -> None:
        """Load documents from a JSON or JSONL file."""
        try:
            documents = self.document_loader.load_json(path, jq_schema)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from JSON: {e}")

    def add_document_from_aws_s3(self, bucket: str, bot_id: str, prefix: str = "", aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None) -> None:
        """Load documents from an AWS S3 Directory."""
        try:
            documents = self.document_loader.load_aws_s3(bucket, prefix, aws_access_key_id, aws_secret_access_key)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from AWS S3: {e}")

    def add_document_from_google_drive(self, folder_id: str, bot_id: str, credentials_path: str = "credentials.json") -> None:
        """Load documents from a Google Drive folder."""
        try:
            documents = self.document_loader.load_google_drive(folder_id, credentials_path)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from Google Drive: {e}")

    def add_document_from_confluence(self, url: str, username: str, api_key: str, bot_id: str, space_key: Optional[str] = None) -> None:
        """Load documents from a Confluence Workspace."""
        try:
            documents = self.document_loader.load_confluence(url, username, api_key, space_key)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from Confluence: {e}")

    def add_document_from_dynamic_loader(self, bot_id: str, loader_class_name: str, **kwargs) -> None:
        """Instantiate ANY LangChain document loader dynamically.

        Args:
            bot_id: The bot's unique identifier.
            loader_class_name: The exact class name of the LangChain loader (e.g. 'SlackDirectoryLoader').
            **kwargs: Arguments to pass to the loader's `__init__`.
        """
        try:
            documents = self.document_loader.load_dynamic_loader(loader_class_name, **kwargs)
            for doc in documents:
                self.storage.save_document(bot_id, serialize_document(doc))
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document dynamically: {e}")

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
