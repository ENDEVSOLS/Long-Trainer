"""Document ingestion pipeline for LongTrainer.

Handles loading, storing, and retrieving documents from various sources.
Supports both sync and async (non-blocking) ingestion paths.
"""

from __future__ import annotations

import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from longtrainer.loaders import DocumentLoader
from longtrainer.storage import MongoStorage
from longtrainer.utils import deserialize_document, serialize_document

# Thread pool used for async wrappers around blocking I/O loaders
_LOADER_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lt_doc_loader")


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

    # ─── Sync helpers ──────────────────────────────────────────────────────────

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

    def _save_docs(self, bot_id: str, documents: list) -> None:
        """Helper: serialize and store a list of documents."""
        for doc in documents:
            self.storage.save_document(bot_id, serialize_document(doc))

    # ─── add_document_from_path (sync + async) ─────────────────────────────────

    def add_document_from_path(
        self, path: str, bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Load and store documents from a file path (blocking).

        Args:
            path: Path to the document file.
            bot_id: The bot's unique identifier.
            use_unstructured: Use UnstructuredLoader for any file type.
        """
        try:
            documents = self._load_from_path(path, use_unstructured)
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from path: {e}")

    async def aadd_document_from_path(
        self, path: str, bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Load and store documents from a file path (non-blocking async).

        Runs the blocking file I/O in a background thread so it does not
        stall the event loop (e.g. when called from a FastAPI endpoint).

        Args:
            path: Path to the document file.
            bot_id: The bot's unique identifier.
            use_unstructured: Use UnstructuredLoader for any file type.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _LOADER_EXECUTOR,
            self.add_document_from_path,
            path,
            bot_id,
            use_unstructured,
        )

    def add_documents_from_paths(
        self, paths: list[str], bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Load and store documents from multiple file paths in parallel.

        Uses a thread pool so that all files are processed concurrently.
        Falls back to sequential loading if the event loop is not running.

        Args:
            paths: List of file paths to load.
            bot_id: The bot's unique identifier.
            use_unstructured: Use UnstructuredLoader for all files.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an async context — schedule as async tasks
                asyncio.ensure_future(
                    self._aadd_documents_from_paths(paths, bot_id, use_unstructured)
                )
            else:
                loop.run_until_complete(
                    self._aadd_documents_from_paths(paths, bot_id, use_unstructured)
                )
        except RuntimeError:
            # No event loop at all — fall back to sequential
            for path in paths:
                self.add_document_from_path(path, bot_id, use_unstructured)

    async def _aadd_documents_from_paths(
        self, paths: list[str], bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Internal async parallel ingestion for multiple file paths."""
        tasks = [
            self.aadd_document_from_path(path, bot_id, use_unstructured)
            for path in paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ERROR] Failed to load '{paths[i]}': {result}")

    # ─── add_document_from_link (sync + async) ─────────────────────────────────

    def add_document_from_link(self, links: list[str], bot_id: str) -> None:
        """Load and store documents from web links (blocking).

        Args:
            links: List of URLs or YouTube links.
            bot_id: The bot's unique identifier.
        """
        for link in links:
            try:
                if "youtube.com" in link.lower() or "youtu.be" in link.lower():
                    documents = self.document_loader.load_youtube_video(link)
                else:
                    documents = self.document_loader.load_urls([link])
                self._save_docs(bot_id, documents)
                del documents
                gc.collect()
            except Exception as e:
                print(f"[ERROR] Error adding document from link '{link}': {e}")

    async def aadd_document_from_link(self, links: list[str], bot_id: str) -> None:
        """Load and store documents from web links in parallel (non-blocking async).

        Each link is fetched concurrently in a background thread pool.

        Args:
            links: List of URLs or YouTube links.
            bot_id: The bot's unique identifier.
        """
        loop = asyncio.get_event_loop()

        async def _fetch_one(link: str) -> None:
            await loop.run_in_executor(
                _LOADER_EXECUTOR,
                lambda: self._load_and_save_single_link(link, bot_id),
            )

        results = await asyncio.gather(
            *[_fetch_one(link) for link in links], return_exceptions=True
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[ERROR] Failed to load link '{links[i]}': {result}")

    def _load_and_save_single_link(self, link: str, bot_id: str) -> None:
        """Blocking helper: load one link and save to MongoDB."""
        if "youtube.com" in link.lower() or "youtu.be" in link.lower():
            documents = self.document_loader.load_youtube_video(link)
        else:
            documents = self.document_loader.load_urls([link])
        self._save_docs(bot_id, documents)
        del documents
        gc.collect()

    # ─── Internal loader helper ────────────────────────────────────────────────

    def _load_from_path(self, path: str, use_unstructured: bool = False) -> list:
        """Return loaded documents for a single file path."""
        if use_unstructured:
            return self.document_loader.load_unstructured(path)

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
            raise ValueError(f"Unsupported file type: .{ext}")
        return loader_fn(path)

    # ─── Remaining sync loaders ────────────────────────────────────────────────

    def add_document_from_query(self, search_query: str, bot_id: str) -> None:
        """Load and store documents from a Wikipedia search.

        Args:
            search_query: Wikipedia search query.
            bot_id: The bot's unique identifier.
        """
        try:
            documents = self.document_loader.wikipedia_query(search_query)
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from query: {e}")

    def add_document_from_github(
        self,
        repo_url: str,
        bot_id: str,
        branch: str = "main",
        access_token: Optional[str] = None,
    ) -> None:
        """Load and store documents from a GitHub repository.

        Args:
            repo_url: URL or 'owner/repo' string.
            bot_id: The bot's unique identifier.
            branch: Repository branch to load.
            access_token: GitHub Personal Access Token.
        """
        try:
            documents = self.document_loader.load_github_repo(repo_url, branch, access_token)
            self._save_docs(bot_id, documents)
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
            self._save_docs(bot_id, documents)
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
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from crawl: {e}")

    def add_document_from_directory(self, path: str, bot_id: str, glob: str = "**/*") -> None:
        """Load documents recursively from a local directory."""
        try:
            documents = self.document_loader.load_directory(path, glob)
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from directory: {e}")

    def add_document_from_json(self, path: str, bot_id: str, jq_schema: str = ".") -> None:
        """Load documents from a JSON or JSONL file."""
        try:
            documents = self.document_loader.load_json(path, jq_schema)
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from JSON: {e}")

    def add_document_from_aws_s3(
        self,
        bucket: str,
        bot_id: str,
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        """Load documents from an AWS S3 Directory."""
        try:
            documents = self.document_loader.load_aws_s3(
                bucket, prefix, aws_access_key_id, aws_secret_access_key
            )
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from AWS S3: {e}")

    def add_document_from_google_drive(
        self, folder_id: str, bot_id: str, credentials_path: str = "credentials.json"
    ) -> None:
        """Load documents from a Google Drive folder."""
        try:
            documents = self.document_loader.load_google_drive(folder_id, credentials_path)
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from Google Drive: {e}")

    def add_document_from_confluence(
        self,
        url: str,
        username: str,
        api_key: str,
        bot_id: str,
        space_key: Optional[str] = None,
    ) -> None:
        """Load documents from a Confluence Workspace."""
        try:
            documents = self.document_loader.load_confluence(url, username, api_key, space_key)
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from Confluence: {e}")

    def add_document_from_dynamic_loader(
        self, bot_id: str, loader_class_name: str, **kwargs
    ) -> None:
        """Instantiate ANY LangChain document loader dynamically.

        Args:
            bot_id: The bot's unique identifier.
            loader_class_name: The exact class name of the LangChain loader
                               (e.g. 'SlackDirectoryLoader').
            **kwargs: Arguments to pass to the loader's ``__init__``.
        """
        try:
            documents = self.document_loader.load_dynamic_loader(loader_class_name, **kwargs)
            self._save_docs(bot_id, documents)
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
            self._save_docs(bot_id, documents)
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding documents: {e}")

    def count_documents(self, bot_id: str) -> int:
        """Count stored documents for a bot."""
        return self.storage.count_documents(bot_id)
