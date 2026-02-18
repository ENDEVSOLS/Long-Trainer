"""Document loaders and text splitters for LongTrainer V2."""

from __future__ import annotations

from typing import Optional

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredURLLoader,
    WikipediaLoader,
    YoutubeLoader,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


class DocumentLoader:
    """Loads documents from various sources.

    Supports: CSV, PDF, DOCX, HTML, Markdown/TXT, URLs, YouTube, Wikipedia,
    and any format via UnstructuredLoader.
    """

    def load_unstructured(self, path: str) -> list[Document]:
        """Load from any file using UnstructuredLoader.

        Supports: csv, doc, docx, epub, image, md, msg, odt, org, pdf,
        ppt, pptx, rtf, rst, tsv, xlsx, and more.

        Args:
            path: File path to load.

        Returns:
            List of loaded documents.
        """
        try:
            loader = UnstructuredLoader(path)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading Unstructured: {e}")
            return []

    def load_csv(self, path: str) -> list[Document]:
        """Load from a CSV file.

        Args:
            path: File path to the CSV.

        Returns:
            List of loaded documents.
        """
        try:
            loader = CSVLoader(file_path=path)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading CSV: {e}")
            return []

    def wikipedia_query(self, search_query: str, max_docs: int = 2) -> list[Document]:
        """Query Wikipedia and return results.

        Args:
            search_query: Search term for Wikipedia.
            max_docs: Maximum number of Wikipedia articles to load.

        Returns:
            List of loaded documents.
        """
        try:
            data = WikipediaLoader(query=search_query, load_max_docs=max_docs).load()
            return data
        except Exception as e:
            print(f"[ERROR] Error querying Wikipedia: {e}")
            return []

    def load_urls(self, urls: list[str]) -> list[Document]:
        """Load and parse content from a list of URLs.

        Args:
            urls: List of URLs to load.

        Returns:
            List of loaded documents.
        """
        try:
            loader = UnstructuredURLLoader(urls=urls)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading URLs: {e}")
            return []

    def load_youtube_video(self, url: str) -> list[Document]:
        """Load YouTube video transcript.

        Args:
            url: YouTube video URL.

        Returns:
            List of loaded documents.
        """
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=["en", "pt", "zh-Hans", "es", "ur", "hi"],
                translation="en",
            )
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading YouTube video: {e}")
            return []

    def load_pdf(self, path: str) -> list[Document]:
        """Load from a PDF file.

        Args:
            path: File path to the PDF.

        Returns:
            List of loaded and split pages.
        """
        try:
            loader = PyPDFLoader(path)
            return loader.load_and_split()
        except Exception as e:
            print(f"[ERROR] Error loading PDF: {e}")
            return []

    def load_text_from_html(self, path: str) -> list[Document]:
        """Load and parse text from an HTML file.

        Args:
            path: File path to the HTML file.

        Returns:
            List of loaded documents.
        """
        try:
            loader = BSHTMLLoader(path)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading text from HTML: {e}")
            return []

    def load_markdown(self, path: str) -> list[Document]:
        """Load from a Markdown or plain text file.

        Args:
            path: File path to the Markdown/text file.

        Returns:
            List of loaded documents.
        """
        try:
            loader = UnstructuredMarkdownLoader(path)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading Markdown: {e}")
            return []

    def load_doc(self, path: str) -> list[Document]:
        """Load from a DOCX file.

        Args:
            path: File path to the DOCX file.

        Returns:
            List of loaded documents.
        """
        try:
            loader = Docx2txtLoader(path)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading DOCX: {e}")
            return []


class TextSplitter:
    """Splits documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        chunk_size: The size of each text chunk.
        chunk_overlap: The overlap size between chunks.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of split document chunks.
        """
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"[ERROR] Error splitting documents: {e}")
            return []
