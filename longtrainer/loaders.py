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
    NotionDirectoryLoader,
    RecursiveUrlLoader,
    GitLoader,
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

    def load_dynamic_loader(self, loader_class_name: str, **kwargs) -> list[Document]:
        """Dynamically instantiate any LangChain Document Loader by class name.

        Args:
            loader_class_name: The exact class name of the LangChain loader (e.g. 'SlackDirectoryLoader').
            **kwargs: Arguments to pass to the loader's `__init__`.

        Returns:
            List of loaded documents.
        """
        import importlib

        try:
            loader_module = importlib.import_module("langchain_community.document_loaders")
            if not hasattr(loader_module, loader_class_name):
                # Try core or standard document_loaders as fallback
                raise ValueError(f"Loader '{loader_class_name}' not found in langchain_community.document_loaders.")
                
            loader_class = getattr(loader_module, loader_class_name)
            loader_instance = loader_class(**kwargs)
            return loader_instance.load()
            
        except ImportError:
            print("[ERROR] langchain_community is required for dynamic document loaders.")
            return []
        except Exception as e:
            print(f"[ERROR] Error loading dynamic loader '{loader_class_name}': {e}")
            return []

    def load_github_repo(
        self, repo_url: str, branch: str = "main", access_token: Optional[str] = None
    ) -> list[Document]:
        """Load documents from a GitHub repository using Git clone.

        Args:
            repo_url: URL or 'owner/repo' string.
            branch: Repository branch to load.
            access_token: GitHub Personal Access Token.

        Returns:
            List of loaded documents.
        """
        import tempfile
        import shutil
        try:
            if not repo_url.startswith("http"):
                repo_url = f"https://github.com/{repo_url}"
                
            if access_token:
                repo_url = repo_url.replace("https://", f"https://oauth2:{access_token}@")

            temp_dir = tempfile.mkdtemp()
            
            loader = GitLoader(
                clone_url=repo_url,
                repo_path=temp_dir,
                branch=branch,
            )
            docs = loader.load()
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return docs
        except Exception as e:
            print(f"[ERROR] Error loading GitHub repo: {e}")
            return []

    def load_notion_directory(self, path: str) -> list[Document]:
        """Load documents from an exported Notion directory.

        Args:
            path: Path to the unzipped Notion export directory.

        Returns:
            List of loaded documents.
        """
        try:
            loader = NotionDirectoryLoader(path)
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error loading Notion directory: {e}")
            return []

    def crawl_website(self, url: str, max_depth: int = 2) -> list[Document]:
        """Deep crawl a website using RecursiveUrlLoader.

        Args:
            url: The root URL to crawl.
            max_depth: Depth of the crawl.

        Returns:
            List of loaded documents.
        """
        try:
            from bs4 import BeautifulSoup
            
            def extractor(html_str: str) -> str:
                soup = BeautifulSoup(html_str, "html.parser")
                return soup.get_text(separator=" ", strip=True)
                
            loader = RecursiveUrlLoader(
                url=url, 
                max_depth=max_depth, 
                extractor=extractor
            )
            return loader.load()
        except Exception as e:
            print(f"[ERROR] Error crawling URL {url}: {e}")
            return []

    def load_directory(self, path: str, glob: str = "**/*") -> list[Document]:
        """Load documents recursively from a local directory."""
        return self.load_dynamic_loader("DirectoryLoader", path=path, glob=glob)

    def load_json(self, path: str, jq_schema: str = ".") -> list[Document]:
        """Load documents from a JSON or JSONL file."""
        return self.load_dynamic_loader("JSONLoader", file_path=path, jq_schema=jq_schema, text_content=False)

    def load_aws_s3(self, bucket: str, prefix: str = "", aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None) -> list[Document]:
        """Load documents from an AWS S3 Directory."""
        return self.load_dynamic_loader(
            "S3DirectoryLoader", 
            bucket=bucket, 
            prefix=prefix, 
            aws_access_key_id=aws_access_key_id, 
            aws_secret_access_key=aws_secret_access_key
        )

    def load_google_drive(self, folder_id: str, credentials_path: str = "credentials.json") -> list[Document]:
        """Load documents from a Google Drive folder."""
        return self.load_dynamic_loader(
            "GoogleDriveLoader", 
            folder_id=folder_id, 
            credentials_path=credentials_path, 
            recursive=True
        )

    def load_confluence(self, url: str, username: str, api_key: str, space_key: Optional[str] = None) -> list[Document]:
        """Load documents from a Confluence Workspace."""
        return self.load_dynamic_loader(
            "ConfluenceLoader", 
            url=url, 
            username=username, 
            api_key=api_key, 
            space_key=space_key
        )


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
