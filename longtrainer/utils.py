"""Utility functions and parsers for LongTrainer V2."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser


def serialize_document(doc: Document) -> dict:
    """Convert a Document object to a serializable dictionary.

    Args:
        doc: A LangChain Document object.

    Returns:
        Dictionary with page_content, metadata, and type.
    """
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "type": doc.type,
    }


def deserialize_document(doc_data: dict) -> Document:
    """Convert a dictionary back to a Document object.

    Args:
        doc_data: Dictionary containing document data.

    Returns:
        A LangChain Document object.
    """
    return Document(
        page_content=doc_data["page_content"],
        metadata=doc_data.get("metadata", {}),
        type=doc_data.get("type", "Document"),
    )


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser that splits LLM output into a list of lines.

    Used by MultiQueryRetriever to split generated queries.
    """

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))
