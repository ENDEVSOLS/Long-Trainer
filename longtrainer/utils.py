from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser


def serialize_document(doc: Document) -> dict:
    """
    Convert a Document object to a serializable dictionary.
    """
    return {
        'page_content': doc.page_content,
        'metadata': doc.metadata,
        'type': doc.type
    }


def deserialize_document(doc_data: dict) -> Document:
    """
    Convert a dictionary back to a Document object.
    """
    return Document(page_content=doc_data['page_content'],
                    metadata=doc_data.get('metadata', {}),
                    type=doc_data.get('type', 'Document'))


class LineListOutputParser(BaseOutputParser[List[str]]):
    """
    Output parser for a list of lines.
    To split the LLM result into a list of queries
    """

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines
