from langchain_core.documents import Document


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

