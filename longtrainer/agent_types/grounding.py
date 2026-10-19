"""Source identifiers for document-grounded support prompts."""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class SupportRetriever(BaseRetriever):
    """Annotate copied documents so source identifiers reach the RAG prompt."""

    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        docs = self.retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        return [Document(
            page_content=f"[Document {index}] Source: {doc.metadata.get('source', 'knowledge base')}\n{doc.page_content}",
            metadata=doc.metadata.copy(),
        ) for index, doc in enumerate(docs, 1)]
