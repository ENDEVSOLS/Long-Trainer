"""Document retrieval module for LongTrainer.

Provides FAISS vector indexing with optional multi-query ensemble retrieval
implemented natively (no deprecated langchain_classic dependencies).

Also exposes similarity_search_with_score() so callers can obtain a
confidence/relevance score alongside each retrieved document.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from longtrainer.utils import LineListOutputParser


_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an AI language model assistant. Your task is to generate five "
        "different versions of the given user question to retrieve relevant documents "
        "from a vector database. By generating multiple perspectives on the user "
        "question, your goal is to help the user overcome some of the limitations "
        "of the distance-based similarity search. "
        "Provide these alternative questions separated by newlines.\n"
        "Original question: {question}"
    ),
)


class MultiQueryEnsembleRetriever(BaseRetriever):
    """Custom retriever that combines FAISS similarity search with multi-query expansion.

    Generates alternative queries using an LLM, runs each against FAISS,
    and de-duplicates the results for higher recall.
    """

    base_retriever: BaseRetriever
    llm: BaseLanguageModel
    k: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> list[Document]:
        """Retrieve documents using both direct and multi-query retrieval."""
        direct_docs = self.base_retriever.invoke(query)

        try:
            chain = _MULTI_QUERY_PROMPT | self.llm | StrOutputParser() | LineListOutputParser()
            alt_queries = chain.invoke({"question": query})
        except Exception:
            alt_queries = []

        all_docs = list(direct_docs)
        seen_contents = {doc.page_content for doc in all_docs}

        for alt_query in alt_queries[:5]:
            if not alt_query.strip():
                continue
            try:
                alt_docs = self.base_retriever.invoke(alt_query.strip())
                for doc in alt_docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
            except Exception:
                continue

        return all_docs


class DocumentRetriever:
    """Wrapper class to manage FAISS indices and ensemble retrieval."""

    def __init__(
        self,
        documents: list[Document],
        embedding_model: Embeddings,
        llm: Optional[BaseChatModel] = None,
        ensemble: bool = False,
        existing_faiss_index=None,
        num_k: int = 3,
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.ensemble = ensemble
        self.k = num_k

        if existing_faiss_index:
            self.faiss_index = existing_faiss_index
            self.faiss_index.add_documents(documents)
        else:
            from langchain_community.vectorstores import FAISS
            self.faiss_index = FAISS.from_documents(documents, self.embedding_model)

    def retrieve_documents(self) -> BaseRetriever:
        """Returns the configured retriever (base FAISS or Ensemble)."""
        base_retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.k})
        if self.ensemble and self.llm:
            return MultiQueryEnsembleRetriever(
                base_retriever=base_retriever, llm=self.llm, k=self.k
            )
        return base_retriever

    def update_index(self, new_documents: list[Document]):
        """Adds new documents to the existing FAISS index."""
        if self.faiss_index:
            self.faiss_index.add_documents(new_documents)

    def save_index(self, file_path: str):
        """Saves the underlying FAISS index locally."""
        if self.faiss_index:
            self.faiss_index.save_local(file_path)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """Return (document, similarity_score) pairs for the given query.

        The score is the raw L2 distance from FAISS — lower is more similar.
        A normalised 0-1 confidence score is also added to each document's
        metadata under the key ``retrieval_score`` (1 = most similar).

        Args:
            query: The search string.
            k: Number of results to return (defaults to self.k).

        Returns:
            List of (Document, raw_score) tuples, sorted by ascending L2 distance.
        """
        if self.faiss_index is None:
            return []

        num = k or self.k
        try:
            results = self.faiss_index.similarity_search_with_score(query, k=num)
        except Exception as e:
            print(f"[ERROR] similarity_search_with_score failed: {e}")
            return []

        # Attach a normalised confidence score to metadata (1.0 = best match)
        if results:
            raw_scores = [score for _, score in results]
            max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
            for doc, score in results:
                doc.metadata["retrieval_score"] = round(1.0 - (score / max_score), 4)

        return results
