"""Document retrieval module using FAISS for LongTrainer V2.

Provides FAISS vector indexing with optional multi-query ensemble retrieval
implemented natively (no deprecated langchain_classic dependencies).
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
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

    faiss_retriever: BaseRetriever
    llm: BaseLanguageModel
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> list[Document]:
        """Retrieve documents using both direct and multi-query retrieval."""
        direct_docs = self.faiss_retriever.invoke(query)

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
                alt_docs = self.faiss_retriever.invoke(alt_query.strip())
                for doc in alt_docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
            except Exception:
                continue

        return all_docs


class DocumentRetriever:
    """FAISS-based document retriever with optional multi-query ensemble retrieval.

    Args:
        documents: List of documents to index.
        embedding_model: Embedding model for vectorization.
        llm: Language model (used only for ensemble/multi-query mode).
        ensemble: Whether to enable multi-query ensemble retrieval.
        existing_faiss_index: An existing FAISS index to use instead of creating new.
        num_k: Number of documents to retrieve per query.
    """

    def __init__(
        self,
        documents: list,
        embedding_model: Embeddings,
        llm: BaseLanguageModel,
        ensemble: bool = False,
        existing_faiss_index: Optional[FAISS] = None,
        num_k: int = 3,
    ) -> None:
        try:
            self.embedding_model = embedding_model
            self.document_collection = documents
            self.faiss_index = existing_faiss_index
            self.k = num_k
            self.ensemble = ensemble
            self.llm = llm

            if not existing_faiss_index and not documents:
                raise ValueError("Document collection is empty and no existing index provided.")

            if not self.faiss_index:
                self._index_documents()

            if self.faiss_index:
                self.faiss_retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.k})

                if self.ensemble:
                    self.ensemble_retriever = MultiQueryEnsembleRetriever(
                        faiss_retriever=self.faiss_retriever,
                        llm=self.llm,
                        k=self.k,
                    )
            else:
                self.faiss_retriever = None

        except Exception as e:
            print(f"[ERROR] Initialization error in DocumentRetriever: {e}")

    def _index_documents(self) -> None:
        """Index documents into FAISS in batches of 1000."""
        if self.faiss_index is not None:
            return

        try:
            batch_size = 1000
            docs = self.document_collection

            if len(docs) <= batch_size:
                self.faiss_index = FAISS.from_documents(docs, self.embedding_model)
            else:
                self.faiss_index = FAISS.from_documents(docs[:batch_size], self.embedding_model)
                for i in range(batch_size, len(docs), batch_size):
                    end = min(i + batch_size, len(docs))
                    batch_index = FAISS.from_documents(docs[i:end], self.embedding_model)
                    self.faiss_index.merge_from(batch_index)
        except Exception as e:
            print(f"[ERROR] Error indexing documents: {e}")

    def save_index(self, file_path: str) -> None:
        """Save the FAISS index to disk.

        Args:
            file_path: Directory path to save the index.
        """
        try:
            self.faiss_index.save_local(file_path)
        except Exception as e:
            print(f"[ERROR] Error saving FAISS index: {e}")

    def update_index(self, new_documents: list) -> None:
        """Add new documents to the existing FAISS index.

        Args:
            new_documents: List of new documents to add.
        """
        if not self.faiss_index:
            raise ValueError("FAISS index not initialized.")

        try:
            batch_size = 1000
            if len(new_documents) <= batch_size:
                new_index = FAISS.from_documents(new_documents, self.embedding_model)
            else:
                new_index = FAISS.from_documents(new_documents[:batch_size], self.embedding_model)
                for i in range(batch_size, len(new_documents), batch_size):
                    end = min(i + batch_size, len(new_documents))
                    batch_index = FAISS.from_documents(new_documents[i:end], self.embedding_model)
                    new_index.merge_from(batch_index)

            self.faiss_index.merge_from(new_index)
            self.document_collection.extend(new_documents)
            self.faiss_retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.k})
        except Exception as e:
            print(f"[ERROR] Error updating FAISS index: {e}")

    def delete_index(self, file_path: str) -> None:
        """Delete the FAISS index from disk.

        Args:
            file_path: Path to the FAISS index directory.
        """
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            else:
                print("[ERROR] FAISS index path does not exist.")
        except Exception as e:
            print(f"[ERROR] Error deleting FAISS index: {e}")

    def retrieve_documents(self) -> Optional[BaseRetriever]:
        """Get the retriever (ensemble or plain FAISS).

        Returns:
            The active retriever instance, or None on error.
        """
        try:
            if self.ensemble:
                return self.ensemble_retriever
            else:
                return self.faiss_retriever
        except Exception as e:
            print(f"[ERROR] Error retrieving documents: {e}")
            return None
