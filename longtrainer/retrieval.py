"""Document retrieval module using FAISS for LongTrainer V2.

Provides FAISS vector indexing with optional multi-query ensemble retrieval
implemented natively (no deprecated langchain_classic dependencies).
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
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

    base_retriever: BaseRetriever
    llm: BaseLanguageModel
    k: int = 3

    class Config:
        arbitrary_types_allowed = True

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



