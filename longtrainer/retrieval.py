import os
import shutil
from langchain.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever


class DocRetriever:
    """
    Advanced Document Retriever integrates retrieval techniques
    to efficiently retrieve documents based on provided queries.
    """

    def __init__(self, documents, embedding_model, existing_faiss_index=None, num_k=3):
        """
        Initializes the AdvancedDocumentRetriever with a set of documents and an embedding model.

        Args:
            documents (list): A list of documents to be indexed and retrieved.
            embedding_model (OpenAIEmbeddings): The embedding model used for document vectorization.
        """
        try:
            self.embedding_model = embedding_model
            self.document_collection = documents
            self.faiss_index = existing_faiss_index
            self.k = num_k
            if not existing_faiss_index:
                if not documents:
                    raise ValueError("Document collection is empty.")

            if not self.faiss_index:
                # Index documents using FAISS
                self._index_documents()

            # Initialize FAISS retrievers
            if self.faiss_index:
                # self.faiss_retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.k})

                # The storage layer for the parent documents
                store = InMemoryStore()
                id_key = "doc_id"

                # The retriever (empty to start)
                self.faiss_retriever = MultiVectorRetriever(
                    vectorstore=self.faiss_index,
                    docstore=store,
                    id_key=id_key,
                )

            else:
                self.faiss_retriever = None


        except Exception as e:
            print(f"Initialization error in AdvancedDocumentRetriever: {e}")

    def _index_documents(self):
        """
        Indexes the provided documents into the FAISS index for efficient retrieval.
        Handles large document collections by segmenting them into smaller batches.
        """
        if self.faiss_index is None:  # Only index if there's no existing FAISS index
            try:
                if len(self.document_collection) < 1000:
                    self.faiss_index = FAISS.from_documents(self.document_collection, self.embedding_model)
                else:
                    self.faiss_index = FAISS.from_documents(self.document_collection[:1000], self.embedding_model)
                    for i in range(1000, len(self.document_collection), 1000):
                        end_index = min(i + 1000, len(self.document_collection))
                        additional_index = FAISS.from_documents(self.document_collection[i:end_index],
                                                                self.embedding_model)
                        self.faiss_index.merge_from(additional_index)
            except Exception as e:
                print(f"Error indexing documents: {e}")

    def save_index(self, file_path):
        """
        Saves the FAISS index to a specified file path.

        Args:
            file_path (str): Path where the FAISS index will be saved.
        """
        try:
            self.faiss_index.save_local(file_path)
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    def update_index(self, new_documents):
        """
        Updates the FAISS index with new documents.

        Args:
            new_documents (list): A list of new documents to add to the index.
        """
        # Add this method to handle updates to the existing index
        if not self.faiss_index:
            raise ValueError("FAISS index not initialized.")
        try:
            if len(new_documents) < 1000:
                new_index = FAISS.from_documents(new_documents, self.embedding_model)
            else:
                # self.faiss_index = FAISS.from_documents(self.document_collection[:2000], self.embedding_model)
                new_index = FAISS.from_documents(new_documents[:1000], self.embedding_model)
                for i in range(1000, len(new_documents), 1000):
                    end_index = min(i + 1000, len(new_documents))
                    additional_index = FAISS.from_documents(new_documents[i:end_index], self.embedding_model)
                    new_index.merge_from(additional_index)

            # new_index = FAISS.from_documents(new_documents, self.embedding_model)

            self.faiss_index.merge_from(new_index)

            # Update the document collection
            self.document_collection.extend(new_documents)

            self.faiss_retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.k})

        except Exception as e:
            print(f"Error Updating FAISS index: {e}")

    def delete_index(self, file_path):
        """
        Deletes the FAISS index directory from the specified path.

        Args:
            file_path (str): Path of the FAISS index directory to be deleted.
        """
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            else:
                print("FAISS index path does not exist.")
        except Exception as e:
            print(f"Error deleting FAISS index path: {e}")

    def retrieve_documents(self):
        """
        Retrieves relevant documents based on the provided query using the Ensemble Retriever.

        Args:
            query (str): Query string for retrieving relevant documents.

        Returns:
            A list of documents relevant to the query.
        """
        try:
            return self.faiss_retriever
        except Exception as e:
            print(f"Error retrieving documents: {e}")
