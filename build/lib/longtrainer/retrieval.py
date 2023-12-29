from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import shutil
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

            if not documents:
                raise ValueError("Document collection is empty.")

            if not self.faiss_index:
                # Index documents using FAISS
                self._index_documents()

            # Initialize BM25 and FAISS retrievers
            self.bm25_retriever = BM25Retriever.from_documents(documents)

            if self.faiss_index:
                self.faiss_retriever = self.faiss_index.as_retriever(search_kwargs={"k": num_k})
            else:
                self.faiss_retriever = None

            # Create an Ensemble Retriever combining BM25 and FAISS
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.faiss_retriever],
                weights=[0.5, 0.5]
            )
        except Exception as e:
            print(f"Initialization error in AdvancedDocumentRetriever: {e}")

    def _index_documents(self):
        """
        Indexes the provided documents into the FAISS index for efficient retrieval.
        Handles large document collections by segmenting them into smaller batches.
        """
        if self.faiss_index is None:  # Only index if there's no existing FAISS index
            try:
                if len(self.document_collection) < 2000:
                    self.faiss_index = FAISS.from_documents(self.document_collection, self.embedding_model)
                else:
                    self.faiss_index = FAISS.from_documents(self.document_collection[:2000], self.embedding_model)
                    for i in range(2000, len(self.document_collection), 2000):
                        end_index = min(i + 2000, len(self.document_collection))
                        additional_index = FAISS.from_documents(self.document_collection[i:end_index], self.embedding_model)
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
        if len(new_documents) < 2000:
            new_index = FAISS.from_documents(new_documents, self.embedding_model)
        else:
            # self.faiss_index = FAISS.from_documents(self.document_collection[:2000], self.embedding_model)
            new_index = FAISS.from_documents(new_documents[:2000], self.embedding_model)
            for i in range(2000, len(new_documents), 2000):
                end_index = min(i + 2000, len(new_documents))
                additional_index = FAISS.from_documents(self.new_documents[i:end_index], self.embedding_model)
                new_index.merge_from(additional_index)

        new_index = FAISS.from_documents(new_documents, self.embedding_model)
        self.faiss_index.merge_from(new_index)


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
            return self.ensemble_retriever
        except Exception as e:
            print(f"Error retrieving documents: {e}")
