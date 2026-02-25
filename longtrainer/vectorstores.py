"""Vector Store Factory for LongTrainer.

Dynamically instantiates LangChain Vector Stores (FAISS, Pinecone, Chroma, Qdrant).
"""

import os
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def get_vectorstore(
    provider: str,
    embedding: Embeddings,
    collection_name: str,
    persist_directory: Optional[str] = None,
    **kwargs: Any,
) -> VectorStore:
    """Instantiate a VectorStore based on the provider string.

    Args:
        provider: 'faiss', 'chroma', 'pinecone', 'qdrant', etc.
        embedding: LangChain Embeddings model instance.
        collection_name: Name of the index/collection (typically bot_id).
        persist_directory: Local directory for FAISS/Chroma persistence.
        kwargs: Additional arguments for specific providers (urls, keys).

    Returns:
        A LangChain VectorStore instance.
    """
    provider = provider.lower()

    if provider == "faiss":
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        if persist_directory and os.path.exists(os.path.join(persist_directory, "index.faiss")):
            return FAISS.load_local(
                persist_directory, embedding, allow_dangerous_deserialization=True
            )

        # Create empty FAISS index
        try:
            embedding_dim = len(embedding.embed_query("hello"))
        except Exception:
            embedding_dim = 1536  # Fallback for OpenAI
            
        index = faiss.IndexFlatL2(embedding_dim)
        return FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    elif provider == "chroma":
        try:
            from langchain_chroma import Chroma
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                persist_directory=persist_directory,
                **kwargs,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain-chroma to use Chroma: pip install langchain-chroma"
            )

    elif provider == "pinecone":
        try:
            from pinecone import Pinecone
            from langchain_pinecone import PineconeVectorStore

            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is missing.")

            pc = Pinecone(api_key=api_key)
            index = pc.Index(collection_name)
            return PineconeVectorStore(embedding=embedding, index=index, **kwargs)
        except ImportError:
            raise ImportError(
                "Please install langchain-pinecone to use Pinecone: pip install langchain-pinecone pinecone-client"
            )

    elif provider == "qdrant":
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            url = kwargs.get("url", ":memory:")
            api_key = kwargs.get("api_key", None)

            client = QdrantClient(url=url, api_key=api_key)

            if not client.collection_exists(collection_name):
                try:
                    vector_size = len(embedding.embed_query("sample text"))
                except Exception:
                    vector_size = 1536
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )

            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain-qdrant to use Qdrant: pip install langchain-qdrant qdrant-client"
            )

    elif provider == "pgvector":
        try:
            from langchain_postgres import PGVector
            connection = kwargs.get("connection", "postgresql+psycopg://localhost:5432/postgres")
            return PGVector(
                embeddings=embedding,
                collection_name=collection_name,
                connection=connection,
                use_jsonb=kwargs.get("use_jsonb", True),
            )
        except ImportError:
            raise ImportError("Please install langchain-postgres: pip install langchain-postgres psycopg[binary]")

    elif provider == "mongodb":
        try:
            from langchain_mongodb import MongoDBAtlasVectorSearch
            from pymongo import MongoClient
            
            # Use provided kwargs or fallback to env/defaults
            mongo_uri = kwargs.get("mongo_uri") or os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            db_name = kwargs.get("db_name", "longtrainer")
            
            client = MongoClient(mongo_uri)
            collection = client[db_name][collection_name]
            
            return MongoDBAtlasVectorSearch(
                embedding=embedding,
                collection=collection,
                index_name=kwargs.get("index_name", "default"),
                relevance_score_fn="cosine",
            )
        except ImportError:
            raise ImportError("Please install langchain-mongodb: pip install langchain-mongodb pymongo")

    elif provider == "milvus":
        try:
            from langchain_milvus import Milvus
            uri = kwargs.get("uri", "./milvus_local.db") # Default to local Lite version
            return Milvus(
                embedding_function=embedding,
                collection_name=collection_name,
                connection_args={"uri": uri},
                auto_id=True
            )
        except ImportError:
            raise ImportError("Please install langchain-milvus: pip install langchain-milvus pymilvus")

    elif provider == "weaviate":
        try:
            from langchain_weaviate import WeaviateVectorStore
            import weaviate
            url = kwargs.get("url", "http://localhost:8080")
            api_key = kwargs.get("api_key", None)
            
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
            client = weaviate.Client(url=url, auth_client_secret=auth_config)
            
            return WeaviateVectorStore(
                client=client,
                index_name=collection_name.capitalize(), # Weaviate requires capitalized classes
                text_key="text",
                embedding=embedding,
            )
        except ImportError:
            raise ImportError("Please install langchain-weaviate: pip install langchain-weaviate weaviate-client")

    elif provider == "elasticsearch":
        try:
            from langchain_elasticsearch import ElasticsearchStore
            url = kwargs.get("url", "http://localhost:9200")
            api_key = kwargs.get("api_key", None)
            
            return ElasticsearchStore(
                index_name=collection_name.lower(), # ES requires lowercase
                embedding=embedding,
                es_url=url,
                es_api_key=api_key
            )
        except ImportError:
            raise ImportError("Please install langchain-elasticsearch: pip install langchain-elasticsearch")

    else:
        raise ValueError(f"Unsupported Vector Store provider: {provider}")


def save_vectorstore(vectorstore: VectorStore, provider: str, persist_directory: str) -> None:
    """Save the vector store to disk if the provider requires physical saving."""
    provider = provider.lower()
    if provider == "faiss":
        try:
            vectorstore.save_local(persist_directory)
        except Exception as e:
            print(f"[ERROR] Error saving FAISS index: {e}")
    # Chroma persists automatically, Pinecone and Qdrant are handled via cloud/client.


def delete_vectorstore(provider: str, collection_name: str, persist_directory: str) -> None:
    """Delete a vector store's underlying data."""
    provider = provider.lower()
    if provider in ("faiss", "chroma"):
        import shutil
        if os.path.exists(persist_directory):
            if os.path.isdir(persist_directory):
                shutil.rmtree(persist_directory)
            else:
                os.remove(persist_directory)
    elif provider == "pinecone":
        try:
            from pinecone import Pinecone
            api_key = os.environ.get("PINECONE_API_KEY")
            if api_key:
                pc = Pinecone(api_key=api_key)
                # Note: This will delete the ENTIRE Pinecone index.
                # In LongTrainer, we assume each bot is an index or partition.
                # For safety, we only attempt if we can.
                if collection_name in pc.list_indexes().names():
                    pc.delete_index(collection_name)
        except Exception as e:
            print(f"[ERROR] Error deleting Pinecone index: {e}")
    elif provider == "qdrant":
        try:
            from qdrant_client import QdrantClient
            # We assume in-memory or default local if no env vars.
            # In production, users should manage remote collections carefully.
            client = QdrantClient(path=persist_directory)
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
        except Exception as e:
            print(f"[ERROR] Error deleting Qdrant collection: {e}")
    elif provider == "pgvector":
        try:
            from langchain_postgres import PGVector
            # In a real scenario, you'd connect and drop the collection schema
            print(f"[INFO] PGVector deletion for {collection_name} requires direct DB access. Skipping auto-delete.")
        except Exception:
            pass
    elif provider == "mongodb":
        try:
            from pymongo import MongoClient
            mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            client = MongoClient(mongo_uri)
            # Assuming 'longtrainer' is the default db if not passed here
            client["longtrainer"].drop_collection(collection_name)
        except Exception as e:
            print(f"[ERROR] Error dropping MongoDB collection: {e}")
    elif provider == "milvus":
        try:
            from pymilvus import utility, connections
            # Assuming default local connection if uri isn't tracked here
            connections.connect(uri="./milvus_local.db") 
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
        except Exception as e:
            print(f"[ERROR] Error dropping Milvus collection: {e}")
    elif provider == "weaviate":
        try:
            import weaviate
            client = weaviate.Client("http://localhost:8080")
            class_name = collection_name.capitalize()
            if client.schema.exists(class_name):
                client.schema.delete_class(class_name)
        except Exception as e:
            print(f"[ERROR] Error dropping Weaviate class: {e}")
    elif provider == "elasticsearch":
        try:
            from elasticsearch import Elasticsearch
            es = Elasticsearch("http://localhost:9200")
            index_name = collection_name.lower()
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
        except Exception as e:
            print(f"[ERROR] Error dropping ES index: {e}")
