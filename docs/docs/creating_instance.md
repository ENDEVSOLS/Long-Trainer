## Creating an Instance of LongTrainer

To start using LongTrainer, initialize an instance of the `LongTrainer` class. This instance manages all bots, chat sessions, document ingestion, and MongoDB persistence.

### Basic Initialization

```python
from longtrainer.trainer import LongTrainer

trainer = LongTrainer()
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mongo_endpoint` | `str` | `"mongodb://localhost:27017/"` | MongoDB connection string |
| `llm` | `BaseChatModel` | `ChatOpenAI(model="gpt-4o-2024-08-06")` | Default language model for all bots |
| `embedding_model` | `Embeddings` | `OpenAIEmbeddings()` | Default embedding model for document vectorization |
| `prompt_template` | `str` | Built-in prompt | System prompt template (must include `{context}` placeholder) |
| `max_token_limit` | `int` | `32000` | Token buffer limit for conversation memory |
| `num_k` | `int` | `3` | Number of documents to retrieve per query |
| `chunk_size` | `int` | `2048` | Text splitter chunk size |
| `chunk_overlap` | `int` | `200` | Text splitter overlap between chunks |
| `ensemble` | `bool` | `False` | Enable multi-query ensemble retrieval for better recall |
| `encrypt_chats` | `bool` | `False` | Enable Fernet encryption for stored chats |
| `encryption_key` | `bytes` | Auto-generated | Custom Fernet encryption key |

### Custom Configuration

```python
from longtrainer.trainer import LongTrainer
from langchain_openai import ChatOpenAI

trainer = LongTrainer(
    mongo_endpoint="mongodb://custom-host:27017/",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.3),
    max_token_limit=16000,
    num_k=5,
    chunk_size=1024,
    chunk_overlap=100,
    ensemble=True,
    encrypt_chats=True,
)
```

### Vector Store Selection (NEW in V2)

By default, LongTrainer uses local FAISS. You can easily connect to an enterprise vector database:
Supported names: `faiss`, `pinecone`, `chroma`, `qdrant`, `pgvector`, `mongodb`, `milvus`, `weaviate`, `elasticsearch`

```python
# Connect to MongoDB Atlas Vector Search
trainer = LongTrainer(
    vectorstore_provider="mongodb",
    vectorstore_kwargs={
        "connection_string": "mongodb+srv://<username>:<password>@cluster.mongodb.net",
        "index_name": "vector_index"
    }
)
```

### With Encryption

When `encrypt_chats=True`, all chat history stored in MongoDB is encrypted using Fernet symmetric encryption. You can provide your own key or let LongTrainer generate one:

```python
from cryptography.fernet import Fernet

key = Fernet.generate_key()

trainer = LongTrainer(
    encrypt_chats=True,
    encryption_key=key,
)

# Save the key — you'll need it to decrypt chats later
print(f"Encryption key: {key.decode()}")
```

!!! warning
    If you lose the encryption key, stored chats cannot be recovered.

### Per-Bot Overrides

While the constructor sets global defaults, each bot can override LLM, embeddings, and retrieval settings individually via `create_bot()`. See [Creating and Using a Bot](creating_using_bot.md) for details.
