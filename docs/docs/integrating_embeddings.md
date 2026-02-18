## Integrating Embeddings

LongTrainer supports various embedding models for document vectorization. Pass a custom embedding model to the `LongTrainer` constructor or override per-bot via `create_bot()`.

### Default: OpenAI Embeddings

```python
from longtrainer.trainer import LongTrainer

trainer = LongTrainer()  # Uses OpenAIEmbeddings() by default
```

### OpenAI (Custom Model)

```python
from langchain_openai import OpenAIEmbeddings
from longtrainer.trainer import LongTrainer

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
trainer = LongTrainer(embedding_model=embeddings)
```

### AWS Bedrock Embeddings

```python
from langchain_aws import BedrockEmbeddings
from longtrainer.trainer import LongTrainer

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2",
    region_name="us-east-1",
)
trainer = LongTrainer(embedding_model=embeddings)
```

### HuggingFace Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
from longtrainer.trainer import LongTrainer

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
trainer = LongTrainer(embedding_model=embeddings)
```

### Google VertexAI Embeddings

```python
from langchain_google_vertexai import VertexAIEmbeddings
from longtrainer.trainer import LongTrainer

embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
trainer = LongTrainer(embedding_model=embeddings)
```

### Ollama Embeddings (Local)

```python
from langchain_ollama import OllamaEmbeddings
from longtrainer.trainer import LongTrainer

embeddings = OllamaEmbeddings(model="nomic-embed-text")
trainer = LongTrainer(embedding_model=embeddings)
```

### Per-Bot Embeddings

Each bot can use different embeddings, allowing mixed vector stores:

```python
from langchain_openai import OpenAIEmbeddings

trainer = LongTrainer()  # Global default: OpenAIEmbeddings()
bot_id = trainer.initialize_bot_id()
trainer.add_document_from_path("data.pdf", bot_id)

# This bot uses a different embedding model
trainer.create_bot(
    bot_id,
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
)
```
