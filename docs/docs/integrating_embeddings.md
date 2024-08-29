

## Integrating Different Embeddings in LongTrainer

LongTrainer supports various embedding models that can be utilized to enhance the capabilities of your language models by providing efficient, context-aware text representations. This guide covers how to configure and use different embeddings within LongTrainer, including the default OpenAI embeddings as well as Bedrock and HuggingFace options.

### Default Embedding: OpenAI

OpenAI embeddings are the default option in LongTrainer, known for their robust performance and general applicability across various text processing tasks. There is no need for additional configuration if you choose to use the default settings.

Example for default usage:
```python
from longtrainer.trainer import LongTrainer
import os

os.environ["OPENAI_API_KEY"] = 'sk-'

# Initialize LongTrainer with default OpenAI embeddings
trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/')
```

### AWS Bedrock Embeddings

For those utilizing AWS services, Bedrock Embeddings offer a highly optimized solution tailored for efficient large-scale embedding tasks. These embeddings are particularly useful for applications requiring integration with AWS ecosystems and data services.

#### Configuration Example for Bedrock Embeddings

```python
from langchain_aws import BedrockEmbeddings
from longtrainer.trainer import LongTrainer

# Configure AWS Bedrock embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2", 
    region_name="us-east-1"
)

# Set up LongTrainer with Bedrock Embeddings
trainer = LongTrainer(
    mongo_endpoint='mongodb://localhost:27017/',
    chunk_size=1024,
    encrypt_chats=False,
    embedding_model=bedrock_embeddings
)
```

### HuggingFace Embeddings

HuggingFace offers a diverse range of pre-trained embedding models available through its platform. These embeddings are suitable for those who are looking for specific linguistic traits or need embeddings trained on particular types of data or languages.

#### Configuration Example for HuggingFace Embeddings

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from longtrainer.trainer import LongTrainer

# Initialize HuggingFace embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create a LongTrainer instance with HuggingFace embeddings
trainer = LongTrainer(
    mongo_endpoint='mongodb://localhost:27017/',
    chunk_size=1024,
    encrypt_chats=False,
    embedding_model=embeddings
)
```
