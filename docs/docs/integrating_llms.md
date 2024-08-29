

## Integrating Multiple Large Language Models with LongTrainer

LongTrainer is designed to be flexible and extensible, supporting a variety of Large Language Models (LLMs) and embeddings. This allows users to tailor the AI capabilities of their applications to meet specific requirements and leverage the strengths of different AI models.

### Supported Large Language Models and Embeddings

LongTrainer currently supports the following LLMs:

- ✅ **OpenAI (default)**
- ✅ **VertexAI**
- ✅ **HuggingFace**
- ✅ **AWS Bedrock**
- ✅ **Groq**
- ✅ **TogetherAI**

Each of these models can be integrated seamlessly into your LongTrainer setup, providing specialized capabilities and enhancements to your applications.

### Example Integrations

#### VertexAI Integration

```python
from longtrainer.trainer import LongTrainer
from langchain_community.llms import VertexAI

# Initialize the VertexAI model
llm = VertexAI()

# Create a LongTrainer instance with VertexAI
trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/', llm=llm)
```

#### TogetherAI Integration

```python
from longtrainer.trainer import LongTrainer
from langchain_community.llms import Together

# Configure the TogetherAI model
llm = Together(
    model="togethercomputer/RedPajama-INCITE-7B-Base",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    # together_api_key="..."
)

# Create a LongTrainer instance with TogetherAI
trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/', llm=llm)
```

#### AWS Bedrock Integration

```python
from langchain_aws import ChatBedrock
from longtrainer.trainer import LongTrainer

# Initialize the AWS Bedrock model with specific settings
chat = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5}
)

# Set up LongTrainer with AWS Bedrock
trainer = LongTrainer(
    mongo_endpoint='mongodb://localhost:27017/',
    chunk_size=1024,
    encrypt_chats=False,
    llm=chat
)
```

#### Grok API Integration

```python
from longtrainer.trainer import LongTrainer
from langchain_community.llms import ChatGroq

# Configure the Grok API model
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=1024,
    model_kwargs={
        "top_p": 1,
        "stream": False
    },
    api_key='gsk_...'
)

# Integrate Grok API with LongTrainer
trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/', llm=llm)
```

