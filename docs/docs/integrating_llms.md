## Integrating LLMs

LongTrainer works with any LangChain-compatible language model. Pass a custom LLM to the `LongTrainer` constructor or to individual bots via `create_bot()`.

### Default

LongTrainer uses OpenAI by default:

```python
from longtrainer.trainer import LongTrainer

trainer = LongTrainer()  # Uses ChatOpenAI(model="gpt-4o-2024-08-06")
```

### OpenAI

```python
from langchain_openai import ChatOpenAI
from longtrainer.trainer import LongTrainer

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
trainer = LongTrainer(llm=llm)
```

### Anthropic

```python
from langchain_anthropic import ChatAnthropic
from longtrainer.trainer import LongTrainer

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
trainer = LongTrainer(llm=llm)
```

### Google VertexAI

```python
from langchain_google_vertexai import ChatVertexAI
from longtrainer.trainer import LongTrainer

llm = ChatVertexAI(model="gemini-1.5-pro")
trainer = LongTrainer(llm=llm)
```

### AWS Bedrock

```python
from langchain_aws import ChatBedrock
from longtrainer.trainer import LongTrainer

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5},
)
trainer = LongTrainer(llm=llm)
```

### Groq

```python
from langchain_groq import ChatGroq
from longtrainer.trainer import LongTrainer

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=1024,
    api_key="gsk_...",
)
trainer = LongTrainer(llm=llm)
```

### Together AI

```python
from langchain_together import ChatTogether
from longtrainer.trainer import LongTrainer

llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.7,
    max_tokens=512,
)
trainer = LongTrainer(llm=llm)
```

### Ollama (Local Models)

```python
from langchain_ollama import ChatOllama
from longtrainer.trainer import LongTrainer

llm = ChatOllama(model="llama3.2", temperature=0.3)
trainer = LongTrainer(llm=llm)
```

### Per-Bot LLM

Each bot can use a different LLM, overriding the global default:

```python
from langchain_openai import ChatOpenAI

trainer = LongTrainer()  # Global default: gpt-4o
bot_id = trainer.initialize_bot_id()
trainer.add_document_from_path("data.pdf", bot_id)

# This bot uses gpt-4o-mini instead
trainer.create_bot(bot_id, llm=ChatOpenAI(model="gpt-4o-mini"))
```
