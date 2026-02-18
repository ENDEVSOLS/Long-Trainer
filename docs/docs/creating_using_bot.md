## Creating and Using a Bot

This guide walks through the full bot lifecycle: creating a bot, adding documents, starting chats, and getting responses.

### Step 1: Initialize a Bot

Create a new bot with a unique identifier:

```python
from longtrainer.trainer import LongTrainer

trainer = LongTrainer()
bot_id = trainer.initialize_bot_id()
print(f"Bot ID: {bot_id}")
```

### Step 2: Add Documents

Load documents into the bot's knowledge base before creating it:

```python
# From a file (PDF, DOCX, CSV, HTML, Markdown, TXT)
trainer.add_document_from_path("data/report.pdf", bot_id)

# From web links or YouTube
trainer.add_document_from_link(["https://example.com/article"], bot_id)

# From Wikipedia
trainer.add_document_from_query("Machine Learning", bot_id)

# Pre-loaded LangChain documents
trainer.pass_documents(my_documents, bot_id)
```

### Step 3: Create the Bot

Build the FAISS index and configure the bot:

```python
# RAG mode (default) — document Q&A
trainer.create_bot(bot_id)

# Agent mode — with tool calling
trainer.create_bot(bot_id, agent_mode=True)
```

#### Per-Bot Customization

Each bot can have its own LLM, embeddings, and retrieval settings:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

trainer.create_bot(
    bot_id,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    num_k=5,
    prompt_template="You are a legal assistant. {context}",
    agent_mode=True,
    tools=[web_search, my_tool],
)
```

### Step 4: Start a Chat

```python
chat_id = trainer.new_chat(bot_id)
```

### Step 5: Get Responses

#### Standard Response

```python
answer, sources = trainer.get_response("What is this document about?", bot_id, chat_id)
print(answer)
```

#### Streaming Response

```python
for chunk in trainer.get_response("Summarize the key points", bot_id, chat_id, stream=True):
    print(chunk, end="", flush=True)
```

#### Async Streaming

```python
async for chunk in trainer.aget_response("Explain the methodology", bot_id, chat_id):
    print(chunk, end="", flush=True)
```

#### With Web Search

```python
answer, web_sources = trainer.get_response(
    "What are the latest trends?", bot_id, chat_id, web_search=True
)
print(f"Sources: {web_sources}")
```

### Loading an Existing Bot

Restore a previously created bot from MongoDB and FAISS:

```python
trainer.load_bot(bot_id)

# All previous chats are restored automatically
chat_id = trainer.new_chat(bot_id)
answer, _ = trainer.get_response("Continue our conversation", bot_id, chat_id)
```

### Custom Prompt Templates

Change the system prompt at any time:

```python
trainer.set_custom_prompt_template(
    bot_id,
    "You are a medical assistant. Answer based on the provided context only. {context}"
)
```

!!! note
    Prompt templates must include the `{context}` placeholder where retrieved document context will be inserted.
