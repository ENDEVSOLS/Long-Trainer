## Chat Management

LongTrainer provides tools for managing text and vision chat sessions, viewing history, and training on past conversations.

### Creating Chat Sessions

#### Text Chat

```python
chat_id = trainer.new_chat(bot_id)

answer, sources = trainer.get_response("Your question here", bot_id, chat_id)
print(answer)
```

#### Vision Chat

For conversations that involve images:

```python
vision_chat_id = trainer.new_vision_chat(bot_id)

response, sources = trainer.get_vision_response(
    "What is depicted in this image?",
    image_paths=["path/to/image.jpg"],
    bot_id=bot_id,
    vision_chat_id=vision_chat_id,
)
print(response)
```

### Streaming Responses

#### Synchronous Streaming

```python
for chunk in trainer.get_response("Summarize this", bot_id, chat_id, stream=True):
    print(chunk, end="", flush=True)
```

#### Asynchronous Streaming

```python
async for chunk in trainer.aget_response("Explain this", bot_id, chat_id):
    print(chunk, end="", flush=True)
```

### Listing Chats

Retrieve all chat and vision chat IDs for a bot:

```python
chats = trainer.list_chats(bot_id)
print(f"Text chats: {chats['chat_ids']}")
print(f"Vision chats: {chats['vision_chat_ids']}")
```

### Retrieving Chat History

Get the full conversation history for a specific chat:

```python
# Newest messages first (default)
history = trainer.get_chat_by_id(chat_id)

# Oldest messages first
history = trainer.get_chat_by_id(chat_id, order="oldest")

for message in history:
    print(f"Q: {message['question']}")
    print(f"A: {message['answer']}")
    print()
```

Vision chat history:

```python
vision_history = trainer.get_vision_chat_by_id(vision_chat_id, order="oldest")

for message in vision_history:
    print(f"Q: {message['question']}")
    print(f"A: {message['response']}")
```

### Training on Chat Data

Improve the bot by feeding its own conversation history back into the knowledge base:

```python
result = trainer.train_chats(bot_id)
print(result["message"])
# "Trainer updated with new chat history."

print(f"Exported to: {result['csv_path']}")
```

This method:

1. Finds unprocessed chats (those not yet marked as `trained`)
2. Exports Q&A pairs to a CSV file
3. Adds the CSV as a new document to the bot's knowledge base
4. Rebuilds the bot with the updated index

### Direct Vector Store Access

Query the underlying FAISS index directly for similar documents:

```python
docs = trainer.invoke_vectorstore(bot_id, "search query")
for doc in docs:
    print(doc.page_content[:200])
```
