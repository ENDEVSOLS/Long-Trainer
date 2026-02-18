## Deleting Bots

Remove a bot and all associated data from LongTrainer.

### Delete a Bot

```python
trainer.delete_chatbot(bot_id)
```

### What Gets Deleted

The `delete_chatbot()` method removes:

1. **MongoDB records:** All documents, chat history, and vision chat history for the bot
2. **FAISS index:** The vector store index files on disk
3. **Data files:** Any exported CSV files in the bot's data folder
4. **In-memory state:** All bot references and chat sessions

### Error Handling

```python
try:
    trainer.delete_chatbot(bot_id)
    print("Bot deleted successfully.")
except ValueError as e:
    print(f"Bot not found: {e}")
```

The method raises a `ValueError` if the provided `bot_id` is not found in the current session's bot data.

!!! warning
    Deletion is **irreversible**. All associated data — documents, chats, FAISS index, and exported files — will be permanently removed.

### Best Practices

- **Back up data** before deleting a bot if chat histories or documents may be needed later
- **Export chats** using `trainer.get_chat_by_id()` before deletion if you need to preserve conversations
- **Verify the bot ID** before calling delete to avoid accidental removal
