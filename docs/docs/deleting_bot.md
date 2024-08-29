
## Deleting Bots in LongTrainer

Managing the lifecycle of bots within LongTrainer includes the capability to securely and thoroughly remove a bot and all its associated data. This is crucial for maintaining data hygiene and ensuring that obsolete or redundant bots do not clutter your system.

### Functionality Overview

The `delete_chatbot` method is designed to remove a bot from the system completely, ensuring that all its data, including documents, chats, and any other related records, are permanently deleted.

**Method Signature**:

```python
def delete_chatbot(self, bot_id):
```

### Parameters Description

- **bot_id (str)**: The unique identifier for the bot that you intend to delete.

### Deletion Process

The deletion process follows several steps to ensure that all traces of the bot are removed:

1. **Verify Bot Existence**:
   - The method first checks if the bot exists in the system using the provided `bot_id`. If the bot does not exist, it raises an exception.

2. **Delete Database Entries**:
   - All documents and data entries associated with the bot are deleted from their respective MongoDB collections (`chats`, `vision_chats`, `bots`).

3. **Remove Data Files**:
   - Any files or data specific to the bot stored on the filesystem, such as cached data or exported CSV files, are removed.

4. **Cleanup In-memory Data**:
   - References and objects held in memory that pertain to the bot are cleared to free up resources and prevent any data leaks.

### Error Handling

- The method includes error handling to manage situations where the bot ID does not exist or there are issues accessing the filesystem to delete files.

### Example Usage

```python
# Specify the bot ID to delete
bot_id_to_delete = 'your_bot_id_here'

# Attempt to delete the bot
try:
    trainer.delete_chatbot(bot_id_to_delete)
    print("Bot successfully deleted.")
except Exception as e:
    print(f"Failed to delete bot: {e}")
```

### Considerations

- **Data Backup**: Before deleting a bot, consider backing up any important data or documents associated with the bot if they might be needed later.
- **Irreversibility**: Deletion is irreversible. Once a bot is deleted, it cannot be recovered, so ensure that deletion is the desired action before proceeding.
