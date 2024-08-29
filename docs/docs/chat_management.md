
## Chat Management in LongTrainer

LongTrainer provides comprehensive tools for managing chat sessions, including the creation of new chats, vision-enabled chats, and viewing past chat interactions. This guide will demonstrate how to effectively use these tools to enhance interaction and manage conversations.

### Creating New Chat Sessions

LongTrainer allows you to start new chat sessions where each session is treated uniquely, maintaining its context independently of others. This helps in managing multiple ongoing conversations without overlap.

#### Start a New Text Chat

To begin a new text chat session, simply initialize a chat session and then send queries to get responses.

**Example Usage**:

```python
# Start a new chat session
chat_id = trainer.new_chat(bot_id)
        
# Send a query and get a response
query = 'Your query here'
response = trainer.get_response(query, bot_id, chat_id)
print('Response: ', response)
```

#### Start a New Vision Chat

For chats that require visual context, LongTrainer supports vision chat sessions. These sessions can process and respond based on the visual data provided.

**Example Usage**:

```python
# Initialize a new vision chat session
vision_chat_id = trainer.new_vision_chat(bot_id)

# Prepare your query and image paths
query = 'What is depicted in this image?'
image_paths = ['path/to/image.jpg']

# Send a vision query and get a response
response = trainer.get_vision_response(query, image_paths, bot_id, vision_chat_id)
print('Response: ', response)
```

### Managing Chat History

LongTrainer offers functionalities to list all chats and retrieve detailed histories of individual chat sessions, which is crucial for monitoring and improving the interaction quality over time.

#### List All Chats

You can list all chat sessions associated with a specific bot, providing a snapshot of each session’s initial interaction.

**Example Usage**:

```python
# List all chats for a specific bot
trainer.list_chats(bot_id)
```

#### Retrieve Detailed Chat History

To access the full conversation of a specific chat session, use the method that fetches detailed history, allowing review and analysis of past interactions.

**Example Usage**:

```python
# Retrieve and display the full conversation details of a specific chat session
detailed_chat = trainer.get_chat_by_id(chat_id)
for message in detailed_chat:
    print(f"Q: {message['question']}")
    print(f"A: {message['answer']}")
```

### Exporting Chat Histories

For further analysis or training, you may export chat histories to CSV files. This feature is helpful for data retention policies and training models on historical data.

**Example Usage**:

```python
# Export chat histories to a CSV file
exported_file_path = trainer._export_chats_to_csv(detailed_chat, bot_id)
print(f"Chat history exported to: {exported_file_path}")
```

### Training on Chat Data

To improve the conversational AI model, LongTrainer can train on new, unprocessed chats, continually enhancing the bot’s response quality.

**Example Usage**:

```python
# Train the bot on newly gathered chat data
training_result = trainer.train_chats(bot_id)
print(training_result)
```


