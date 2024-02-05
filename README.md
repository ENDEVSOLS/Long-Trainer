
# LongTrainer - Production-Ready LangChain


## Features 🌟

- ✅ **Long Memory:** Retains context effectively for extended interactions.
- ✅ **Unique Bots/Chat Management:** Sophisticated management of multiple chatbots.
- ✅ **Enhanced Customization:** Tailor the behavior to fit specific needs.
- ✅ **Memory Management:** Efficient handling of chat histories and contexts.
- ✅ **GPT Vision Support:** Integration Context Aware GPT-powered visual models.
- ✅ **Different Data Formats:** Supports various data input formats.
- ✅ **VectorStore Management:** Advanced management of vector storage for efficient retrieval.

## Works for All Langchain Supported LLM and Embeddings

- ✅ OpenAI (default)
- ✅ VertexAI
- ✅ HuggingFace

# Example

 VertexAI LLMs
```python
from longtrainer.trainer import LongTrainer
from langchain_community.llms import VertexAI

llm = VertexAI()

trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/', llm=llm)
```
 TogetherAI LLMs
```python
from longtrainer.trainer import LongTrainer
from langchain_community.llms import Together

llm = Together(
    model="togethercomputer/RedPajama-INCITE-7B-Base",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    # together_api_key="..."
)

trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/', llm=llm)

```

## Usage Example 🚀

```python
pip install longtrainer
```

Here's a quick start guide on how to use LongTrainer:

```python
from longtrainer.trainer import LongTrainer
import os
        
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-"
        
# Initialize LongTrainer
trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/')
bot_id = trainer.initialize_bot_id()
print('Bot ID: ', bot_id)
        
# Add Data
path = 'path/to/your/data'
trainer.add_document_from_path(path, bot_id)
        
# Initialize Bot
trainer.create_bot(bot_id)
        
# Start a New Chat
chat_id = trainer.new_chat(bot_id)
        
# Send a Query and Get a Response
query = 'Your query here'
response = trainer._get_response(query, bot_id, chat_id)
print('Response: ', response)
  ```

Here's a guide on how to use Vision Chat:

```python
chat_id = trainer.new_vision_chat(bot_id)

query = 'Your query here'
image_paths=['nvidia.jpg']
response = trainer._get_vision_response(query, image_paths, str(bot_id),str(vision_id))
print('Response: ', response)
```

List Chats and Display Chat History:

```python
trainer.list_chats(bot_id)

trainer.get_chat_by_id(chat_id=chat_id)
```

This project is still under active development. Community feedback and contributions are highly appreciated. 