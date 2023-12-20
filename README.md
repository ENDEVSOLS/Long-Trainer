
# LongTrainer - Production-Ready LangChain


## Features 🌟

- ✅ **Long Memory:** Retains context effectively for extended interactions.
- ✅ **Unique Bots/Chat Management:** Sophisticated management of multiple chatbots.
- ✅ **Enhanced Customization:** Tailor the behavior to fit specific needs.
- ✅ **Memory Management:** Efficient handling of chat histories and contexts.
- ✅ **GPT Vision Support:** Integration with GPT-powered visual models.
- ✅ **Different Data Formats:** Supports various data input formats.
- ✅ **VectorStore Management:** Advanced management of vector storage for efficient retrieval.

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
trainer = LongTrainer()
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
response = trainer.get_response(query, chat_id, bot_id)
print('Response: ', response)
  ```

## Maintainer 🛠️

This project is proudly maintained by **ENDEVSOLS**.

📧 Contact us at: [technology@endevsols.com](mailto:technology@endevsols.com)

---