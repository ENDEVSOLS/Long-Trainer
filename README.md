<p align="center">
  <img src="https://github.com/ENDEVSOLS/Long-Trainer/blob/master/assets/longtrainer-logo.png?raw=true" alt="LongTrainer Logo">
</p>

<h1 align="center">LongTrainer - Production-Ready LangChain</h1>

<p align="center">
  <a href="https://pypi.org/project/longtrainer/">
    <img src="https://img.shields.io/pypi/v/longtrainer" alt="PyPI Version">
  </a>
  <a href="https://pepy.tech/project/longtrainer">
    <img src="https://static.pepy.tech/badge/longtrainer" alt="Total Downloads">
  </a>
  <a href="https://pepy.tech/project/longtrainer">
    <img src="https://static.pepy.tech/badge/longtrainer/month" alt="Monthly Downloads">
  </a>
  <a href="https://colab.research.google.com/drive/1HE30D5q5onD8sfS50-06XPDXnbdvnjIy?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>

<hr />




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
trainer = LongTrainer(mongo_endpoint='mongodb://localhost:27017/', encrypt_chats=True)
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


## Citation
If you utilize this repository, please consider citing it with:

```
@misc{longtrainer,
  author = {Endevsols},
  title = {LongTrainer: Production-Ready LangChain},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ENDEVSOLS/Long-Trainer}},
}
```
