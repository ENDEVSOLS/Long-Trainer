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
<p align="center">
  <a href="https://endevsols.com/longtrainer-the-next-evolution-in-production-ready-langchain-frameworks/">
    Visit Blog Post
  </a>
</p>

<p align="center">
  <a class="github-button" href="https://github.com/ENDEVSOLS/Long-Trainer" data-color-scheme="no-preference: light; light: light; dark: dark;" data-size="large" data-show-count="true" aria-label="Star ENDEVSOLS/Long-Trainer on GitHub">Star</a>
  <a class="github-button" href="https://github.com/ENDEVSOLS" data-color-scheme="no-preference: light; light: light; dark: dark;" data-size="large" data-show-count="true" aria-label="Follow @ENDEVSOLS on GitHub">Follow @ENDEVSOLS</a>
</p>


# Official Documentation

Explore the comprehensive [LongTrainer Documentation](https://endevsols.github.io/Long-Trainer/) for detailed
instructions on installation, features, and API usage.

# Installation

Introducing LongTrainer, a sophisticated extension of the LangChain framework designed specifically for managing
multiple bots and providing isolated, context-aware chat sessions. Ideal for developers and businesses looking to
integrate complex conversational AI into their systems, LongTrainer simplifies the deployment and customization of LLMs.

```markdown
pip install longtrainer
```

## Installation Instructions for Required Libraries and Tools

### 1. Linux (Ubuntu/Debian)

To install the required packages on a Linux system (specifically Ubuntu or Debian), you can use the apt package manager.
The following command installs several essential libraries and tools:

```markdown
sudo apt install libmagic-dev poppler-utils tesseract-ocr qpdf libreoffice pandoc
```

### 2. macOS

On macOS, you can install these packages using brew, the Homebrew package manager. If you don't have Homebrew installed,
you can install it from brew.sh.

```markdown
brew install libmagic poppler tesseract qpdf libreoffice pandoc
```

# Features 🌟

- ✅ **Long Memory:** Retains context effectively for extended interactions.
- ✅ **Multi-Bot Management:** Easily configure and manage multiple bots within a single framework, perfect for scaling
  across various use cases
- ✅ **Isolated Chat Sessions:** Each bot operates within its own session, ensuring interactions remain distinct and
  contextually relevant without overlap.
- ✅ **Context-Aware Interactions:**  Utilize enhanced memory capabilities to maintain context over extended dialogues,
  significantly improving user experience
- ✅ **Scalable Architecture:** Designed to scale effortlessly with your needs, whether you're handling hundreds of users
  or just a few.
- ✅ **Enhanced Customization:** Tailor the behavior to fit specific needs.
- ✅ **Memory Management:** Efficient handling of chat histories and contexts.
- ✅ **GPT Vision Support:** Integration Context Aware GPT-powered visual models.
- ✅ **Different Data Formats:** Supports various data input formats.
- ✅ **VectorStore Management:** Advanced management of vector storage for efficient retrieval.

## Diverse Use Cases:

- ✅ **Enterprise Solutions:** Streamline customer interactions, automate responses, and manage multiple departmental
  bots from a single platform.
- ✅ **Educational Platforms:** Enhance learning experiences with AI tutors capable of maintaining context throughout
  sessions.
- ✅ **Healthcare Applications:** Support patient management with bots that provide consistent, context-aware
  interactions.

## Works for All Langchain Supported LLM and Embeddings

- ✅ OpenAI (default)
- ✅ VertexAI
- ✅ HuggingFace
- ✅ AWS Bedrock
- ✅ Groq
- ✅ TogetherAI

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
response = trainer.get_response(query, bot_id, chat_id)
print('Response: ', response)
  ```

Here's a guide on how to use Vision Chat:

```python
chat_id = trainer.new_vision_chat(bot_id)

query = 'Your query here'
image_paths = ['nvidia.jpg']
response = trainer.get_vision_response(query, image_paths, str(bot_id), str(vision_id))
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

