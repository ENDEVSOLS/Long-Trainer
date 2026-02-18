# Installation

## Prerequisites

- Python 3.9+
- MongoDB running locally or remotely
- An OpenAI API key (or another LangChain-compatible LLM)

## Install LongTrainer

```bash
pip install longtrainer
```

### With Agent Mode Support (Optional)

Agent mode requires LangGraph for tool calling:

```bash
pip install longtrainer[agent]
```

## System Dependencies

Some document loaders require system-level libraries for parsing PDFs, images, and office files.

### Linux (Ubuntu/Debian)

```bash
sudo apt install libmagic-dev poppler-utils tesseract-ocr qpdf libreoffice pandoc
```

### macOS

```bash
brew install libmagic poppler tesseract qpdf libreoffice pandoc
```

!!! note
    System dependencies are only required if you plan to load documents using
    `use_unstructured=True` or process PDFs, images, and office formats.

## Verify Installation

```python
import longtrainer
print(longtrainer.__version__)  # Should print "1.0.0"

from longtrainer import LongTrainer, ToolRegistry, web_search
print("LongTrainer 1.0.0 installed successfully!")
```

## Environment Variables

Set your OpenAI API key before using LongTrainer:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

Or export it in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```
