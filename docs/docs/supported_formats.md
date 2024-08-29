

## Supported Document Formats in LongTrainer

LongTrainer is designed to accommodate a wide range of document formats, enabling you to leverage diverse data sources to enrich your bot's conversational context. This guide details the supported formats and demonstrates how to add documents from different sources to your bot.

### Types of Supported Documents

LongTrainer can handle various document types, including structured and unstructured data, ensuring flexibility in how information is ingested and utilized. Here are the document types you can work with:

- **Text Documents**: Includes `.txt`, `.docx`, `.md` (Markdown), and more.
- **Data Sheets**: Such as `.csv`, `.xlsx`, and `.tsv` files.
- **Presentations**: Including `.ppt` and `.pptx`.
- **Web Pages**: HTML files and content extracted from URLs.
- **PDFs**: Comprehensive support for PDF documents.
- **Images**: Text extraction from image files.
- **Other Formats**: `.epub`, `.msg`, `.odt`, `.org`, `.rtf`, and `.rst`.

### Adding Documents to a Bot

You can add documents to your bot using one of several methods provided by LongTrainer, each tailored to different types of data sources:

#### 1. From Local and Network Paths

Add documents directly from file paths, supporting both structured and unstructured data.

**Example Usage**:

```python
from longtrainer.trainer import LongTrainer

# Initialize the trainer
trainer = LongTrainer()

# Create a new bot
bot_id = trainer.initialize_bot_id()

# Add documents from a local path
path = 'path/to/your/document.pdf'
trainer.add_document_from_path(path, bot_id, use_unstructured=True)
```

#### 2. From Web Links

Incorporate content directly from the internet by specifying URLs. This method is particularly useful for adding dynamic content from the web to your bot's knowledge base.

**Example Usage**:

```python
# List of web links
links = ['http://example.com/report1', 'http://example.com/data.csv']

# Add documents from these links
trainer.add_document_from_link(links, bot_id)
```

#### 3. From Search Queries

Utilize search queries to fetch and load content from platforms like Wikipedia, providing a rich source of information for your bot.

**Example Usage**:

```python
# Search query for Wikipedia
search_query = "Deep Learning"

# Add documents from a Wikipedia search
trainer.add_document_from_query(search_query, bot_id)
```

#### 4. Custom Loaders

For advanced use cases, you can use custom loaders to integrate specialized or proprietary data formats.

**Example Usage**:

```python
# Custom document loader
documents = custom_langchain_loader('path/to/data')

# Add these documents to the bot
trainer.pass_documents(documents, bot_id)
```

### Supported File Types for Unstructured Data

When loading unstructured data, LongTrainer can handle a variety of file types, ensuring you can work with data exactly as it exists within your organization.

**Supported Unstructured File Types**:

- `"csv", "doc", "docx", "epub", "image", "md", "msg", "odt", "org", "pdf", "ppt", "pptx", "rtf", "rst", "tsv", "xlsx"`

Each document added to your bot enhances its ability to understand and respond to queries more effectively, leveraging the rich context provided by diverse data sources.
