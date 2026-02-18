## Supported Document Formats

LongTrainer handles a wide range of document formats, enabling diverse data sources for your bot's knowledge base.

### Supported File Types

| Format | Extensions | Loader |
|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` |
| Word | `.docx` | `Docx2txtLoader` |
| CSV | `.csv` | `CSVLoader` |
| HTML | `.html`, `.htm` | `BSHTMLLoader` |
| Markdown | `.md`, `.markdown` | `UnstructuredMarkdownLoader` |
| Plain Text | `.txt` | `UnstructuredMarkdownLoader` |
| Any format | All above + more | `UnstructuredFileLoader` (via `use_unstructured=True`) |

### Adding Documents

#### From File Paths

```python
# Auto-detected by extension
trainer.add_document_from_path("report.pdf", bot_id)
trainer.add_document_from_path("data.csv", bot_id)
trainer.add_document_from_path("notes.md", bot_id)

# Use UnstructuredLoader for any file type
trainer.add_document_from_path("presentation.pptx", bot_id, use_unstructured=True)
```

#### From Web Links

```python
# URLs
trainer.add_document_from_link(["https://example.com/article"], bot_id)

# YouTube videos (transcript extraction)
trainer.add_document_from_link(["https://youtube.com/watch?v=..."], bot_id)
```

#### From Wikipedia

```python
trainer.add_document_from_query("Artificial Intelligence", bot_id)
```

#### Pre-Loaded Documents

Pass pre-loaded LangChain `Document` objects directly:

```python
from langchain_core.documents import Document

documents = [Document(page_content="Custom content", metadata={"source": "manual"})]
trainer.pass_documents(documents, bot_id)
```

### Unstructured Data

When `use_unstructured=True`, LongTrainer uses LangChain's `UnstructuredFileLoader` which supports:

`csv`, `doc`, `docx`, `epub`, `image`, `md`, `msg`, `odt`, `org`, `pdf`, `pptx`, `rtf`, `rst`, `tsv`, `xlsx`

This requires system dependencies — see [Installation](installation.md).

### Text Splitting

All documents are automatically split into chunks for FAISS indexing:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `2048` | Maximum characters per chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |

Configure these when creating the `LongTrainer` instance:

```python
trainer = LongTrainer(chunk_size=1024, chunk_overlap=100)
```
