## Updating Bots

LongTrainer allows you to update bots with new documents, links, and prompt templates after creation.

### Update with New Documents

```python
trainer.update_chatbot(
    paths=["new_data/report.pdf", "new_data/notes.md"],
    bot_id=bot_id,
)
```

### Update with Links and Search Queries

```python
trainer.update_chatbot(
    paths=[],
    bot_id=bot_id,
    links=["https://example.com/latest-news"],
    search_query="Current trends in AI",
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `paths` | `list[str]` | File paths to load |
| `bot_id` | `str` | The bot's unique identifier |
| `links` | `list[str]` | Optional web URLs or YouTube links |
| `search_query` | `str` | Optional Wikipedia search query |
| `documents` | `list` | Optional pre-loaded LangChain documents |
| `prompt_template` | `str` | Optional new prompt template |
| `use_unstructured` | `bool` | Use UnstructuredLoader for files |

### Update Behavior

When updating a bot:

1. New documents are added to MongoDB
2. The FAISS index is updated incrementally (existing index is preserved)
3. The retriever is rebuilt with the expanded index
4. Existing chat sessions continue to work with the updated knowledge base

### Update the Prompt Template

Change the prompt without adding new documents:

```python
trainer.set_custom_prompt_template(
    bot_id,
    "You are a customer support agent. Use only the provided context. {context}"
)
```

### Full Update Example

```python
trainer.update_chatbot(
    paths=["reports/q4_2024.pdf"],
    bot_id=bot_id,
    links=["https://example.com/blog/update"],
    search_query="quarterly earnings reports",
    prompt_template="You are a financial analyst. {context}",
    use_unstructured=True,
)
print("Bot updated successfully.")
```
