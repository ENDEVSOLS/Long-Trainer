
## Updating Bots in LongTrainer

LongTrainer allows you to seamlessly update your bots with new documents, links, search queries, and prompt templates. This capability ensures that your bot remains relevant and equipped with the latest information, enhancing its conversational abilities and accuracy.



### Parameters Description

- **paths (list)**: List of file paths from which documents are loaded. These can include local or networked file locations.
- **bot_id (str)**: The unique identifier for the bot that is being updated.
- **links (list, optional)**: List of URLs from which to fetch and load content directly into the bot's database.
- **search_query (str, optional)**: A query string used to perform a search, typically on the internet or a specific database like Wikipedia, to gather content.
- **prompt_template (str, optional)**: A new or modified prompt template to customize the bot's conversational prompts.
- **use_unstructured (bool)**: Specifies whether to use unstructured data loaders for documents that do not follow a fixed format or schema.


### Example Usage

```python
# Define paths to new documents and other update parameters
paths = ['new_data/reports.pdf', 'new_data/summary.txt']
links = ['http://example.com/latest-news', 'http://example.com/data']
search_query = "current trends in AI"

# Update the bot
trainer.update_chatbot(
    paths=paths,
    bot_id='your_bot_id',
    links=links,
    search_query=search_query,
    prompt_template="Your updated prompt template here",
    use_unstructured=True
)

# Confirm the bot has been updated
print("Bot updated successfully with new data and configurations.")
```
