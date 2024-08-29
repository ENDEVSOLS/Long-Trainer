
## Creating an Instance of LongTrainer

To start using LongTrainer, you need to initialize an instance of the `LongTrainer` class. This instance will allow you to manage chatbots, handle conversational states, and interact with your MongoDB to store and retrieve data securely and efficiently.

### Initialization Code Snippet

```python
from longtrainer.trainer import LongTrainer

# Initialize the LongTrainer with default parameters
trainer = LongTrainer()
```

### Constructor Parameters

The `LongTrainer` constructor accepts several parameters that allow you to customize its behavior to fit your specific needs:

- **`mongo_endpoint` (str)**: The connection string URL to your MongoDB instance. This parameter defaults to `'mongodb://localhost:27017/'`, pointing to a MongoDB running on the default port on your local machine. Specify a different URL if your MongoDB instance is hosted elsewhere or configured differently.

- **`llm` (Any)**: The language learning model (LLM) used for processing queries and generating responses. By default, this is set to `None`, which means the system will use `ChatOpenAI` with a GPT-4-Turbo model. You can pass any compatible LLM instance according to your project's requirements.

- **`embedding_model` (Any)**: This parameter allows you to specify the embedding model used for document vectorization. The default is `OpenAIEmbeddings`, which is optimized for general-purpose language understanding.

- **`prompt_template` (Any)**: A template used to generate prompts that guide the LLM in generating appropriate responses. The system uses a predefined template by default, but you can customize this to better suit the nuances of your specific application.

- **`max_token_limit` (int)**: Specifies the maximum number of tokens (words and characters) that the LLM can handle in a single query. This is set to `32000` by default, which is typically sufficient for most conversational applications.

- **`num_k` (int)**: Defines the number of top results (`k`) retrieved by the document retriever during the search process. The default value is `3`, balancing performance and relevance.

- **`chunk_size` (int)**: Determines the size of the text chunks that the `TextSplitter` processes. The default size is `2048`, which affects how text is segmented for processing.

- **`chunk_overlap` (int)**: Defines the overlap size between consecutive text chunks processed by the `TextSplitter`. This parameter is set to `200` by default, ensuring that context isn't lost at chunk boundaries.

- **`encrypt_chats` (Bool)**: A boolean flag that indicates whether chat data should be encrypted before being stored in MongoDB. This is set to `False` by default for development ease, but it is recommended to enable encryption (`True`) in production environments to ensure data privacy.

- **`encryption_key` (Any)**: This parameter is used to initialize the encryption algorithm (Fernet) if `encrypt_chats` is set to `True`. It should be a secure key that remains confidential. By default, it is `None`, and a key will be generated automatically if encryption is enabled.

### Example with Custom Parameters

```python
from longtrainer.trainer import LongTrainer

# Customized LongTrainer initialization
trainer = LongTrainer(
    mongo_endpoint='mongodb://custom-host:27017/',
    max_token_limit=4096,
    num_k=3,
    chunk_size=1024,
    chunk_overlap=100,
    encrypt_chats=False,
)
```
