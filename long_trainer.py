from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from document_loaders import DocumentLoader, TextSplitter
from doc_retrieval import DocRetriever
from chain_bot import ChainBot
class LongTrainer:
    """
    ExpertAssistantTrainer is designed to train and manage a conversational AI assistant,
    capable of handling document retrieval and providing responses based on a given context.
    """

    def __init__(self, llm=None, embedding_model=None, prompt_template=None, max_token_limit=2000):
        """
        Initialize the LongTrainer with optional language learning model, embedding model,
        prompt template, and maximum token limit.

        Args:
            llm: Language learning model, defaults to ChatOpenAI with GPT-4.
            embedding_model: Embedding model for document vectorization, defaults to OpenAIEmbeddings.
            prompt_template: Template for generating prompts, defaults to a predefined template.
            max_token_limit (int): Maximum token limit for the conversational buffer.
        """
        self.llm = llm if llm is not None else ChatOpenAI(model_name='gpt-4-1106-preview')
        self.embedding_model = embedding_model if embedding_model is not None else OpenAIEmbeddings()
        self.prompt_template = prompt_template if prompt_template is not None else self._default_prompt_template()
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "chat_history", "question"])
        self.max_token_limit = max_token_limit
        self.documents = []
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=1024, chunk_overlap=100)
        self.conversational_chain = None

    def _default_prompt_template(self):
        """
        Returns the default prompt template for the assistant.
        """
        return """
        As a General Expert Assistant:

        {context}
        Use the following pieces of information to answer the user's question.
        If the answer is unknown, admitting ignorance is preferred over fabricating a response.
        Answers should be direct, professional, and to the point without any irrelevant details.
        Assistant must focus solely on the provided question, considering the chat history for context.
        Chat History: {chat_history}
        Question: {question}
        Answer:
          """

    def add_document_from_path(self, path):
        """
        Loads and adds documents from a specified file path.

        Args:
            path (str): Path to the document file.
        """
        try:
            file_extension = path.split('.')[-1].lower()
            if file_extension == 'csv':
                documents = self.document_loader.load_csv(path)
            elif file_extension == 'docx':
                documents = self.document_loader.load_doc(path)
            elif file_extension == 'pdf':
                documents = self.document_loader.load_pdf(path)
            elif file_extension in ['md', 'markdown', 'txt']:
                documents = self.document_loader.load_markdown(path)
            elif file_extension in ['html', 'htm']:
                documents = self.document_loader.load_text_from_html(path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            self.documents.extend(documents)
        except Exception as e:
            print(f"Error adding document from path: {e}")

    def add_document_from_link(self, links):
        """
        Loads and adds documents from provided web links.

        Args:
            links (list): List of web links to load documents from.
        """
        try:
            for link in links:
                if 'youtube.com' in link.lower() or 'youtu.be' in link.lower():
                    documents = self.document_loader.load_YouTubeVideo(link)
                else:
                    documents = self.document_loader.load_urls([link])
                self.documents.extend(documents)
        except Exception as e:
            print(f"Error adding document from link: {e}")

    def add_document_from_query(self, search_query):
        """
        Loads and adds documents from a Wikipedia search query.

        Args:
            search_query (str): Search query for Wikipedia.
        """
        try:
            query_documents = self.document_loader.wikipedia_query(search_query)
            self.documents.extend(query_documents)
        except Exception as e:
            print(f"Error adding document from query: {e}")

    def create_bot(self):
        """
        Creates and returns a conversational AI assistant based on the loaded documents.

        Returns:
            A function that takes a query and returns the assistant's response.
        """
        try:
            all_splits = self.text_splitter.split_documents(self.documents)
            retriever = DocRetriever(all_splits, self.embedding_model)
            ensemble_retriever = retriever.retrieve_documents()

            bot = ChainBot(retriever=ensemble_retriever, llm=self.llm, prompt=self.prompt, token_limit=self.max_token_limit)
            self.conversational_chain = bot.get_chain()

            return lambda query: self._get_response(query)
        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def _get_response(self, query):
        """
        Retrieves a response from the conversational AI assistant for a given query.

        Args:
            query (str): Query string for the assistant.

        Returns:
            The assistant's response to the query.
        """
        if not self.conversational_chain:
            raise Exception("Conversational chain not created. Please call create_bot first.")
        return self.conversational_chain(query)
