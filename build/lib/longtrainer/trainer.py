from longtrainer.loaders import DocumentLoader, TextSplitter
from longtrainer.retrieval  import DocRetriever
from longtrainer.bot import ChainBot
from longtrainer.vision_bot import VisionMemory, VisionBot
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import sqlite3
import uuid


class LongTrainer:


    def __init__(self, llm=None, embedding_model=None, prompt_template=None, max_token_limit=32000):
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
        self.prompt = PromptTemplate(template=self.prompt_template,
                                     input_variables=["context", "chat_history", "question"])
        self.max_token_limit = max_token_limit
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=1024, chunk_overlap=100)
        self.bot_data = {}
        self.database_path = 'long_trainer.db'
        self._initialize_database()

    def _initialize_database(self):
        # Connect to SQLite database (or create if not exists)
        self.conn = sqlite3.connect(self.database_path)
        cursor = self.conn.cursor()

        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bots (
                bot_id TEXT PRIMARY KEY,
                faiss_path TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT,
                chat_id TEXT,
                question TEXT,
                answer TEXT,
                FOREIGN KEY (bot_id) REFERENCES bots (bot_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vision_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT,
                vision_chat_id TEXT,
                image_path TEXT,
                question TEXT,
                response TEXT,
                FOREIGN KEY (bot_id) REFERENCES bots (bot_id)
            )

        ''')
        self.conn.commit()

    def initialize_bot_id(self):
        bot_id = str(uuid.uuid4())
        self.bot_data[bot_id] = {
            'documents': [],
            'chains': {},
            'assistants': {},
            'retriever': None,
            'ensemble_retriever': None,
            'conversational_chain': None,
            'faiss_path': f'faiss_index_{bot_id}',
            'assistant': None
        }
        # Insert data into the bots table
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO bots (bot_id, faiss_path) VALUES (?, ?)",
                       (bot_id, self.bot_data[bot_id]['faiss_path']))
        self.conn.commit()

        return bot_id

    def _default_prompt_template(self):
        """
        Returns the default prompt template for the assistant.
        """
        return """
        As a General Expert Assistant:

        {context}
        Use the following pieces of information to answer the user's question. If the answer is unknown, admitting ignorance is preferred over fabricating a response. Dont need toa dd irrelevant text explanation in response.
        Answers should be direct, professional, and to the point without any irrelevant details.
        Assistant must focus solely on the provided question, considering the chat history for context.
        Chat History: {chat_history}
        Question: {question}
        Answer:
        """

    def add_document_from_path(self, path, bot_id):

        """
        Loads and adds documents from a specified file path.

        Args:
            path (str): Path to the document file.
        """
        try:
            if bot_id in self.bot_data:
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
                self.bot_data[bot_id]['documents'].extend(documents)

            # self.documents.extend(documents)
        except Exception as e:
            print(f"Error adding document from path: {e}")

    def add_document_from_link(self, links, bot_id):
        """
        Loads and adds documents from provided web links.

        Args:
            links (list): List of web links to load documents from.
        """
        try:
            if bot_id in self.bot_data:
                for link in links:
                    if 'youtube.com' in link.lower() or 'youtu.be' in link.lower():
                        documents = self.document_loader.load_YouTubeVideo(link)
                    else:
                        documents = self.document_loader.load_urls([link])

                    self.bot_data[bot_id]['documents'].extend(documents)

                # self.documents.extend(documents)
        except Exception as e:
            print(f"Error adding document from link: {e}")

    def add_document_from_query(self, search_query, bot_id):
        """
        Loads and adds documents from a Wikipedia search query.

        Args:
            search_query (str): Search query for Wikipedia.
        """
        try:
            if bot_id in self.bot_data:
                query_documents = self.document_loader.wikipedia_query(search_query)
                self.bot_data[bot_id]['documents'].extend(query_documents)

        except Exception as e:
            print(f"Error adding document from query: {e}")

    def create_bot(self, bot_id):
        """
        Creates and returns a conversational AI assistant based on the loaded documents.

        Returns:
            A function that takes a query and returns the assistant's response.
        """
        try:
            if bot_id in self.bot_data:
                all_splits = self.text_splitter.split_documents(self.bot_data[bot_id]['documents'])
                self.bot_data[bot_id]['retriever'] = DocRetriever(all_splits, self.embedding_model,
                                                                  existing_faiss_index=self.bot_data[bot_id][
                                                                      'retriever'].faiss_index if self.bot_data[bot_id][
                                                                      'retriever'] else None)
                self.bot_data[bot_id]['retriever'].save_index(file_path=self.bot_data[bot_id]['faiss_path'])
                self.bot_data[bot_id]['ensemble_retriever'] = self.bot_data[bot_id]['retriever'].retrieve_documents()

        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def new_chat(self, bot_id):
        try:
            chat_id = str(uuid.uuid4())
            bot = ChainBot(retriever=self.bot_data[bot_id]['ensemble_retriever'], llm=self.llm, prompt=self.prompt,
                           token_limit=self.max_token_limit)
            self.bot_data[bot_id]['conversational_chain'] = bot.get_chain()
            self.bot_data[bot_id]['chains'][chat_id] = self.bot_data[bot_id]['conversational_chain']
            return chat_id
        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def new_vision_chat(self, bot_id):
        try:
            vision_chat_id = str(uuid.uuid4())
            self.bot_data[bot_id]['assistant'] = VisionMemory(self.max_token_limit,
                                                              self.bot_data[bot_id]['ensemble_retriever'])
            self.bot_data[bot_id]['assistants'][vision_chat_id] = self.bot_data[bot_id]['assistant']

            return vision_chat_id
        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def update_chatbot(self, paths, links, search_query, bot_id):
        print(len(self.bot_data[bot_id]['documents']))

        # Initial length of documents
        initial_docs_len = len(self.bot_data[bot_id]['documents'])

        # Add new documents from paths, links, and query
        for path in paths:
            self.add_document_from_path(path, bot_id)

        self.add_document_from_link(links, bot_id)  # Assuming links is a list of lists

        if search_query:
            self.add_document_from_query(search_query, bot_id)
        print(len(self.bot_data[bot_id]['documents']))
        # Calculate new documents added
        new_docs = self.bot_data[bot_id]['documents'][initial_docs_len:]

        if self.bot_data[bot_id]['retriever'] and self.bot_data[bot_id]['retriever'].faiss_index:
            print(len(new_docs))
            # Use the new method to update the existing index
            self.bot_data[bot_id]['retriever'].update_index(new_docs)
            self.bot_data[bot_id]['retriever'].save_index(file_path=self.bot_data[bot_id]['faiss_path'])

        else:
            # If no retriever or FAISS index, create a new one
            self.retriever = DocRetriever(self.documents, self.embedding_model,
                                          existing_faiss_index=self.bot_data[bot_id]['retriever'].faiss_index if
                                          self.bot_data[bot_id]['retriever'] else None)

        self.create_bot(bot_id)

    def _get_response(self, query, chat_id, bot_id):
        """
        Retrieves a response from the conversational AI assistant for a given query.

        Args:
            query (str): Query string for the assistant.

        Returns:
            The assistant's response to the query.
        """
        if bot_id not in self.bot_data or chat_id not in self.bot_data[bot_id]['chains']:
            raise Exception(f"Bot ID {bot_id} or Chat ID {chat_id} not found")

        chain = self.bot_data[bot_id]['chains'][chat_id]
        result = chain(query)
        result = result.get('answer')

        # Insert data into the chats table
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO chats (bot_id, chat_id, question, answer) VALUES (?, ?, ?, ?)",
                       (bot_id, chat_id, query, result))
        self.conn.commit()

        return result

    def _get_vision_response(self, query, image_paths, vision_chat_id, bot_id):

        if bot_id not in self.bot_data or vision_chat_id not in self.bot_data[bot_id]['assistants']:
            raise Exception(f"Bot ID {bot_id} or Vision Chat ID {vision_chat_id} not found")

        assistant = self.bot_data[bot_id]['assistants'][vision_chat_id]
        prompt = assistant.get_answer(query)
        vision = VisionBot(prompt)
        vision.create_vision_bot(image_paths)
        vision_response = vision.get_response(query)
        assistant.save_chat_history(query, vision_response)
        # Insert data into the vision_chats table
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO vision_chats (bot_id, vision_chat_id, image_path, question, response) VALUES (?, ?, ?, ?, ?)",
            (bot_id, vision_chat_id, ','.join(image_paths), query, vision_response))
        self.conn.commit()

        return vision_response

    def delet_chatbot(self, bot_id):
        if bot_id in self.bot_data:
            cursor = self.conn.cursor()

            # Delete related rows from the chats table
            cursor.execute("DELETE FROM chats WHERE bot_id = ?", (bot_id,))

            # Delete related rows from the vision_chats table
            cursor.execute("DELETE FROM vision_chats WHERE bot_id = ?", (bot_id,))

            # Delete row from the bots table
            cursor.execute("DELETE FROM bots WHERE bot_id = ?", (bot_id,))

            # Commit the changes
            self.conn.commit()

            # Additional in-memory cleanup if needed
            bot = self.bot_data[bot_id]
            if bot['retriever']:
                bot['retriever'].delete_index(file_path=bot['faiss_path'])
            del self.bot_data[bot_id]
        else:
            raise Exception(f"Bot ID {bot_id} not found")



