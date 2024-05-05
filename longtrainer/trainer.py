import os
import re
import uuid
import shutil
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from cryptography.fernet import Fernet
from longtrainer.loaders import DocumentLoader, TextSplitter
from longtrainer.retrieval import DocRetriever
from longtrainer.bot import ChainBot
from longtrainer.vision_bot import VisionMemory, VisionBot
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from longtrainer.utils import serialize_document, deserialize_document


class LongTrainer:

    def __init__(
            self,
            mongo_endpoint='mongodb://localhost:27017/',
            llm=None,
            embedding_model=None,
            prompt_template=None,
            max_token_limit=32000,
            num_k=3,
            chunk_size=2048,
            chunk_overlap=200,
            encrypt_chats=False,
            encryption_key=None
    ):
        """
        Initialize the LongTrainer with optional language learning model, embedding model,
        prompt template, maximum token limit, and MongoDB endpoint, num_k, encryption_configs

        Args:
            mongo_endpoint (str): MongoDB connection string.
            llm: Language learning model, defaults to ChatOpenAI with GPT-4.
            embedding_model: Embedding model for document vectorization, defaults to OpenAIEmbeddings.
            prompt_template: Template for generating prompts, defaults to a predefined template.
            max_token_limit (int): Maximum token limit for the conversational buffer.
            num_k (int): Define no of K for Retriever.
            chunk_size (int): Chunk Size for Splitter.
            chunk_overlap (int): Chunk Overlap size for Splitter.
            encrypt_chats (Bool): Whether to use encryption while storing Chats in Mongodb or not.
            encryption_key : For initializing Fernet.

        """

        self.llm = llm if llm is not None else ChatOpenAI(model_name='gpt-4-1106-preview')
        self.embedding_model = embedding_model if embedding_model is not None else OpenAIEmbeddings()
        self.prompt_template = prompt_template if prompt_template is not None else self._default_prompt_template()
        self.prompt = PromptTemplate(template=self.prompt_template,
                                     input_variables=["context", "chat_history", "question"])
        self.search = DuckDuckGoSearchResults()
        self.k = num_k
        self.max_token_limit = max_token_limit
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.bot_data = {}

        # MongoDB setup
        self.client = MongoClient(mongo_endpoint)
        self.db = self.client['longtrainer_db']
        self.bots = self.db['bots']
        self.chats = self.db['chats']
        self.vision_chats = self.db['vision_chats']
        self.documents_collection = self.db['documents']

        # Encryption Setup
        self.encrypt_chats = encrypt_chats
        if encrypt_chats:
            self.encryption_key = encryption_key if encryption_key else Fernet.generate_key()
            self.fernet = Fernet(self.encryption_key)

    def initialize_bot_id(self):
        """
        Initializes a new bot with a unique identifier and initial data structure.

        This method generates a unique bot_id, sets up the initial structure for the bot data, and stores
        this data in Redis. Additionally, it inserts a record into the bots table in the database.

        Returns:
            str: The unique identifier (bot_id) for the newly initialized bot.

        The bot data initialized with this method includes empty structures for documents, chains, assistants,
        and other fields necessary for the bot's operation.
        """
        try:
            bot_id = 'bot-' + str(uuid.uuid4())
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
            self.bots.insert_one({"bot_id": bot_id, "faiss_path": self.bot_data[bot_id]['faiss_path']})
            return bot_id

        except Exception as e:
            print(f"Error Initializing Bot: {e}")

    def web_searching(self, query):
        """
        Performs a web search for the given query and returns the search results as text.

        This method utilizes an integrated search service to query the web and retrieve results. The
        results are returned in a text format, potentially including titles, snippets, and links.

        Args:
            query (str): The search query to be executed.

        Returns:
            str: The text representation of the search results, which may include various metadata
                 such as titles, snippets, and links.
        """
        try:
            text = self.search.run(query)
            return text
        except Exception as e:
            print(f"Error Web Searching: {e}")

    def get_websearch_links(self, text):
        """
        Extracts and returns web search result links from the given text.

        This method parses the provided text to extract web links. It expects the text to contain
        search results in a specific format, where each result may include a title, a snippet, and a
        link. Only the links are extracted and returned.

        Args:
            text (str): The text containing web search results, expected to follow a specific format
                        with segments delineated by brackets '[]' and key-value pairs for titles,
                        snippets, and links.

        Returns:
            list of str: A list of extracted web links from the search results. If no valid links are
                         found within the text, an empty list is returned.

        Note:
            The method assumes a specific formatting of the search result text, relying on regex
            patterns to identify and extract link information. This may need adjustments if the
            formatting of the search results changes.
        """
        try:
            segments = re.findall(r'\[([^]]+)\]', text)

            results = []
            for segment in segments:
                snippet_search = re.search(r'snippet: (.*?), title:', segment)
                title_search = re.search(r'title: (.*?), link:', segment)
                link_search = re.search(r'link: (.*)', segment)

                if snippet_search and title_search and link_search:
                    snippet = snippet_search.group(1)
                    title = title_search.group(1)
                    link = link_search.group(1)

                    results.append(link)
            return results

        except Exception as e:
            print(f"Error getting Web Links: {e}")

    def _default_prompt_template(self):
        """
        Returns the default prompt template for the assistant.
        """
        return """
        You will act as Intelligent assistant and your name is longtrainer and you will asnwer the any kind of query. YOu will act like conversation chatbot to interact with user. You will introduce your self as longtrainer.
        {context}
        Use the following pieces of information to answer the user's question. If the answer is unknown, admitting ignorance is preferred over fabricating a response. Dont need to add irrelevant text explanation in response.
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

                for doc in documents:
                    self.documents_collection.insert_one({
                        'bot_id': bot_id,
                        'document': serialize_document(doc),
                    })
                self.bot_data[bot_id]['documents'].extend(documents)
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

                    for doc in documents:
                        self.documents_collection.insert_one({
                            'bot_id': bot_id,
                            'document': serialize_document(doc),
                        })
                    self.bot_data[bot_id]['documents'].extend(documents)
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
                for doc in query_documents:
                    self.documents_collection.insert_one({
                        'bot_id': bot_id,
                        'document': serialize_document(doc),
                    })
                self.bot_data[bot_id]['documents'].extend(query_documents)

        except Exception as e:
            print(f"Error adding document from query: {e}")

    def pass_documents(self, documents, bot_id):
        """
       Add documents from any custom Loader.

        Args:
            documents (list): list of loaded documents.
        """
        try:
            if bot_id in self.bot_data:
                for doc in documents:
                    self.documents_collection.insert_one({
                        'bot_id': bot_id,
                        'document': serialize_document(doc),
                    })
                self.bot_data[bot_id]['documents'].extend(documents)

        except Exception as e:
            print(f"Error adding documents: {e}")

    def create_bot(self, bot_id):
        """
        Creates and returns a conversational AI assistant based on the loaded documents.

        Args:
            bot_id (str): The unique identifier for the bot.

        Returns:
            A function that will initialize Chatbot, or return None if an error occurs.
        """
        try:
            if bot_id in self.bot_data:
                all_splits = self.text_splitter.split_documents(self.bot_data[bot_id]['documents'])
                self.bot_data[bot_id]['retriever'] = DocRetriever(all_splits, self.embedding_model,
                                                                  existing_faiss_index=self.bot_data[bot_id][
                                                                      'retriever'].faiss_index if self.bot_data[bot_id][
                                                                      'retriever'] else None, num_k=self.k)
                self.bot_data[bot_id]['retriever'].save_index(file_path=self.bot_data[bot_id]['faiss_path'])
                self.bot_data[bot_id]['ensemble_retriever'] = self.bot_data[bot_id]['retriever'].retrieve_documents()

        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def load_bot(self, bot_id):
        """
        Attempts to load a bot by the given bot_id. If the bot exists, it loads its configuration and initializes
        or updates its components. If the bot does not exist, it initializes a new bot configuration with default settings.

        Args:
            bot_id (str): The unique identifier of the bot to load or initialize.

        Raises:
            ValueError: If bot_id is not provided.

        Returns:
            None: This method initializes or updates the bot's components in the system.
        """
        if not bot_id:
            raise ValueError("Bot ID must be provided.")

        try:
            bot_config = self.bots.find_one({"bot_id": bot_id})

            if not bot_config:
                # Initialize new bot configuration if not found
                print(f"No configuration found for {bot_id}. Initializing new bot...")
                bot_config = {
                    'bot_id': bot_id,
                    'faiss_path': f'faiss_index_{bot_id}'
                }
                self.bots.insert_one(bot_config)

            # Fetch documents from MongoDB for this bot
            mongo_documents = list(self.documents_collection.find({'bot_id': bot_id}))
            documents = [deserialize_document(doc['document']) for doc in mongo_documents]
            # Ensure all necessary keys in self.bot_data for bot initialization or loading
            self.bot_data[bot_id] = {
                'documents': documents,  # Update this line to use the fetched documents
                'chains': {},
                'assistants': {},
                'retriever': None,  # To be initialized below
                'ensemble_retriever': None,
                'conversational_chain': None,
                'faiss_path': bot_config['faiss_path'],
                'assistant': None
            }

            faiss_path = self.bot_data[bot_id]['faiss_path']
            if os.path.exists(faiss_path):
                # Load existing FAISS index if available
                faiss_index = FAISS.load_local(faiss_path, self.embedding_model, allow_dangerous_deserialization=True)
            else:
                # Initialize a new FAISS index if not found
                faiss_index = None

            all_splits = self.text_splitter.split_documents(self.bot_data[bot_id]['documents'])
            # Initialize the DocRetriever with the documents fetched from MongoDB
            self.bot_data[bot_id]['retriever'] = DocRetriever(all_splits,
                                                              self.embedding_model,
                                                              existing_faiss_index=faiss_index,
                                                              num_k=self.k)
            # After initializing DocRetriever, possibly use it to update or initialize ensemble_retriever and conversational_chain as needed
            self.bot_data[bot_id]['ensemble_retriever'] = self.bot_data[bot_id][
                'retriever'].retrieve_documents()  # Example
            print(f"Bot {bot_id} initialized or loaded successfully with documents from MongoDB.")
        except Exception as e:
            print(f"Error loading bot by ID: {e}")

    def new_chat(self, bot_id):
        """
        Creates and returns a new Chat based on the loaded documents.

        Args:
            bot_id (str): The unique identifier for the bot.

        Returns:
            A string representing the unique identifier for the new chat session, or None if an error occurs.
        """
        try:
            chat_id = 'chat-' + str(uuid.uuid4())
            bot = ChainBot(retriever=self.bot_data[bot_id]['ensemble_retriever'], llm=self.llm, prompt=self.prompt,
                           token_limit=self.max_token_limit)
            self.bot_data[bot_id]['conversational_chain'] = bot.get_chain()
            self.bot_data[bot_id]['chains'][chat_id] = self.bot_data[bot_id]['conversational_chain']
            return chat_id
        except Exception as e:
            print(f"Error creating New Chat: {e}")
            return None

    def new_vision_chat(self, bot_id):
        """
        Creates a new vision chat session for the given bot.

        Args:
            bot_id (str): The unique identifier for the bot.

        Returns:
            A string representing the unique identifier for the new vision chat session, or None if an error occurs.
        """
        try:
            vision_chat_id = 'vision-' + str(uuid.uuid4())
            self.bot_data[bot_id]['assistant'] = VisionMemory(self.max_token_limit,
                                                              self.bot_data[bot_id]['ensemble_retriever'],
                                                              prompt_template=self.prompt_template)
            self.bot_data[bot_id]['assistants'][vision_chat_id] = self.bot_data[bot_id]['assistant']

            return vision_chat_id
        except Exception as e:
            print(f"Error creating new Vision Chat: {e}")
            return None

    def update_chatbot(self, paths, bot_id, links=None, search_query=None):

        """
        Updates the chatbot with new documents from given paths, links, and a search query.

        Args:
            paths (list): List of file paths to load documents from.
            links (list): List of web links to load documents from.
            search_query (str): Wikipedia search query to load documents from.
            bot_id (str): The unique identifier for the bot.

        Raises:
            Exception: If the bot ID is not found.
        """
        try:
            initial_docs_len = len(self.bot_data[bot_id]['documents'])
            for path in paths:
                self.add_document_from_path(path, bot_id)
            if links:
                self.add_document_from_link(links, bot_id)
            if search_query:
                self.add_document_from_query(search_query, bot_id)

            # Calculate new documents added
            new_docs = self.bot_data[bot_id]['documents'][initial_docs_len:]

            if self.bot_data[bot_id]['retriever'] and self.bot_data[bot_id]['retriever'].faiss_index:
                print(len(new_docs))
                all_splits = self.text_splitter.split_documents(new_docs)

                # Use the new method to update the existing index
                self.bot_data[bot_id]['retriever'].update_index(all_splits)
                self.bot_data[bot_id]['retriever'].save_index(file_path=self.bot_data[bot_id]['faiss_path'])
                self.bot_data[bot_id]['ensemble_retriever'] = self.bot_data[bot_id]['retriever'].retrieve_documents()

            else:
                all_splits = self.text_splitter.split_documents(self.bot_data[bot_id]['documents'])

                # If no retriever or FAISS index, create a new one
                self.bot_data[bot_id]['retriever'] = DocRetriever(all_splits, self.embedding_model,
                                                                  existing_faiss_index=self.bot_data[bot_id][
                                                                      'retriever'].faiss_index if
                                                                  self.bot_data[bot_id]['retriever'] else None,
                                                                  num_k=self.k)
                self.bot_data[bot_id]['retriever'].save_index(file_path=self.bot_data[bot_id]['faiss_path'])
                self.bot_data[bot_id]['ensemble_retriever'] = self.bot_data[bot_id]['retriever'].retrieve_documents()

            self.create_bot(bot_id)
        except Exception as e:
            print(f"Error updating chatbot: {e}")

    def _encrypt_data(self, data):
        '''
        Receives Chats Data and encrypt them .

        Args:
            data : Queries and Responses.

        Returns:
            Encrypted Data.
        '''
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            print(f"Error Encrypting Documents: {e}")

    def _decrypt_data(self, data):
        '''
        Receives Encrypted Chats Data and decrypt them .

        Args:
            data : Encrypted Queries and Responses.

        Returns:
            Decrypted Data.
        '''
        try:
            return self.fernet.decrypt(data.encode()).decode()
        except Exception as e:
            print(f"Error Decrypting Documents: {e}")

    def get_response(self, query, bot_id, chat_id, web_search=False):
        """
        Retrieves a response from the conversational AI assistant for a given query, potentially
        incorporating web search results.

        This method fetches a response from the bot's conversational chain. If web_search is enabled, it
        performs a web search based on the query and includes the results in the context for generating
        the response. The method also handles chat data encryption if it's enabled.

        Args:
            query (str): The query string for the assistant.
            bot_id (str): The unique identifier for the bot.
            chat_id (str): The unique identifier for the chat session.
            web_search (bool, optional): Flag to enable or disable web search integration. Defaults to False.

        Returns:
            tuple: A tuple containing the assistant's response and the list of web sources used, if any.

        Raises:
            Exception: If the bot ID or chat ID is not found in the system.
        """
        try:
            if bot_id not in self.bot_data:
                raise Exception(f"Bot ID {bot_id} not found.")

            if chat_id not in self.bot_data[bot_id]['chains']:
                raise Exception(
                    f"Chat ID {chat_id} not found in bot {bot_id}. Available chats: {list(self.bot_data[bot_id]['chains'].keys())}")

            web_source = []
            webdata = None
            seen_sources = set()
            unique_sources = []
            if web_search:
                webdata = self.web_searching(query)
                web_source = self.get_websearch_links(webdata)

            chain = self.bot_data[bot_id]['chains'][chat_id]

            updated_query = f"{query}\nKindly consider the following text that's extracted from web search while answering the question. The following wensearch context will help you to provide upfated knowledge and kindly consider it must in answering the question.\n{webdata}" if webdata else query

            result = chain(updated_query)

            answer = result.get('answer')

            for doc in result['source_documents']:
                # Assuming 'metadata' is an attribute of the 'Document' object and 'source' is a key in that metadata dictionary
                source = doc.metadata.get('source') if hasattr(doc, 'metadata') and isinstance(doc.metadata,
                                                                                               dict) else None

                if source and source not in seen_sources:
                    unique_sources.append(source)
                    seen_sources.add(source)

            if self.encrypt_chats:
                encrypted_query = self._encrypt_data(query)
                encrypted_result = self._encrypt_data(answer)
                if len(web_source) > 0:
                    encrypted_web_source = [self._encrypt_data(source) for source in web_source]
                else:
                    encrypted_web_source = []
            else:
                encrypted_web_source = web_source

                # Insert the chat data along with web sources into MongoDB
            self.chats.insert_one({
                "bot_id": bot_id,
                "chat_id": chat_id,
                "question": encrypted_query if self.encrypt_chats else query,
                "answer": encrypted_result if self.encrypt_chats else answer,
                "web_sources": encrypted_web_source,  # Inserting web sources
                "trained": False  # Indicate this chat has not been trained yet
            })

            return answer, web_source
        except Exception as e:
            print(f"Error getting Bot Response: {e}")

    def get_vision_response(self, query, image_paths, bot_id, vision_chat_id, web_search=False):
        """
        Retrieves a response from the vision AI assistant for a given query and set of images,
        potentially incorporating web search results.

        This method processes a query along with provided image paths using the vision AI assistant. If
        web_search is enabled, it performs a web search based on the query and includes this information
        in the context for the response generation. Chat data encryption is handled if enabled.

        Args:
            query (str): The query string for the assistant.
            image_paths (list of str): A list of paths to the images for the vision chat.
            bot_id (str): The unique identifier for the bot.
            vision_chat_id (str): The unique identifier for the vision chat session.
            web_search (bool, optional): Flag to enable or disable web search integration. Defaults to False.

        Returns:
            tuple: A tuple containing the vision assistant's response and the list of web sources used, if any.

        Raises:
            Exception: If the bot ID or vision chat ID is not found in the system.
        """
        try:

            if bot_id not in self.bot_data or vision_chat_id not in self.bot_data[bot_id]['assistants']:
                raise Exception(f"Bot ID {bot_id} or Vision Chat ID {vision_chat_id} not found")
            web_source = []
            text = None
            if web_search:
                text = self.web_searching(query)
                web_source = self.get_websearch_links(text)

            assistant = self.bot_data[bot_id]['assistants'][vision_chat_id]
            prompt, doc_sources = assistant.get_answer(query, text)
            vision = VisionBot(prompt)
            vision.create_vision_bot(image_paths)
            vision_response = vision.get_response(query)
            assistant.save_chat_history(query, vision_response)

            if self.encrypt_chats:
                encrypted_query = self._encrypt_data(query)
                encrypted_vision_response = self._encrypt_data(vision_response)
                if len(web_source) > 0:
                    encrypted_web_source = [self._encrypt_data(source) for source in web_source]
                else:
                    encrypted_web_source = []
            else:
                encrypted_web_source = web_source

            # Insert the vision chat data along with web sources into MongoDB
            self.vision_chats.insert_one({
                "bot_id": bot_id,
                "vision_chat_id": vision_chat_id,
                "image_path": ','.join(image_paths),
                "question": encrypted_query if self.encrypt_chats else query,
                "response": encrypted_vision_response if self.encrypt_chats else vision_response,
                "web_sources": encrypted_web_source,  # Inserting web sources
                "trained": False  # Indicate this chat has not been trained yet

            })

            return vision_response, web_source
        except Exception as e:
            print(f"Error getting vision response: {e}")
            return None

    def delete_chatbot(self, bot_id):
        """
        Deletes a chatbot and its associated data from the system, including any stored CSV files.

        Args:
            bot_id (str): The unique identifier for the bot to be deleted.

        Raises:
            Exception: If the bot ID is not found.
        """
        if bot_id in self.bot_data:
            # Delete related documents from the collections
            self.chats.delete_many({"bot_id": bot_id})
            self.vision_chats.delete_many({"bot_id": bot_id})
            self.bots.delete_one({"bot_id": bot_id})

            # Additional in-memory cleanup
            bot = self.bot_data[bot_id]
            if bot['retriever']:
                bot['retriever'].delete_index(file_path=bot['faiss_path'])
            del self.bot_data[bot_id]

            # Delete the bot-specific data folder
            data_folder_path = f"./data-{bot_id}"
            try:
                shutil.rmtree(data_folder_path)
                print(f"Deleted data folder for bot ID {bot_id}.")
            except Exception as e:
                print(f"Error deleting data folder for bot ID {bot_id}: {e}")
        else:
            raise Exception(f"Bot ID {bot_id} not found")

    def list_chats(self, bot_id):
        """
        Lists unique chats for a specified bot, showing only the start of the first prompt for each chat.

        This method retrieves a list of unique chat sessions associated with the given bot_id from the MongoDB database.
        It displays the chat ID and the first few words of the first question part of each unique chat session. If encryption
        is enabled for the chats, it decrypts the questions before displaying them.

        Args:
            bot_id (str): The unique identifier of the bot for which the chat list is requested.

        Returns:
            None: This method prints the chat ID and a snippet of the first question of each unique chat session directly to the console.
                  It does not return any value.

        Note:
            The method assumes each chat session has a unique chat_id and fetches the first question to give an overview.
            The chat content is truncated for brevity in the listing.
        """
        try:
            unique_chats = self.chats.aggregate([
                {"$match": {"bot_id": bot_id}},
                {"$group": {"_id": "$chat_id", "first_question": {"$first": "$question"}}}
            ])

            for chat in unique_chats:
                chat_id = chat["_id"]
                question = chat["first_question"]
                if self.encrypt_chats:
                    question = self._decrypt_data(question)
                print(f"Chat ID: {chat_id}, Question: {' '.join(question.split()[:5])}...")
        except Exception as e:
            print(f"Error Getting Chats: {e}")

    def get_chat_by_id(self, chat_id):
        """
        Retrieves the full conversation details belonging to a specific chat using its unique chat ID.

        This method fetches all chat data related to the given chat_id from the MongoDB database. If the chats are encrypted,
        it decrypts the questions, answers, and web sources before compiling them into a single response object.
        If no chats are found with the given chat_id, it returns None.

        Args:
            chat_id (str): The unique identifier of the chat session to be retrieved.

        Returns:
            list: A list of dictionaries, each containing the details of a part of the chat session,
                  including 'question', 'answer', and 'web_sources', if found. Each element is decrypted if encryption is enabled.
            None: If no chats are found with the given chat_id.

        Note:
            If 'encrypt_chats' is set to True, the method decrypts the data before returning it. The decryption is applied to
            'question', 'answer', and each element in the 'web_sources' list for every part of the conversation.
        """
        try:
            chat_data = list(self.chats.find({"chat_id": chat_id}))

            if not chat_data:
                return None

            for chat in chat_data:
                if self.encrypt_chats:
                    chat["question"] = self._decrypt_data(chat["question"])
                    chat["answer"] = self._decrypt_data(chat["answer"])
                    if "web_sources" in chat and chat["web_sources"]:
                        chat["web_sources"] = [self._decrypt_data(source) for source in chat["web_sources"]]

            return chat_data
        except Exception as e:
            print(f"Error Getting Chat with ID: {chat_id} : {e}")

    def _export_chats_to_csv(self, dataframe, bot_id):
        """
        Helper method to export chats to a CSV file.
        """
        try:
            csv_folder = f"./data-{bot_id}"
            os.makedirs(csv_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            csv_path = os.path.join(csv_folder, f"{bot_id}_data_{timestamp}.csv")
            dataframe.to_csv(csv_path, index=False)
            return csv_path
        except Exception as e:
            print(f"Error exporting Chats to csv: {e}")

    def train_chats(self, bot_id):
        """
        Trains the AI model on new chats that have not been processed yet.

        This method identifies chats associated with the specified bot_id that haven't been processed for training,
        exports these chats to a CSV file, and then trains the model using this new data. To avoid redundancy, only
        new chats are processed and trained on.

        Args:
            bot_id (str): The unique identifier of the bot to train on new chats.

        Returns:
            dict: A dictionary containing a message and the path to the generated CSV file with new chats.
        """
        try:
            # Fetch new (unprocessed) chats
            new_chats_query = {"bot_id": bot_id, "trained": {"$ne": True}}
            new_chats = list(self.chats.find(new_chats_query))

            if not new_chats:
                return {"message": "No new chats found for training.", "csv_path": None}

            # Decrypt if necessary and prepare DataFrame
            if self.encrypt_chats:
                for chat in new_chats:
                    chat["question"] = self._decrypt_data(chat["question"])
                    chat["answer"] = self._decrypt_data(chat["answer"])
            df = pd.DataFrame(new_chats, columns=["question", "answer"])
            df.columns = ["Question", "Answer"]

            csv_path = self._export_chats_to_csv(df, bot_id)

            # Mark chats as processed
            chat_ids = [chat["_id"] for chat in new_chats]
            self.chats.update_many({"_id": {"$in": chat_ids}}, {"$set": {"trained": True}})

            # Train on the new data
            self.add_document_from_path(path=csv_path, bot_id=bot_id)
            self.create_bot(bot_id=bot_id)

            return {"message": "Trainer updated with new chat history.", "csv_path": csv_path}
        except Exception as e:
            print(f"Error training Bot on Chats: {e}")
