import json
from longtrainer.loaders import DocumentLoader, TextSplitter
from longtrainer.retrieval import DocRetriever
from longtrainer.bot import ChainBot
from longtrainer.vision_bot import VisionMemory, VisionBot
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pymongo import MongoClient
import uuid
import redis


class LongTrainer:

    def __init__(self, redis_endpoint='redis://localhost:6379', mongo_endpoint='mongodb://localhost:27017/', llm=None,
                 embedding_model=None,
                 prompt_template=None, max_token_limit=32000):
        """
        Initialize the LongTrainer with optional language learning model, embedding model,
        prompt template, maximum token limit, and MongoDB endpoint.

        Args:
            mongo_endpoint (str): MongoDB connection string.
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

        # Redis setup
        self.redis_client = redis.StrictRedis.from_url(redis_endpoint)

        # MongoDB setup
        self.client = MongoClient(mongo_endpoint)
        self.db = self.client['longtrainer_db']
        self.bots = self.db['bots']
        self.chats = self.db['chats']
        self.vision_chats = self.db['vision_chats']

    # Redis-based methods for state management
    def _get_bot_data(self, bot_id):
        bot_data_json = self.redis_client.get(bot_id)
        if bot_data_json:
            return json.loads(bot_data_json)
        return None

    def _set_bot_data(self, bot_id, bot_data):
        self.redis_client.set(bot_id, json.dumps(bot_data))

    def initialize_bot_id(self):
        bot_id = 'bot-' + str(uuid.uuid4())
        bot_data = {
            'documents': [],
            'chains': {},
            'assistants': {},
            'retriever': None,
            'ensemble_retriever': None,
            'conversational_chain': None,
            'faiss_path': f'faiss_index_{bot_id}',
            'assistant': None
        }

        # Set initial bot data in Redis
        self._set_bot_data(bot_id, bot_data)

        # Insert data into the bots table
        self.bots.insert_one({"bot_id": bot_id, "faiss_path": bot_data['faiss_path']})
        return bot_id

    def _default_prompt_template(self):
        """
        Returns the default prompt template for the assistant.
        """
        return """
        Act as an Intelligent Assistant:

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
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")
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

            bot_data['documents'].extend(documents)
            self._set_bot_data(bot_id, bot_data)
        except Exception as e:
            print(f"Error adding document from path: {e}")

    def add_document_from_link(self, links, bot_id):
        """
        Loads and adds documents from provided web links.

        Args:
            links (list): List of web links to load documents from.
        """
        try:
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")
            for link in links:
                if 'youtube.com' in link.lower() or 'youtu.be' in link.lower():
                    documents = self.document_loader.load_YouTubeVideo(link)
                else:
                    documents = self.document_loader.load_urls([link])

                bot_data['documents'].extend(documents)
                self._set_bot_data(bot_id, bot_data)

        except Exception as e:
            print(f"Error adding document from link: {e}")

    def add_document_from_query(self, search_query, bot_id):
        """
        Loads and adds documents from a Wikipedia search query.

        Args:
            search_query (str): Search query for Wikipedia.
        """
        try:
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")
            documents = self.document_loader.wikipedia_query(search_query)

            bot_data['documents'].extend(documents)
            self._set_bot_data(bot_id, bot_data)

        except Exception as e:
            print(f"Error adding document from query: {e}")

    def pass_documents(self, documents, bot_id):
        """
       Add documents from any custom Loader.

        Args:
            documents (list): list of loaded documents.
        """
        try:
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")

            bot_data['documents'].extend(documents)
            self._set_bot_data(bot_id, bot_data)
        except Exception as e:
            print(f"Error adding documents: {e}")

    def create_bot(self, bot_id):
        """
        Creates and returns a conversational AI assistant based on the loaded documents.

        Returns:
            A function that takes a query and returns the assistant's response.
        """
        try:
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")

            all_splits = self.text_splitter.split_documents(bot_data['documents'])
            bot_data['retriever'] = DocRetriever(all_splits, self.embedding_model,
                                                 existing_faiss_index=bot_data['retriever'].faiss_index if
                                                 self.bot_data[bot_id]['retriever'] else None)
            bot_data['retriever'].save_index(file_path=bot_data['faiss_path'])
            bot_data['ensemble_retriever'] = bot_data['retriever'].retrieve_documents()
            self._set_bot_data(bot_id, bot_data)
        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def new_chat(self, bot_id):
        try:
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")

            chat_id = 'chat-' + str(uuid.uuid4())
            bot = ChainBot(retriever=bot_data['ensemble_retriever'], llm=self.llm, prompt=self.prompt,
                           token_limit=self.max_token_limit)
            bot_data['conversational_chain'] = bot.get_chain()
            bot_data['chains'][chat_id] = bot_data['conversational_chain']
            self._set_bot_data(bot_id, bot_data)
            return chat_id
        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def new_vision_chat(self, bot_id):
        try:
            bot_data = self._get_bot_data(bot_id)
            if bot_data is None:
                raise Exception(f"Bot ID {bot_id} not found")

            vision_chat_id = 'vision-' + str(uuid.uuid4())
            bot_data['assistant'] = VisionMemory(self.max_token_limit, bot_data['ensemble_retriever'])
            bot_data['assistants'][vision_chat_id] = bot_data['assistant']
            self._set_bot_data(bot_id, bot_data)

            return vision_chat_id
        except Exception as e:
            print(f"Error creating bot: {e}")
            return None

    def update_chatbot(self, paths, links, search_query, bot_id):
        """
        Updates the chatbot with new documents.
        """
        bot_data = self._get_bot_data(bot_id)
        if bot_data is None:
            raise Exception(f"Bot ID {bot_id} not found")

        initial_docs_len = len(bot_data['documents'])
        for path in paths:
            self.add_document_from_path(path, bot_id)

        self.add_document_from_link(links, bot_id)
        if search_query:
            self.add_document_from_query(search_query, bot_id)

        # Calculate new documents added
        new_docs = bot_data['documents'][initial_docs_len:]

        if bot_data['retriever'] and bot_data['retriever'].faiss_index:
            print(len(new_docs))
            # Use the new method to update the existing index
            bot_data['retriever'].update_index(new_docs)
            bot_data['retriever'].save_index(file_path=bot_data['faiss_path'])

        else:
            # If no retriever or FAISS index, create a new one
            self.retriever = DocRetriever(self.documents, self.embedding_model,
                                          existing_faiss_index=bot_data['retriever'].faiss_index if
                                          bot_data['retriever'] else None)

        self.create_bot(bot_id)
        self._set_bot_data(bot_id, bot_data)

    def _get_response(self, query, chat_id, bot_id):
        """
        Retrieves a response from the conversational AI assistant for a given query.

        Args:
            query (str): Query string for the assistant.

        Returns:
            The assistant's response to the query.
        """
        bot_data = self._get_bot_data(bot_id)
        if bot_data is None or chat_id not in bot_data['chains']:
            raise Exception(f"Bot ID {bot_id} or Chat ID {chat_id} not found")

        chain = bot_data['chains'][chat_id]
        result = chain(query)
        result = result.get('answer')

        # Insert data into the chats collection
        self.chats.insert_one({
            "bot_id": bot_id,
            "chat_id": chat_id,
            "question": query,
            "answer": result
        })

        return result

    def _get_vision_response(self, query, image_paths, vision_chat_id, bot_id):

        bot_data = self._get_bot_data(bot_id)
        if bot_data is None or vision_chat_id not in bot_data['assistants']:
            raise Exception(f"Bot ID {bot_id} or Vision Chat ID {vision_chat_id} not found")

        assistant = bot_data['assistants'][vision_chat_id]
        prompt = assistant.get_answer(query)
        vision = VisionBot(prompt)
        vision.create_vision_bot(image_paths)
        vision_response = vision.get_response(query)
        assistant.save_chat_history(query, vision_response)

        # Insert data into the vision_chats collection
        self.vision_chats.insert_one({
            "bot_id": bot_id,
            "vision_chat_id": vision_chat_id,
            "image_path": ','.join(image_paths),
            "question": query,
            "response": vision_response
        })

        return vision_response

    def delete_chatbot(self, bot_id):
        """
        Deletes a chatbot and its associated data.
        """
        bot_data = self._get_bot_data(bot_id)
        if bot_data is not None:
            # Delete related documents from the collections
            self.chats.delete_many({"bot_id": bot_id})
            self.vision_chats.delete_many({"bot_id": bot_id})
            self.bots.delete_one({"bot_id": bot_id})

            # Delete Redis Data
            self.redis_client.delete(bot_id)

            # Additional in-memory cleanup
            if bot_data['retriever']:
                bot_data['retriever'].delete_index(file_path=bot_data['faiss_path'])
            del bot_data
        else:
            raise Exception(f"Bot ID {bot_id} not found")
