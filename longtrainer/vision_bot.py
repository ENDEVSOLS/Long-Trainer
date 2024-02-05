import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import base64


class VisionMemory:
    """
    A class to manage the memory and conversation history for a vision-enabled conversational AI model.

    This class encapsulates the logic for handling the chat history, generating prompts, and retrieving
    answers from a language learning model (LLM), with an optional integration of an ensemble retriever
    for document retrieval. It also formats queries with additional context from web search results.

    Attributes:
        token_limit (int): The maximum number of tokens allowed in the conversational buffer.
        ensemble_retriever (object, optional): An instance of an ensemble retriever for document retrieval.
        prompt_template (str, optional): A template for generating prompts for the AI model.
        llm (object): An instance of the language learning model.
        memory (ConversationTokenBufferMemory): An instance for managing conversation token buffer memory.
        chat_history (list): A list to keep track of the chat history.

    Methods:
        save_chat_history(query, answer): Saves the given query and answer pair into the chat history.
        generate_prompt(query, additional_context): Generates a prompt for the AI model based on the given query and additional context.
        get_answer(query, webdata): Retrieves the answer from the AI model for a given query, considering additional web search data.
    """

    def __init__(self, token_limit, ensemble_retriever=None, prompt_template=None):
        """
        Initializes the VisionMemory object with a token limit for the conversation buffer, an optional
        ensemble retriever for document retrieval, and an optional prompt template.

        Args:
            token_limit (int): The maximum number of tokens allowed in the conversational buffer.
            ensemble_retriever (object, optional): An instance of an ensemble retriever for document retrieval. Defaults to None.
            prompt_template (str, optional): A template for generating prompts for the AI model. Defaults to a predefined template.
        """
        model_name = 'gpt-4-1106-preview'
        self.llm = ChatOpenAI(model_name=model_name)
        self.memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=token_limit,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        self.chat_history = []
        self.prompt_template = prompt_template if prompt_template else '''
        You will act as Intelligent assistant and your name is longtrainer and you will asnwer the any kind of query. YOu will act like conversation chatbot to interact with user. You will introduce your self as longtrainer.
        {context}
        Your task is to answer the query with accurate answer using the chat history context.
        If the answer is unknown, admitting ignorance is preferred over fabricating a response. Dont need to add irrelevant text explanation in response.

        Chat History: {chat_history}

        Question: {question}

        Answer
        '''
        self.ensemble_retriever = ensemble_retriever

    def save_chat_history(self, query, answer):
        """
        Saves a query and its corresponding answer to the chat history.

        This method appends the query-answer pair to the chat history and updates the conversation token buffer memory.

        Args:
            query (str): The user's query or question.
            answer (str): The AI model's answer or response to the query.
        """
        self.chat_history.append([query, answer])
        self.memory.save_context({"input": query}, {"answer": answer})

    def generate_prompt(self, query, additional_context):
        """
        Generates a prompt for the AI model based on the given query and additional context.

        This method formats the prompt according to the predefined template, incorporating the provided query and
        additional context retrieved from relevant documents.

        Args:
            query (str): The user's query or question.
            additional_context (str): Additional context to be included in the prompt, typically extracted from relevant documents.

        Returns:
            str: A formatted prompt ready to be used by the AI model.
        """
        memory_history = self.memory.load_memory_variables({})
        return self.prompt_template.format(
            context=f"you will answer the query from provided context: {additional_context}",
            chat_history=memory_history, question=query)

    def get_answer(self, query, webdata):
        """
        Retrieves an answer from the AI model for a given query, considering additional web search data.

        This method uses the ensemble retriever to fetch relevant documents, formats the query with additional web search context,
        generates a prompt, and then retrieves the AI model's response.

        Args:
            query (str): The user's query or question.
            webdata (str): Additional context from web search to be considered in the response.

        Returns:
            tuple: A tuple containing the generated prompt and a list of unique sources from the retrieved documents.
        """
        unique_sources = set()
        docs = self.ensemble_retriever.get_relevant_documents(query)
        for doc in docs:
            # Accessing 'metadata' as an attribute of the 'Document' object
            source = doc.metadata.get('source') if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else None
            if source:
                unique_sources.add(source)

        updated_query = f"{query}\nKindly consider the following text that's extracted from web search while answering the question. The following wensearch context will help you to provide upfated knowledge and kindly consider it must in answering the question.\n{webdata}" if webdata else query
        prompt = self.generate_prompt(updated_query, docs)
        return prompt, list(unique_sources)


class VisionBot:
    """
    A class for a vision-based conversational AI model, capable of processing and responding to queries
    with visual content.

    This class provides functionality to encode images, manage conversation history, and generate
    responses using a vision-enabled language learning model.

    Attributes:
        prompt_template (str): A template for generating prompts for the AI model.
        max_tokens (int): The maximum number of tokens that the AI model can generate in a single response.
        vision_chain (ChatOpenAI): An instance of the ChatOpenAI model configured for vision-related tasks.
        human_message_content (list): A list to store the content of messages from the human user, including encoded images.

    Methods:
        encode_image(image_path): Encodes the image at the specified path into a base64 string.
        create_vision_bot(image_files): Prepares the vision bot with a set of encoded images.
        get_response(query): Generates a response from the AI model for a given query, incorporating any images that have been provided.
    """

    def __init__(self, prompt_template, max_tokens=1024):
        """
        Initializes the VisionBot with a specific prompt template and token limit.

        This constructor sets up a vision-based conversational AI model, using the ChatOpenAI model configured for vision tasks.

        Args:
            prompt_template (str): The template used for generating prompts for the AI model.
            max_tokens (int): The maximum number of tokens the AI model can generate for each response.

        The initialized instance contains a vision_chain for AI interactions, a prompt template for generating queries, and an empty list for storing human message content.
        """
        model_name = "gpt-4-vision-preview"
        self.vision_chain = ChatOpenAI(model=model_name, max_tokens=max_tokens)
        self.prompt_template = prompt_template  # Save prompt template to instance variable
        self.human_message_content = []  # Initialize as an empty list

    def encode_image(self, image_path):
        """
        Encodes an image at a given file path into a base64 string.

        This method is used to prepare images for processing by the vision AI model. The image file is read and converted into a base64 encoded string.

        Args:
            image_path (str): The file path of the image to be encoded.

        Returns:
            str: The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_vision_bot(self, image_files):
        """
        Prepares the vision bot with a set of encoded images.

        This method takes a list of image file paths, encodes each image using the encode_image method, and stores the encoded images in a format suitable for processing by the vision AI model.

        Args:
            image_files (list of str): A list of paths to the image files to be encoded and processed.
        """
        for file in image_files:
            encoded_image = self.encode_image(file)  # Use the encode_image function
            image_snippet = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}  # Corrected key to "url"
            }
            self.human_message_content.append(image_snippet)

    def get_response(self, query):
        """
        Generates a response from the AI model for a given text query, incorporating any previously provided images.

        This method constructs a conversational context that includes the query and any encoded images, and then invokes the AI model to generate a response based on this context.

        Args:
            query (str): The text query for which a response is sought from the AI model.

        Returns:
            str: The AI model's response to the query, considering both the text and any visual content.
        """
        # Create a message with the current query
        self.human_message_content.insert(0, {"type": "text", "text": query})
        # Uncomment and modify the invoke call
        msg = self.vision_chain.invoke(
            [AIMessage(
                content=self.prompt_template  # Use self.prompt_template
            ),
                HumanMessage(content=self.human_message_content)
            ]
        )
        return msg.content
