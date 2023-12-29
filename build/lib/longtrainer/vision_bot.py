import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import base64
class VisionMemory:
    def __init__(self, token_limit, ensemble_retriever=None):
        model_name='gpt-4-1106-preview'
        self.llm = ChatOpenAI(model_name=model_name)
        self.memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=token_limit,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        self.chat_history = []
        self.prompt_template = '''
        Act as Intelligent assistant
        {context}
        Your task is to answer the query with accurate answer using the chat history context.
        If the answer is unknown, admitting ignorance is preferred over fabricating a response. Dont need to add irrelevant text explanation in response.

        Chat History: {chat_history}

        Question: {question}

        Answer
        '''
        self.ensemble_retriever = ensemble_retriever

    def save_chat_history(self, query, answer):
        self.chat_history.append([query, answer])
        self.memory.save_context({"input": query}, {"answer": answer})

    def generate_prompt(self, query, additional_context):
        memory_history = self.memory.load_memory_variables({})
        return self.prompt_template.format(context=f"you will answer the query from provided context: {additional_context}", chat_history=memory_history, question=query)

    def get_answer(self, query):
        docs = self.ensemble_retriever.get_relevant_documents(query)
        prompt = self.generate_prompt(query, docs)
        return prompt

class VisionBot:
    def __init__(self, prompt_template, max_tokens=1024):
        model_name = "gpt-4-vision-preview"
        self.vision_chain = ChatOpenAI(model=model_name, max_tokens=max_tokens)
        self.prompt_template = prompt_template  # Save prompt template to instance variable
        self.human_message_content = []  # Initialize as an empty list

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_vision_bot(self, image_files):
        for file in image_files:
            encoded_image = self.encode_image(file)  # Use the encode_image function
            image_snippet = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}  # Corrected key to "url"
            }
            self.human_message_content.append(image_snippet)

    def get_response(self, query):

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