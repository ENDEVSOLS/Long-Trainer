from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationTokenBufferMemory

class ChainBot:
    def __init__(self, retriever, llm, prompt, token_limit):
        """
        Initialize the ChainBot with a retriever, language model (llm), prompt,
        and an optional maximum token limit.

        Args:
            retriever: The document retriever object.
            llm: Language learning model for generating responses.
            prompt (str): The initial prompt to start the conversation.
            max_token_limit (int, optional): Maximum token limit for the conversation buffer. Defaults to 200.
        """
        try:
            # Memory and chain setup with dynamic max token limit
            self.memory = ConversationTokenBufferMemory(
                llm=llm,
                max_token_limit=token_limit,
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )

            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type='stuff',  # Modify as needed
                combine_docs_chain_kwargs={"prompt": prompt},
                memory=self.memory,
                verbose=False,
            )
        except Exception as e:
            # Handle any exceptions that occur during initialization
            print(f"Error initializing ChainBot: {e}")

    def get_chain(self):
        """
        Retrieve the conversational retrieval chain.

        Returns:
            The ConversationalRetrievalChain instance.
        """
        return self.chain
