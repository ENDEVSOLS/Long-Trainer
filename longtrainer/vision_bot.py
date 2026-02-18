"""Vision bot modules for LongTrainer V2.

Provides GPT-4 Vision–style chat with image understanding.
"""

from __future__ import annotations

import base64
from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever


class VisionMemory:
    """Manages memory and context retrieval for vision chat sessions.

    Args:
        token_limit: Maximum token limit for conversation buffer.
        llm: Language model instance.
        ensemble_retriever: Retriever for document context lookup.
        prompt_template: System prompt template string.
    """

    def __init__(
        self,
        token_limit: int,
        llm: BaseChatModel,
        ensemble_retriever: Optional[BaseRetriever] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        try:
            self.llm = llm
            self.chat_history_store = InMemoryChatMessageHistory()
            self.chat_history: list = []
            self.prompt_template = prompt_template or (
                "You are an intelligent assistant named LongTrainer.\n"
                "{context}\n"
                "Use the following information to respond accurately. "
                "If the answer is unknown, admit it rather than fabricating a response.\n"
                "Chat History: {chat_history}\n"
                "Question: {question}\n"
                "Answer:"
            )
            self.ensemble_retriever = ensemble_retriever
        except Exception as e:
            print(f"[ERROR] VisionMemory initialization error: {e}")

    def save_chat_history(self, query: str, answer: str) -> None:
        """Save a query-answer pair to chat history.

        Args:
            query: The user's question.
            answer: The AI's response.
        """
        try:
            self.chat_history.append([query, answer])
            self.chat_history_store.add_message(HumanMessage(content=query))
            self.chat_history_store.add_message(AIMessage(content=answer))
        except Exception as e:
            print(f"[ERROR] Error saving chat history: {e}")

    def save_context(self, query: str, answer: str) -> None:
        """Alias for save_chat_history for backward compatibility."""
        self.save_chat_history(query, answer)

    def generate_prompt(self, query: str, additional_context: str) -> str:
        """Generate a formatted prompt for the vision model.

        Args:
            query: The user's question.
            additional_context: Context from retrieved documents.

        Returns:
            Formatted prompt string.
        """
        try:
            messages = self.chat_history_store.messages
            return self.prompt_template.format(
                context=f"Answer the query from provided context: {additional_context}",
                chat_history=str(messages),
                question=query,
            )
        except Exception as e:
            print(f"[ERROR] Error generating prompt: {e}")
            return ""

    def get_answer(self, query: str, webdata: Optional[str] = None) -> tuple[str, list[str]]:
        """Retrieve context documents and generate a prompt.

        Args:
            query: The user's question.
            webdata: Optional web search context.

        Returns:
            Tuple of (formatted_prompt, list_of_source_paths).
        """
        try:
            unique_sources: set[str] = set()
            docs = self.ensemble_retriever.invoke(query) if self.ensemble_retriever else []
            for doc in docs:
                source = (
                    doc.metadata.get("source")
                    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict)
                    else None
                )
                if source:
                    unique_sources.add(source)

            updated_query = (
                f"{query}\nAdditional web search context:\n{webdata}" if webdata else query
            )
            prompt = self.generate_prompt(updated_query, docs)
            return prompt, list(unique_sources)
        except Exception as e:
            print(f"[ERROR] Error getting answer: {e}")
            return "", []

    @property
    def memory(self):
        """Backward-compatible memory accessor."""
        return self


class VisionBot:
    """Vision-based conversational AI that processes images with text queries.

    Encodes images to base64 and sends them alongside text prompts to
    a multimodal LLM (e.g. GPT-4 Vision).

    Args:
        llm: Language model instance (must support vision/multimodal).
        prompt_template: System prompt for the vision conversation.
        max_tokens: Maximum tokens for response generation.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: str,
        max_tokens: int = 1024,
    ) -> None:
        try:
            self.vision_chain = llm
            self.prompt_template = prompt_template
            self.human_message_content: list[dict] = []
        except Exception as e:
            print(f"[ERROR] VisionBot initialization error: {e}")

    def encode_image(self, image_path: str) -> str:
        """Encode an image file to a base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64-encoded string of the image.
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[ERROR] Image encoding error: {e}")
            return ""

    def create_vision_bot(self, image_files: list[str]) -> None:
        """Prepare the bot with encoded images.

        Args:
            image_files: List of image file paths to encode.
        """
        try:
            for file_path in image_files:
                encoded = self.encode_image(file_path)
                if encoded:
                    self.human_message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    })
        except Exception as e:
            print(f"[ERROR] Error creating vision bot: {e}")

    def get_response(self, query: str) -> str:
        """Generate a response using images and text query.

        Args:
            query: Text query for the vision model.

        Returns:
            The model's text response.
        """
        try:
            self.human_message_content.insert(0, {"type": "text", "text": query})
            msg = self.vision_chain.invoke([
                AIMessage(content=self.prompt_template),
                HumanMessage(content=self.human_message_content),
            ])
            return msg.content
        except Exception as e:
            print(f"[ERROR] Error getting vision response: {e}")
            return ""
