"""Chat session management for LongTrainer.

Handles chat creation, response generation (sync, streaming, async),
vision chat, and web search augmentation.
"""

from __future__ import annotations

import re
import uuid
from typing import AsyncIterator, Iterator, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from longtrainer.bot import RAGBot, AgentBot
from longtrainer.storage import MongoStorage
from longtrainer.tools import ToolRegistry
from longtrainer.vision_bot import VisionBot, VisionMemory


def build_chat_prompt(system_template: str) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with chat history support.

    Args:
        system_template: System message template string.

    Returns:
        A ChatPromptTemplate instance.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


class ChatManager:
    """Manages chat sessions, response generation, and web search.

    Args:
        storage: MongoStorage instance for persisting chats.
        llm: Default language model.
        max_token_limit: Token buffer limit for conversation memory.
    """

    def __init__(
        self,
        storage: MongoStorage,
        llm: BaseChatModel,
        max_token_limit: int = 32000,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.max_token_limit = max_token_limit

    # ─── Chat Session Creation ────────────────────────────────────────────────

    def new_chat(
        self,
        bot_data: dict,
        bot_id: str,
        prompt_template: str,
        global_tools: ToolRegistry,
    ) -> str:
        """Create a new chat session.

        Args:
            bot_data: The bot's runtime data dict.
            bot_id: The bot's unique identifier.
            prompt_template: Default prompt template.
            global_tools: Global tool registry.

        Returns:
            The generated chat_id string.
        """
        try:
            chat_id = "chat-" + str(uuid.uuid4())

            if bot_data.get("agent_mode"):
                tools = global_tools.get_tools()
                tools.extend(bot_data["tools"].get_tools())
                agent_bot = AgentBot(
                    llm=self.llm,
                    tools=tools,
                    system_prompt=bot_data.get("prompt_template", prompt_template),
                    token_limit=self.max_token_limit,
                )
                bot_data["chains"][chat_id] = agent_bot
            else:
                rag_bot = RAGBot(
                    retriever=bot_data["ensemble_retriever"],
                    llm=self.llm,
                    prompt=bot_data["prompt"],
                    token_limit=self.max_token_limit,
                )
                bot_data["chains"][chat_id] = rag_bot

            return chat_id
        except Exception as e:
            print(f"[ERROR] Error creating new chat: {e}")
            return ""

    def new_vision_chat(
        self,
        bot_data: dict,
        prompt_template: str,
    ) -> str:
        """Create a new vision chat session.

        Args:
            bot_data: The bot's runtime data dict.
            prompt_template: Default prompt template.

        Returns:
            The generated vision_chat_id string.
        """
        try:
            vision_chat_id = "vision-" + str(uuid.uuid4())

            vision_mem = VisionMemory(
                token_limit=self.max_token_limit,
                llm=self.llm,
                ensemble_retriever=bot_data["ensemble_retriever"],
                prompt_template=bot_data.get("prompt_template", prompt_template),
            )
            bot_data["assistants"][vision_chat_id] = vision_mem
            return vision_chat_id
        except Exception as e:
            print(f"[ERROR] Error creating vision chat: {e}")
            return ""

    # ─── Responses ────────────────────────────────────────────────────────────

    def get_response(
        self,
        query: str,
        bot_id: str,
        chat_id: str,
        bot_data: dict,
        stream: bool = False,
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> Union[tuple[str, list[str]], Iterator[str]]:
        """Get a response from the chatbot.

        Args:
            query: The user's question.
            bot_id: The bot's unique identifier.
            chat_id: The chat session's unique identifier.
            bot_data: The bot's runtime data dict.
            stream: If True, returns an iterator yielding response chunks.
            uploaded_files: Optional list of uploaded file metadata.
            web_search: Enable web search augmentation (for RAG mode).

        Returns:
            If stream=False: (answer_string, web_sources_list)
            If stream=True: Iterator yielding response token strings
        """
        try:
            if chat_id not in bot_data["chains"]:
                raise ValueError(f"Chat ID {chat_id} not found in bot {bot_id}.")

            bot_instance = bot_data["chains"][chat_id]
            web_source: list[str] = []
            final_query = query

            if web_search and not bot_data.get("agent_mode"):
                webdata = self._web_search(query)
                web_source = self._extract_web_links(webdata)
                final_query = f"{query}\n\nAdditional web context:\n{webdata}"

            if uploaded_files:
                file_details = "\n".join(
                    f"File: {f['name']} (Type: {f['type']})\n"
                    f"URL: {f.get('url', 'N/A')}\n"
                    f"Extracted Text: {f.get('extracted_text', 'N/A')}"
                    for f in uploaded_files
                )
                final_query = f"Uploaded Files:\n{file_details}\n\nQuestion:\n{final_query}"

            if stream:
                return self._stream_response(
                    final_query, bot_id, chat_id, bot_instance, query, web_source
                )

            answer = bot_instance.invoke(final_query)

            self.storage.store_chat(
                bot_id=bot_id,
                chat_id=chat_id,
                query=query,
                answer=answer,
                web_source=web_source,
                uploaded_files=uploaded_files,
            )

            return answer, web_source
        except Exception as e:
            print(f"[ERROR] Error getting response: {e}")
            return "", []

    def _stream_response(
        self,
        final_query: str,
        bot_id: str,
        chat_id: str,
        bot_instance: Union[RAGBot, AgentBot],
        original_query: str,
        web_source: list[str],
    ) -> Iterator[str]:
        """Internal streaming response generator."""
        full_response = ""
        for chunk in bot_instance.stream(final_query):
            full_response += chunk
            yield chunk

        self.storage.store_chat(
            bot_id=bot_id,
            chat_id=chat_id,
            query=original_query,
            answer=full_response,
            web_source=web_source,
        )

    async def aget_response(
        self,
        query: str,
        bot_id: str,
        chat_id: str,
        bot_data: dict,
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> AsyncIterator[str]:
        """Async streaming response.

        Args:
            query: The user's question.
            bot_id: The bot's unique identifier.
            chat_id: The chat session's unique identifier.
            bot_data: The bot's runtime data dict.
            uploaded_files: Optional uploaded file metadata.
            web_search: Enable web search augmentation.

        Yields:
            Response token strings.
        """
        try:
            if chat_id not in bot_data["chains"]:
                raise ValueError(f"Chat ID {chat_id} not found.")

            bot_instance = bot_data["chains"][chat_id]
            final_query = query

            if web_search and not bot_data.get("agent_mode"):
                webdata = self._web_search(query)
                final_query = f"{query}\n\nAdditional web context:\n{webdata}"

            if uploaded_files:
                file_details = "\n".join(
                    f"File: {f['name']} (Type: {f['type']})\n"
                    f"Extracted Text: {f.get('extracted_text', 'N/A')}"
                    for f in uploaded_files
                )
                final_query = f"Uploaded Files:\n{file_details}\n\nQuestion:\n{final_query}"

            full_response = ""
            async for chunk in bot_instance.astream(final_query):
                full_response += chunk
                yield chunk

            self.storage.store_chat(
                bot_id=bot_id,
                chat_id=chat_id,
                query=query,
                answer=full_response,
            )
        except Exception as e:
            print(f"[ERROR] Error in async response: {e}")

    def get_vision_response(
        self,
        query: str,
        image_paths: list[str],
        bot_id: str,
        vision_chat_id: str,
        bot_data: dict,
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> tuple[str, list[str]]:
        """Get a response from the vision AI assistant.

        Args:
            query: Text query for the vision model.
            image_paths: List of image file paths.
            bot_id: The bot's unique identifier.
            vision_chat_id: The vision chat session ID.
            bot_data: The bot's runtime data dict.
            uploaded_files: Optional uploaded file metadata.
            web_search: Enable web search augmentation.

        Returns:
            Tuple of (response_string, web_sources_list).
        """
        try:
            if vision_chat_id not in bot_data["assistants"]:
                raise ValueError(f"Vision chat ID {vision_chat_id} not found.")

            web_source: list[str] = []
            web_text = None
            if web_search:
                web_text = self._web_search(query)
                web_source = self._extract_web_links(web_text)

            assistant = bot_data["assistants"][vision_chat_id]

            final_query = query
            if uploaded_files:
                file_details = "\n".join(
                    f"File: {f['name']} (Type: {f['type']})\n"
                    f"Extracted Text: {f.get('extracted_text', 'N/A')}"
                    for f in uploaded_files
                )
                final_query = f"Uploaded Files:\n{file_details}\n\nQuestion:\n{query}"

            prompt, doc_sources = assistant.get_answer(final_query, web_text)
            vision = VisionBot(prompt_template=prompt, llm=self.llm)
            vision.create_vision_bot(image_paths)
            vision_response = vision.get_response(query)
            assistant.save_chat_history(query, vision_response)

            self.storage.store_vision_chat(
                bot_id=bot_id,
                vision_chat_id=vision_chat_id,
                image_paths=image_paths,
                query=query,
                response=vision_response,
                web_source=web_source,
                uploaded_files=uploaded_files,
            )

            return vision_response, web_source
        except Exception as e:
            print(f"[ERROR] Error getting vision response: {e}")
            return "", []

    # ─── Web Search Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _web_search(query: str) -> str:
        """Perform a DuckDuckGo web search."""
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = ddgs.text(query, max_results=5)
            if not results:
                return ""
            return "\n".join(
                f"[snippet: {r.get('body', '')}, title: {r.get('title', '')}, link: {r.get('href', '')}]"
                for r in results
            )
        except Exception as e:
            print(f"[ERROR] Web search error: {e}")
            return ""

    @staticmethod
    def _extract_web_links(text: str) -> list[str]:
        """Extract links from web search results text."""
        try:
            segments = re.findall(r"\[([^\]]+)\]", text)
            links = []
            for segment in segments:
                link_match = re.search(r"link: (.*)", segment)
                if link_match:
                    links.append(link_match.group(1).strip())
            return links
        except Exception as e:
            print(f"[ERROR] Error extracting web links: {e}")
            return []
