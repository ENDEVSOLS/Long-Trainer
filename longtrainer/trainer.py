"""LongTrainer V2 — Production-Ready RAG Framework.

Main orchestrator class providing:
- Multi-bot management with isolated chat sessions
- Dual mode: RAG (LCEL) or Agent (LangGraph with tools)
- Streaming responses
- Custom tool calling
- MongoDB persistence with optional chat encryption
- FAISS vector store management
- Vision chat support
"""

from __future__ import annotations

import gc
import os
import re
import uuid
import shutil
from datetime import datetime, timezone
from typing import AsyncIterator, Iterator, Optional, Union

import pandas as pd
from cryptography.fernet import Fernet
from pymongo import MongoClient

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from longtrainer.bot import RAGBot, AgentBot
from longtrainer.loaders import DocumentLoader, TextSplitter
from longtrainer.retrieval import DocumentRetriever
from longtrainer.tools import ToolRegistry, get_builtin_tools
from longtrainer.utils import deserialize_document, serialize_document
from longtrainer.vision_bot import VisionBot, VisionMemory


_DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent assistant named LongTrainer. "
    "Your purpose is to answer all kind of queries and interact with the user "
    "in a helpful and conversational manner.\n"
    "{context}\n"
    "Use the following information to respond to the user's question. "
    "If the answer is unknown, admit it rather than fabricating a response. "
    "Avoid unnecessary details or irrelevant explanations.\n"
    "Responses should be direct, professional, and focused solely on the user's query."
)


def _build_chat_prompt(system_template: str) -> ChatPromptTemplate:
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


class LongTrainer:
    """Production-Ready RAG Framework with multi-bot management.

    Args:
        mongo_endpoint: MongoDB connection string.
        llm: Language model (defaults to ChatOpenAI gpt-4o-2024-08-06).
        embedding_model: Embedding model (defaults to OpenAIEmbeddings).
        prompt_template: System prompt template string.
        max_token_limit: Token buffer limit for conversation memory.
        num_k: Number of documents to retrieve per query.
        chunk_size: Text splitter chunk size.
        chunk_overlap: Text splitter overlap size.
        ensemble: Enable ensemble retriever (FAISS + MultiQuery).
        encrypt_chats: Enable Fernet encryption for stored chats.
        encryption_key: Fernet key (auto-generated if not provided).
    """

    def __init__(
        self,
        mongo_endpoint: str = "mongodb://localhost:27017/",
        llm: Optional[BaseChatModel] = None,
        embedding_model: Optional[Embeddings] = None,
        prompt_template: Optional[str] = None,
        max_token_limit: int = 32000,
        num_k: int = 3,
        chunk_size: int = 2048,
        chunk_overlap: int = 200,
        ensemble: bool = False,
        encrypt_chats: bool = False,
        encryption_key: Optional[bytes] = None,
    ) -> None:
        self.llm = llm or ChatOpenAI(model_name="gpt-4o-2024-08-06")
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.prompt_template = prompt_template or _DEFAULT_SYSTEM_PROMPT
        self.prompt = _build_chat_prompt(self.prompt_template)
        self.k = num_k
        self.max_token_limit = max_token_limit
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.bot_data: dict = {}
        self.ensemble = ensemble

        self._global_tools = ToolRegistry()

        self.client = MongoClient(mongo_endpoint)
        self.db = self.client["longtrainer_db"]
        self.bots = self.db["bots"]
        self.chats = self.db["chats"]
        self.vision_chats = self.db["vision_chats"]
        self.documents_collection = self.db["documents"]

        self.encrypt_chats = encrypt_chats
        if encrypt_chats:
            self.encryption_key = encryption_key or Fernet.generate_key()
            self.fernet = Fernet(self.encryption_key)

    # ─── Bot Lifecycle ────────────────────────────────────────────────────────

    def initialize_bot_id(self) -> str:
        """Create a new bot with a unique identifier.

        Returns:
            The generated bot_id string.
        """
        try:
            bot_id = "bot-" + str(uuid.uuid4())
            self.bot_data[bot_id] = {
                "chains": {},
                "assistants": {},
                "retriever": None,
                "ensemble_retriever": None,
                "faiss_path": f"faiss_index_{bot_id}",
                "agent_mode": False,
                "tools": ToolRegistry(),
            }
            self.bots.insert_one({
                "bot_id": bot_id,
                "faiss_path": self.bot_data[bot_id]["faiss_path"],
            })
            return bot_id
        except Exception as e:
            print(f"[ERROR] Error initializing bot: {e}")
            return ""

    def create_bot(
        self,
        bot_id: str,
        prompt_template: Optional[str] = None,
        agent_mode: bool = False,
        tools: Optional[list[BaseTool]] = None,
        llm: Optional[BaseChatModel] = None,
        embedding_model: Optional[Embeddings] = None,
        num_k: Optional[int] = None,
    ) -> None:
        """Create and initialize a bot from loaded documents.

        Args:
            bot_id: The bot's unique identifier.
            prompt_template: Custom system prompt (uses default if None).
            agent_mode: Enable agent mode with tool calling.
            tools: List of tools for agent mode.
            llm: Custom LLM for this bot (uses global default if None).
            embedding_model: Custom embeddings for this bot.
            num_k: Custom number of retrieved documents.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found. Call initialize_bot_id() first.")

            bot = self.bot_data[bot_id]

            bot_llm = llm or self.llm
            bot_embedding = embedding_model or self.embedding_model
            bot_k = num_k or self.k

            pt = prompt_template or self.prompt_template
            bot["prompt_template"] = pt
            bot["prompt"] = _build_chat_prompt(pt)
            bot["agent_mode"] = agent_mode

            if tools:
                for t in tools:
                    bot["tools"].register(t)

            documents = self.get_documents(bot_id)
            all_splits = self.text_splitter.split_documents(documents)

            bot["retriever"] = DocumentRetriever(
                documents=all_splits,
                embedding_model=bot_embedding,
                llm=bot_llm,
                ensemble=self.ensemble,
                existing_faiss_index=(
                    bot["retriever"].faiss_index if bot["retriever"] else None
                ),
                num_k=bot_k,
            )
            bot["retriever"].save_index(file_path=bot["faiss_path"])
            bot["ensemble_retriever"] = bot["retriever"].retrieve_documents()

            self.bots.update_one(
                {"bot_id": bot_id},
                {"$set": {
                    "prompt_template": pt,
                    "agent_mode": agent_mode,
                }},
            )

            del documents, all_splits
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error creating bot: {e}")

    def load_bot(self, bot_id: str) -> None:
        """Load an existing bot from MongoDB and FAISS.

        Restores bot configuration, FAISS index, and all previous chat histories.

        Args:
            bot_id: The bot's unique identifier.

        Raises:
            ValueError: If bot_id is not provided.
        """
        if not bot_id:
            raise ValueError("Bot ID must be provided.")

        try:
            bot_config = self.bots.find_one({"bot_id": bot_id})

            if not bot_config:
                print(f"No configuration found for {bot_id}. Initializing new bot...")
                bot_config = {
                    "bot_id": bot_id,
                    "faiss_path": f"faiss_index_{bot_id}",
                    "prompt_template": self.prompt_template,
                    "agent_mode": False,
                }
                self.bots.insert_one(bot_config)

            self.bot_data[bot_id] = {
                "chains": {},
                "assistants": {},
                "retriever": None,
                "ensemble_retriever": None,
                "faiss_path": bot_config["faiss_path"],
                "prompt_template": bot_config.get("prompt_template", self.prompt_template),
                "agent_mode": bot_config.get("agent_mode", False),
                "tools": ToolRegistry(),
            }

            bot = self.bot_data[bot_id]
            bot["prompt"] = _build_chat_prompt(bot["prompt_template"])

            faiss_path = bot["faiss_path"]
            faiss_index = None
            if os.path.exists(faiss_path):
                faiss_index = FAISS.load_local(
                    faiss_path, self.embedding_model, allow_dangerous_deserialization=True
                )

            bot["retriever"] = DocumentRetriever(
                documents=[],
                embedding_model=self.embedding_model,
                llm=self.llm,
                ensemble=self.ensemble,
                existing_faiss_index=faiss_index,
                num_k=self.k,
            )
            bot["ensemble_retriever"] = bot["retriever"].retrieve_documents()

            load_chat_history = self.list_chats(bot_id)

            print("[INFO] Loading previous chats...")
            for chat_id in load_chat_history["chat_ids"]:
                data = self.get_chat_by_id(chat_id, "oldest")
                if not data:
                    continue

                rag_bot = RAGBot(
                    retriever=bot["ensemble_retriever"],
                    llm=self.llm,
                    prompt=bot["prompt"],
                    token_limit=self.max_token_limit,
                )
                for item in data:
                    rag_bot.save_context(
                        str(item["question"]),
                        str(item["answer"]),
                    )
                bot["chains"][chat_id] = rag_bot

            for vision_chat_id in load_chat_history["vision_chat_ids"]:
                data = self.get_vision_chat_by_id(vision_chat_id, "oldest")
                if not data:
                    continue

                vision_mem = VisionMemory(
                    token_limit=self.max_token_limit,
                    llm=self.llm,
                    ensemble_retriever=bot["ensemble_retriever"],
                    prompt_template=bot["prompt_template"],
                )
                for item in data:
                    vision_mem.save_context(
                        str(item["question"]),
                        str(item["response"]),
                    )
                bot["assistants"][vision_chat_id] = vision_mem

            print("[INFO] Previous chats loaded successfully.")
            gc.collect()
            print(f"[INFO] Bot {bot_id} loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Error loading bot: {e}")

    def delete_chatbot(self, bot_id: str) -> None:
        """Delete a bot and all associated data.

        Args:
            bot_id: The bot's unique identifier.

        Raises:
            ValueError: If bot_id is not found.
        """
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")

        self.chats.delete_many({"bot_id": bot_id})
        self.vision_chats.delete_many({"bot_id": bot_id})
        self.documents_collection.delete_many({"bot_id": bot_id})
        self.bots.delete_one({"bot_id": bot_id})

        bot = self.bot_data[bot_id]
        if bot["retriever"]:
            bot["retriever"].delete_index(file_path=bot["faiss_path"])
        del self.bot_data[bot_id]

        data_folder = f"./data-{bot_id}"
        try:
            if os.path.exists(data_folder):
                shutil.rmtree(data_folder)
        except Exception as e:
            print(f"[ERROR] Error deleting data folder: {e}")

    # ─── Document Management ──────────────────────────────────────────────────

    def get_documents(self, bot_id: str) -> list:
        """Retrieve documents from MongoDB for a bot.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            List of deserialized Document objects.
        """
        try:
            return [
                deserialize_document(doc["document"])
                for doc in self.documents_collection.find({"bot_id": bot_id})
            ]
        except Exception as e:
            print(f"[ERROR] Error loading documents for bot {bot_id}: {e}")
            return []

    def add_document_from_path(
        self, path: str, bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Load and store documents from a file path.

        Args:
            path: Path to the document file.
            bot_id: The bot's unique identifier.
            use_unstructured: Use UnstructuredLoader for any file type.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            if use_unstructured:
                documents = self.document_loader.load_unstructured(path)
            else:
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                loaders = {
                    "csv": self.document_loader.load_csv,
                    "docx": self.document_loader.load_doc,
                    "pdf": self.document_loader.load_pdf,
                    "md": self.document_loader.load_markdown,
                    "markdown": self.document_loader.load_markdown,
                    "txt": self.document_loader.load_markdown,
                    "html": self.document_loader.load_text_from_html,
                    "htm": self.document_loader.load_text_from_html,
                }
                loader_fn = loaders.get(ext)
                if not loader_fn:
                    raise ValueError(f"Unsupported file type: {ext}")
                documents = loader_fn(path)

            for doc in documents:
                self.documents_collection.insert_one({
                    "bot_id": bot_id,
                    "document": serialize_document(doc),
                })

            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from path: {e}")

    def add_document_from_link(self, links: list[str], bot_id: str) -> None:
        """Load and store documents from web links.

        Args:
            links: List of URLs or YouTube links.
            bot_id: The bot's unique identifier.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            for link in links:
                if "youtube.com" in link.lower() or "youtu.be" in link.lower():
                    documents = self.document_loader.load_youtube_video(link)
                else:
                    documents = self.document_loader.load_urls([link])

                for doc in documents:
                    self.documents_collection.insert_one({
                        "bot_id": bot_id,
                        "document": serialize_document(doc),
                    })
                del documents
                gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from link: {e}")

    def add_document_from_query(self, search_query: str, bot_id: str) -> None:
        """Load and store documents from a Wikipedia search.

        Args:
            search_query: Wikipedia search query.
            bot_id: The bot's unique identifier.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            documents = self.document_loader.wikipedia_query(search_query)
            for doc in documents:
                self.documents_collection.insert_one({
                    "bot_id": bot_id,
                    "document": serialize_document(doc),
                })
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding document from query: {e}")

    def pass_documents(self, documents: list, bot_id: str) -> None:
        """Store pre-loaded LangChain documents.

        Args:
            documents: List of Document objects.
            bot_id: The bot's unique identifier.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            for doc in documents:
                self.documents_collection.insert_one({
                    "bot_id": bot_id,
                    "document": serialize_document(doc),
                })
            del documents
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error adding documents: {e}")

    # ─── Tool Management ──────────────────────────────────────────────────────

    def add_tool(self, tool: BaseTool, bot_id: Optional[str] = None) -> None:
        """Register a tool globally or for a specific bot.

        Args:
            tool: A LangChain-compatible tool.
            bot_id: If provided, register only for this bot. Otherwise global.
        """
        if bot_id:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            self.bot_data[bot_id]["tools"].register(tool)
        else:
            self._global_tools.register(tool)

    def remove_tool(self, tool_name: str, bot_id: Optional[str] = None) -> None:
        """Remove a tool by name globally or from a specific bot.

        Args:
            tool_name: Name of the tool to remove.
            bot_id: If provided, remove from this bot only. Otherwise global.
        """
        if bot_id:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            self.bot_data[bot_id]["tools"].unregister(tool_name)
        else:
            self._global_tools.unregister(tool_name)

    def list_tools(self, bot_id: Optional[str] = None) -> list[str]:
        """List tool names for a bot (including global tools).

        Args:
            bot_id: If provided, list tools for this bot + global tools.

        Returns:
            List of tool name strings.
        """
        global_names = self._global_tools.list_tool_names()
        if bot_id and bot_id in self.bot_data:
            bot_names = self.bot_data[bot_id]["tools"].list_tool_names()
            return list(set(global_names + bot_names))
        return global_names

    def _get_bot_tools(self, bot_id: str) -> list[BaseTool]:
        """Get combined global + bot-specific tools.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            List of tools.
        """
        tools = self._global_tools.get_tools()
        if bot_id in self.bot_data:
            tools.extend(self.bot_data[bot_id]["tools"].get_tools())
        return tools

    # ─── Chat Sessions ────────────────────────────────────────────────────────

    def new_chat(self, bot_id: str) -> str:
        """Create a new chat session.

        Creates either a RAGBot or AgentBot based on the bot's mode.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            The generated chat_id string.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            chat_id = "chat-" + str(uuid.uuid4())
            bot = self.bot_data[bot_id]

            if bot.get("agent_mode"):
                tools = self._get_bot_tools(bot_id)
                agent_bot = AgentBot(
                    llm=self.llm,
                    tools=tools,
                    system_prompt=bot.get("prompt_template", self.prompt_template),
                    token_limit=self.max_token_limit,
                )
                bot["chains"][chat_id] = agent_bot
            else:
                rag_bot = RAGBot(
                    retriever=bot["ensemble_retriever"],
                    llm=self.llm,
                    prompt=bot["prompt"],
                    token_limit=self.max_token_limit,
                )
                bot["chains"][chat_id] = rag_bot

            return chat_id
        except Exception as e:
            print(f"[ERROR] Error creating new chat: {e}")
            return ""

    def new_vision_chat(self, bot_id: str) -> str:
        """Create a new vision chat session.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            The generated vision_chat_id string.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            vision_chat_id = "vision-" + str(uuid.uuid4())
            bot = self.bot_data[bot_id]

            vision_mem = VisionMemory(
                token_limit=self.max_token_limit,
                llm=self.llm,
                ensemble_retriever=bot["ensemble_retriever"],
                prompt_template=bot.get("prompt_template", self.prompt_template),
            )
            bot["assistants"][vision_chat_id] = vision_mem
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
        stream: bool = False,
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> Union[tuple[str, list[str]], Iterator[str]]:
        """Get a response from the chatbot.

        Args:
            query: The user's question.
            bot_id: The bot's unique identifier.
            chat_id: The chat session's unique identifier.
            stream: If True, returns an iterator yielding response chunks.
            uploaded_files: Optional list of uploaded file metadata.
            web_search: Enable web search augmentation (for RAG mode).

        Returns:
            If stream=False: (answer_string, web_sources_list)
            If stream=True: Iterator yielding response token strings
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            if chat_id not in self.bot_data[bot_id]["chains"]:
                raise ValueError(f"Chat ID {chat_id} not found in bot {bot_id}.")

            bot_instance = self.bot_data[bot_id]["chains"][chat_id]

            web_source: list[str] = []
            final_query = query

            if web_search and not self.bot_data[bot_id].get("agent_mode"):
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
                return self._stream_response(final_query, bot_id, chat_id, bot_instance, query, web_source)

            answer = bot_instance.invoke(final_query)

            self._store_chat(
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
        """Internal streaming response generator.

        Yields response chunks and stores the full response when complete.
        """
        full_response = ""
        for chunk in bot_instance.stream(final_query):
            full_response += chunk
            yield chunk

        self._store_chat(
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
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> AsyncIterator[str]:
        """Async streaming response.

        Args:
            query: The user's question.
            bot_id: The bot's unique identifier.
            chat_id: The chat session's unique identifier.
            uploaded_files: Optional uploaded file metadata.
            web_search: Enable web search augmentation.

        Yields:
            Response token strings.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            if chat_id not in self.bot_data[bot_id]["chains"]:
                raise ValueError(f"Chat ID {chat_id} not found.")

            bot_instance = self.bot_data[bot_id]["chains"][chat_id]
            final_query = query

            if web_search and not self.bot_data[bot_id].get("agent_mode"):
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

            self._store_chat(
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
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> tuple[str, list[str]]:
        """Get a response from the vision AI assistant.

        Args:
            query: Text query for the vision model.
            image_paths: List of image file paths.
            bot_id: The bot's unique identifier.
            vision_chat_id: The vision chat session ID.
            uploaded_files: Optional uploaded file metadata.
            web_search: Enable web search augmentation.

        Returns:
            Tuple of (response_string, web_sources_list).
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            if vision_chat_id not in self.bot_data[bot_id]["assistants"]:
                raise ValueError(f"Vision chat ID {vision_chat_id} not found.")

            web_source: list[str] = []
            web_text = None
            if web_search:
                web_text = self._web_search(query)
                web_source = self._extract_web_links(web_text)

            assistant = self.bot_data[bot_id]["assistants"][vision_chat_id]

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

            self._store_vision_chat(
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

    # ─── Update Bot ───────────────────────────────────────────────────────────

    def update_chatbot(
        self,
        paths: list[str],
        bot_id: str,
        links: Optional[list[str]] = None,
        search_query: Optional[str] = None,
        documents: Optional[list] = None,
        prompt_template: Optional[str] = None,
        use_unstructured: bool = False,
    ) -> None:
        """Update a bot with new documents.

        Args:
            paths: List of file paths to load.
            bot_id: The bot's unique identifier.
            links: Optional list of web links.
            search_query: Optional Wikipedia search query.
            documents: Optional pre-loaded documents.
            prompt_template: Optional new prompt template.
            use_unstructured: Use UnstructuredLoader for files.
        """
        try:
            existing_count = self.documents_collection.count_documents({"bot_id": bot_id})

            for path in paths:
                self.add_document_from_path(path=path, bot_id=bot_id, use_unstructured=use_unstructured)
            if links:
                self.add_document_from_link(links, bot_id)
            if search_query:
                self.add_document_from_query(search_query, bot_id)
            if documents:
                self.pass_documents(documents, bot_id)

            updated_documents = self.get_documents(bot_id)
            new_docs = updated_documents[existing_count:]

            bot = self.bot_data[bot_id]
            if bot["retriever"] and bot["retriever"].faiss_index:
                all_splits = self.text_splitter.split_documents(new_docs)
                bot["retriever"].update_index(all_splits)
                bot["retriever"].save_index(file_path=bot["faiss_path"])
                bot["ensemble_retriever"] = bot["retriever"].retrieve_documents()
            else:
                all_splits = self.text_splitter.split_documents(updated_documents)
                bot["retriever"] = DocumentRetriever(
                    documents=all_splits,
                    embedding_model=self.embedding_model,
                    llm=self.llm,
                    ensemble=self.ensemble,
                    existing_faiss_index=(
                        bot["retriever"].faiss_index if bot["retriever"] else None
                    ),
                    num_k=self.k,
                )
                bot["retriever"].save_index(file_path=bot["faiss_path"])
                bot["ensemble_retriever"] = bot["retriever"].retrieve_documents()

            self.create_bot(bot_id, prompt_template)

            del updated_documents, new_docs
            gc.collect()
        except Exception as e:
            print(f"[ERROR] Error updating chatbot: {e}")
            gc.collect()

    def set_custom_prompt_template(self, bot_id: str, prompt_template: str) -> None:
        """Set a custom system prompt for a bot.

        Args:
            bot_id: The bot's unique identifier.
            prompt_template: The new prompt template string.
        """
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            self.bot_data[bot_id]["prompt_template"] = prompt_template
            self.bot_data[bot_id]["prompt"] = _build_chat_prompt(prompt_template)
            self.bots.update_one(
                {"bot_id": bot_id},
                {"$set": {"prompt_template": prompt_template}},
            )
        except Exception as e:
            print(f"[ERROR] Error setting prompt template: {e}")

    # ─── Vector Store Direct Access ───────────────────────────────────────────

    def invoke_vectorstore(self, bot_id: str, query: str) -> list:
        """Retrieve similar documents directly from the vector store.

        Args:
            bot_id: The bot's unique identifier.
            query: The search query.

        Returns:
            List of retrieved Document objects.
        """
        try:
            return self.bot_data[bot_id]["ensemble_retriever"].invoke(query)
        except Exception as e:
            print(f"[ERROR] Error invoking vector store: {e}")
            return []

    # ─── Chat History ─────────────────────────────────────────────────────────

    def list_chats(self, bot_id: str) -> dict:
        """List all chat and vision chat IDs for a bot.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            Dict with "chat_ids" and "vision_chat_ids" lists.
        """
        try:
            return {
                "chat_ids": list(self.chats.distinct("chat_id", {"bot_id": bot_id})),
                "vision_chat_ids": list(
                    self.vision_chats.distinct("vision_chat_id", {"bot_id": bot_id})
                ),
            }
        except Exception as e:
            print(f"[ERROR] Error listing chats: {e}")
            return {"chat_ids": [], "vision_chat_ids": []}

    def get_chat_by_id(self, chat_id: str, order: str = "newest") -> Optional[list[dict]]:
        """Retrieve full chat history for a chat session.

        Args:
            chat_id: The chat session's unique identifier.
            order: Sort order — "newest" or "oldest".

        Returns:
            List of chat records, or None if not found.
        """
        try:
            sort_order = -1 if order == "newest" else 1
            chat_data = list(self.chats.find({"chat_id": chat_id}).sort("timestamp", sort_order))
            if not chat_data:
                return None

            if self.encrypt_chats:
                for chat in chat_data:
                    chat["question"] = self._decrypt_data(chat["question"])
                    chat["answer"] = self._decrypt_data(chat["answer"])
                    if chat.get("web_sources"):
                        chat["web_sources"] = [
                            self._decrypt_data(s) for s in chat["web_sources"]
                        ]
            return chat_data
        except Exception as e:
            print(f"[ERROR] Error getting chat {chat_id}: {e}")
            return None

    def get_vision_chat_by_id(
        self, vision_chat_id: str, order: str = "newest"
    ) -> Optional[list[dict]]:
        """Retrieve full vision chat history.

        Args:
            vision_chat_id: The vision chat session's unique identifier.
            order: Sort order — "newest" or "oldest".

        Returns:
            List of vision chat records, or None if not found.
        """
        try:
            sort_order = -1 if order == "newest" else 1
            data = list(
                self.vision_chats.find({"vision_chat_id": vision_chat_id}).sort(
                    "timestamp", sort_order
                )
            )
            if not data:
                return None

            if self.encrypt_chats:
                for chat in data:
                    chat["question"] = self._decrypt_data(chat["question"])
                    chat["response"] = self._decrypt_data(chat["response"])
                    if chat.get("web_sources"):
                        chat["web_sources"] = [
                            self._decrypt_data(s) for s in chat["web_sources"]
                        ]
            return data
        except Exception as e:
            print(f"[ERROR] Error getting vision chat {vision_chat_id}: {e}")
            return None

    # ─── Train on Chats ───────────────────────────────────────────────────────

    def train_chats(self, bot_id: str) -> dict:
        """Train the bot on its own unprocessed chat history.

        Exports new chats to CSV, adds them as documents, and rebuilds the bot.

        Args:
            bot_id: The bot's unique identifier.

        Returns:
            Dict with "message" and "csv_path" keys.
        """
        try:
            new_chats = list(
                self.chats.find({"bot_id": bot_id, "trained": {"$ne": True}})
            )
            if not new_chats:
                return {"message": "No new chats found for training.", "csv_path": None}

            if self.encrypt_chats:
                for chat in new_chats:
                    chat["question"] = self._decrypt_data(chat["question"])
                    chat["answer"] = self._decrypt_data(chat["answer"])

            df = pd.DataFrame(new_chats, columns=["question", "answer"])
            df.columns = ["Question", "Answer"]

            csv_path = self._export_chats_to_csv(df, bot_id)

            chat_ids = [chat["_id"] for chat in new_chats]
            self.chats.update_many(
                {"_id": {"$in": chat_ids}}, {"$set": {"trained": True}}
            )

            self.add_document_from_path(path=csv_path, bot_id=bot_id)
            self.create_bot(bot_id=bot_id)

            return {"message": "Trainer updated with new chat history.", "csv_path": csv_path}
        except Exception as e:
            print(f"[ERROR] Error training on chats: {e}")
            return {"message": f"Error: {e}", "csv_path": None}

    # ─── Internal Helpers ─────────────────────────────────────────────────────

    def _web_search(self, query: str) -> str:
        """Perform a DuckDuckGo web search.

        Args:
            query: Search query.

        Returns:
            Formatted search results string.
        """
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

    def _extract_web_links(self, text: str) -> list[str]:
        """Extract links from web search results text.

        Args:
            text: Search results text.

        Returns:
            List of extracted URLs.
        """
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

    def _store_chat(
        self,
        bot_id: str,
        chat_id: str,
        query: str,
        answer: str,
        web_source: Optional[list[str]] = None,
        uploaded_files: Optional[list[dict]] = None,
    ) -> None:
        """Store a chat exchange in MongoDB."""
        try:
            current_time = datetime.now(timezone.utc)

            if self.encrypt_chats:
                enc_query = self._encrypt_data(query)
                enc_answer = self._encrypt_data(answer)
                enc_sources = (
                    [self._encrypt_data(s) for s in web_source]
                    if web_source
                    else []
                )
            else:
                enc_query, enc_answer = query, answer
                enc_sources = web_source or []

            self.chats.insert_one({
                "bot_id": bot_id,
                "chat_id": chat_id,
                "timestamp": current_time,
                "question": enc_query,
                "answer": enc_answer,
                "web_sources": enc_sources,
                "uploaded_files": uploaded_files,
                "trained": False,
            })
        except Exception as e:
            print(f"[ERROR] Error storing chat: {e}")

    def _store_vision_chat(
        self,
        bot_id: str,
        vision_chat_id: str,
        image_paths: list[str],
        query: str,
        response: str,
        web_source: Optional[list[str]] = None,
        uploaded_files: Optional[list[dict]] = None,
    ) -> None:
        """Store a vision chat exchange in MongoDB."""
        try:
            current_time = datetime.now(timezone.utc)

            if self.encrypt_chats:
                enc_query = self._encrypt_data(query)
                enc_response = self._encrypt_data(response)
                enc_sources = (
                    [self._encrypt_data(s) for s in web_source]
                    if web_source
                    else []
                )
            else:
                enc_query, enc_response = query, response
                enc_sources = web_source or []

            self.vision_chats.insert_one({
                "bot_id": bot_id,
                "vision_chat_id": vision_chat_id,
                "timestamp": current_time,
                "image_path": ",".join(image_paths),
                "question": enc_query,
                "response": enc_response,
                "web_sources": enc_sources,
                "uploaded_files": uploaded_files,
                "trained": False,
            })
        except Exception as e:
            print(f"[ERROR] Error storing vision chat: {e}")

    def _encrypt_data(self, data: str) -> str:
        """Encrypt a string using Fernet."""
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            print(f"[ERROR] Encryption error: {e}")
            return data

    def _decrypt_data(self, data: str) -> str:
        """Decrypt a Fernet-encrypted string."""
        try:
            return self.fernet.decrypt(data.encode()).decode()
        except Exception as e:
            print(f"[ERROR] Decryption error: {e}")
            return data

    def _export_chats_to_csv(self, dataframe: pd.DataFrame, bot_id: str) -> str:
        """Export a DataFrame of chats to a CSV file."""
        try:
            csv_folder = f"./data-{bot_id}"
            os.makedirs(csv_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            csv_path = os.path.join(csv_folder, f"{bot_id}_data_{timestamp}.csv")
            dataframe.to_csv(csv_path, index=False)
            return csv_path
        except Exception as e:
            print(f"[ERROR] Error exporting chats: {e}")
            return ""
