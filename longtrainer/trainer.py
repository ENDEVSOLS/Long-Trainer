"""LongTrainer — Production-Ready RAG Framework.

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
import shutil
import uuid
from typing import AsyncIterator, Iterator, Optional, Union

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from longtrainer.bot import RAGBot, AgentBot
from longtrainer.chat import ChatManager, build_chat_prompt
from longtrainer.config import LongTrainerConfig, _DEFAULT_SYSTEM_PROMPT
from longtrainer.documents import DocumentManager
from longtrainer.loaders import DocumentLoader, TextSplitter
from longtrainer.retrieval import MultiQueryEnsembleRetriever
from longtrainer.vectorstores import get_vectorstore, save_vectorstore, delete_vectorstore
from longtrainer.storage import MongoStorage
from longtrainer.tools import ToolRegistry, get_builtin_tools
from longtrainer.utils import deserialize_document, serialize_document
from longtrainer.vision_bot import VisionBot, VisionMemory
from longtrainer.models import get_llm, get_embedding_model


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
        llm_provider: str = "openai",
        default_llm: str = "gpt-4o-2024-08-06",
        embedding_provider: str = "openai",
        embedding_model_name: str = "text-embedding-3-small",
        vector_store_provider: str = "faiss",
        prompt_template: Optional[str] = None,
        max_token_limit: int = 32000,
        num_k: int = 3,
        chunk_size: int = 2048,
        chunk_overlap: int = 200,
        ensemble: bool = False,
        encrypt_chats: bool = False,
        encryption_key: Optional[bytes] = None,
    ) -> None:
        # Models
        self.llm = llm or get_llm(llm_provider, default_llm)
        self.embedding_model = embedding_model or get_embedding_model(embedding_provider, embedding_model_name)
        self.prompt_template = prompt_template or _DEFAULT_SYSTEM_PROMPT
        self.prompt = build_chat_prompt(self.prompt_template)
        self.k = num_k
        self.max_token_limit = max_token_limit
        self.ensemble = ensemble

        # Internal config
        self._config = LongTrainerConfig(
            mongo_endpoint=mongo_endpoint,
            llm_provider=llm_provider,
            default_llm=default_llm,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model_name,
            vector_store_provider=vector_store_provider,
            vector_store_kwargs=vector_store_kwargs or {},
            prompt_template=self.prompt_template,
            max_token_limit=max_token_limit,
            num_k=num_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ensemble=ensemble,
            encrypt_chats=encrypt_chats,
            encryption_key=encryption_key,
        )

        # Managers
        self._storage = MongoStorage(self._config)
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._doc_manager = DocumentManager(self._storage, self.document_loader)
        self._chat_manager = ChatManager(self._storage, self.llm, max_token_limit)

        # Bot runtime state
        self.bot_data: dict = {}
        self._global_tools = ToolRegistry()

        # Expose storage collections for backwards compatibility
        self.client = self._storage.client
        self.db = self._storage.db
        self.bots = self._storage.bots
        self.chats = self._storage.chats
        self.vision_chats = self._storage.vision_chats
        self.documents_collection = self._storage.documents_collection

        # Encryption attributes for backwards compatibility
        self.encrypt_chats = encrypt_chats
        if encrypt_chats:
            self.encryption_key = self._storage.encryption_key
            self.fernet = self._storage._fernet

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
                "vectorstore": None,
                "ensemble_retriever": None,
                "db_path": f"db_{bot_id}",
                "agent_mode": False,
                "tools": ToolRegistry(),
            }
            self._storage.save_bot(bot_id, self.bot_data[bot_id]["db_path"])
            return bot_id
        except Exception as e:
            print(f"[ERROR] Error initializing bot: {e}")
            return ""

    def create_bot(
        self,
        bot_id: str,
        prompt_template: Optional[str] = None,
        agent_mode: bool = False,
        tools: Optional[list] = None,
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
            bot["prompt"] = build_chat_prompt(pt)
            bot["agent_mode"] = agent_mode

            if tools:
                from longtrainer.tools import load_dynamic_tools
                dynamic_tool_names = [t for t in tools if isinstance(t, str)]
                if dynamic_tool_names:
                    dynamic_tools = load_dynamic_tools(dynamic_tool_names)
                    for t in dynamic_tools:
                        if not bot["tools"].has_tool(t.name):
                            bot["tools"].register(t)
                for t in tools:
                    if not isinstance(t, str):
                        if not bot["tools"].has_tool(t.name):
                            bot["tools"].register(t)

            documents = self._doc_manager.get_documents(bot_id)
            all_splits = self.text_splitter.split_documents(documents)

            bot["vectorstore"] = get_vectorstore(
                provider=self._config.vector_store_provider,
                embedding=bot_embedding,
                collection_name=bot_id,
                persist_directory=bot["db_path"],
                **self._config.vector_store_kwargs,
            )

            if all_splits:
                bot["vectorstore"].add_documents(all_splits)
                save_vectorstore(bot["vectorstore"], self._config.vector_store_provider, bot["db_path"])

            base_retriever = bot["vectorstore"].as_retriever(search_kwargs={"k": bot_k})

            if self.ensemble:
                bot["ensemble_retriever"] = MultiQueryEnsembleRetriever(
                    base_retriever=base_retriever,
                    llm=bot_llm,
                    k=bot_k,
                )
            else:
                bot["ensemble_retriever"] = base_retriever

            bot["retriever"] = bot["ensemble_retriever"]

            self._storage.update_bot(bot_id, {
                "prompt_template": pt,
                "agent_mode": agent_mode,
                "dynamic_tools": dynamic_tool_names if tools else [],
            })

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
            bot_config = self._storage.find_bot(bot_id)

            if not bot_config:
                print(f"No configuration found for {bot_id}. Initializing new bot...")
                bot_config = {
                    "bot_id": bot_id,
                    "db_path": f"db_{bot_id}",
                    "prompt_template": self.prompt_template,
                    "agent_mode": False,
                }
                self._storage.save_bot(bot_id, bot_config["db_path"])

            self.bot_data[bot_id] = {
                "chains": {},
                "assistants": {},
                "retriever": None,
                "vectorstore": None,
                "ensemble_retriever": None,
                "db_path": bot_config.get("db_path", bot_config.get("faiss_path", f"db_{bot_id}")),
                "prompt_template": bot_config.get("prompt_template", self.prompt_template),
                "agent_mode": bot_config.get("agent_mode", False),
                "tools": ToolRegistry(),
            }

            bot = self.bot_data[bot_id]
            
            # Restore dynamic tools
            dynamic_tools_list = bot_config.get("dynamic_tools", [])
            if dynamic_tools_list:
                from longtrainer.tools import load_dynamic_tools
                dynamic_tools = load_dynamic_tools(dynamic_tools_list)
                for t in dynamic_tools:
                    if not bot["tools"].has_tool(t.name):
                        bot["tools"].register(t)
                        
            bot["prompt"] = build_chat_prompt(bot["prompt_template"])

            bot["vectorstore"] = get_vectorstore(
                provider=self._config.vector_store_provider,
                embedding=self.embedding_model,
                collection_name=bot_id,
                persist_directory=bot["db_path"],
                **self._config.vector_store_kwargs,
            )

            base_retriever = bot["vectorstore"].as_retriever(search_kwargs={"k": self.k})

            if self.ensemble:
                bot["ensemble_retriever"] = MultiQueryEnsembleRetriever(
                    base_retriever=base_retriever,
                    llm=self.llm,
                    k=self.k,
                )
            else:
                bot["ensemble_retriever"] = base_retriever

            bot["retriever"] = bot["ensemble_retriever"]

            load_chat_history = self._storage.list_chats(bot_id)

            print("[INFO] Loading previous chats...")
            for chat_id in load_chat_history["chat_ids"]:
                data = self._storage.get_chat_by_id(chat_id, "oldest")
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
                data = self._storage.get_vision_chat_by_id(vision_chat_id, "oldest")
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

        self._storage.delete_bot(bot_id)

        bot = self.bot_data[bot_id]
        delete_vectorstore(self._config.vector_store_provider, bot_id, bot["db_path"])
        del self.bot_data[bot_id]

        data_folder = f"./data-{bot_id}"
        try:
            if os.path.exists(data_folder):
                shutil.rmtree(data_folder)
        except Exception as e:
            print(f"[ERROR] Error deleting data folder: {e}")

    # ─── Document Management ──────────────────────────────────────────────────

    def get_documents(self, bot_id: str) -> list:
        """Retrieve documents from MongoDB for a bot."""
        return self._doc_manager.get_documents(bot_id)

    def add_document_from_path(
        self, path: str, bot_id: str, use_unstructured: bool = False
    ) -> None:
        """Load and store documents from a file path."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_path(path, bot_id, use_unstructured)

    def add_document_from_link(self, links: list[str], bot_id: str) -> None:
        """Load and store documents from web links."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_link(links, bot_id)

    def add_document_from_query(self, search_query: str, bot_id: str) -> None:
        """Load and store documents from a Wikipedia search."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_query(search_query, bot_id)

    def add_document_from_github(self, repo_url: str, bot_id: str, branch: str = "main", access_token: Optional[str] = None) -> None:
        """Load and store documents from a GitHub repository."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_github(repo_url, bot_id, branch, access_token)

    def add_document_from_notion(self, path: str, bot_id: str) -> None:
        """Load and store documents from an exported Notion directory."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_notion(path, bot_id)

    def add_document_from_crawl(self, url: str, bot_id: str, max_depth: int = 2) -> None:
        """Deep crawl a website and store documents."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_crawl(url, bot_id, max_depth)

    def add_document_from_dynamic_loader(self, bot_id: str, loader_class_name: str, **kwargs) -> None:
        """Instantiate ANY LangChain document loader dynamically."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_dynamic_loader(bot_id, loader_class_name, **kwargs)

    def add_document_from_directory(self, path: str, bot_id: str, glob: str = "**/*") -> None:
        """Load documents recursively from a local directory."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_directory(path, bot_id, glob)

    def add_document_from_json(self, path: str, bot_id: str, jq_schema: str = ".") -> None:
        """Load documents from a JSON or JSONL file."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_json(path, bot_id, jq_schema)

    def add_document_from_aws_s3(self, bucket: str, bot_id: str, prefix: str = "", aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None) -> None:
        """Load documents from an AWS S3 Directory."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_aws_s3(bucket, bot_id, prefix, aws_access_key_id, aws_secret_access_key)

    def add_document_from_google_drive(self, folder_id: str, bot_id: str, credentials_path: str = "credentials.json") -> None:
        """Load documents from a Google Drive folder."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_google_drive(folder_id, bot_id, credentials_path)

    def add_document_from_confluence(self, url: str, username: str, api_key: str, bot_id: str, space_key: Optional[str] = None) -> None:
        """Load documents from a Confluence Workspace."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.add_document_from_confluence(url, username, api_key, bot_id, space_key)

    def pass_documents(self, documents: list, bot_id: str) -> None:
        """Store pre-loaded LangChain documents."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        self._doc_manager.pass_documents(documents, bot_id)
    # ─── Tool Management ──────────────────────────────────────────────────────

    def add_tool(self, tool: BaseTool, bot_id: Optional[str] = None) -> None:
        """Register a tool globally or for a specific bot."""
        if bot_id:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            self.bot_data[bot_id]["tools"].register(tool)
        else:
            self._global_tools.register(tool)

    def remove_tool(self, tool_name: str, bot_id: Optional[str] = None) -> None:
        """Remove a tool by name globally or from a specific bot."""
        if bot_id:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")
            self.bot_data[bot_id]["tools"].unregister(tool_name)
        else:
            self._global_tools.unregister(tool_name)

    def list_tools(self, bot_id: Optional[str] = None) -> list[str]:
        """List tool names for a bot (including global tools)."""
        global_names = self._global_tools.list_tool_names()
        if bot_id and bot_id in self.bot_data:
            bot_names = self.bot_data[bot_id]["tools"].list_tool_names()
            return list(set(global_names + bot_names))
        return global_names

    def _get_bot_tools(self, bot_id: str) -> list[BaseTool]:
        """Get combined global + bot-specific tools."""
        tools = self._global_tools.get_tools()
        if bot_id in self.bot_data:
            tools.extend(self.bot_data[bot_id]["tools"].get_tools())
        return tools

    # ─── Chat Sessions ────────────────────────────────────────────────────────

    def new_chat(self, bot_id: str) -> str:
        """Create a new chat session."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        return self._chat_manager.new_chat(
            self.bot_data[bot_id], bot_id, self.prompt_template, self._global_tools
        )

    def new_vision_chat(self, bot_id: str) -> str:
        """Create a new vision chat session."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        return self._chat_manager.new_vision_chat(
            self.bot_data[bot_id], self.prompt_template
        )

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
        """Get a response from the chatbot."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        return self._chat_manager.get_response(
            query, bot_id, chat_id, self.bot_data[bot_id],
            stream, uploaded_files, web_search,
        )

    async def aget_response(
        self,
        query: str,
        bot_id: str,
        chat_id: str,
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> AsyncIterator[str]:
        """Async streaming response."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        async for chunk in self._chat_manager.aget_response(
            query, bot_id, chat_id, self.bot_data[bot_id],
            uploaded_files, web_search,
        ):
            yield chunk

    def get_vision_response(
        self,
        query: str,
        image_paths: list[str],
        bot_id: str,
        vision_chat_id: str,
        uploaded_files: Optional[list[dict]] = None,
        web_search: bool = False,
    ) -> tuple[str, list[str]]:
        """Get a response from the vision AI assistant."""
        if bot_id not in self.bot_data:
            raise ValueError(f"Bot ID {bot_id} not found.")
        return self._chat_manager.get_vision_response(
            query, image_paths, bot_id, vision_chat_id,
            self.bot_data[bot_id], uploaded_files, web_search,
        )

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
        """Update a bot with new documents."""
        try:
            existing_count = self._doc_manager.count_documents(bot_id)

            for path in paths:
                self._doc_manager.add_document_from_path(
                    path=path, bot_id=bot_id, use_unstructured=use_unstructured
                )
            if links:
                self._doc_manager.add_document_from_link(links, bot_id)
            if search_query:
                self._doc_manager.add_document_from_query(search_query, bot_id)
            if documents:
                self._doc_manager.pass_documents(documents, bot_id)

            updated_documents = self._doc_manager.get_documents(bot_id)
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
        """Set a custom system prompt for a bot."""
        try:
            if bot_id not in self.bot_data:
                raise ValueError(f"Bot ID {bot_id} not found.")

            self.bot_data[bot_id]["prompt_template"] = prompt_template
            self.bot_data[bot_id]["prompt"] = build_chat_prompt(prompt_template)
            self._storage.update_bot(bot_id, {"prompt_template": prompt_template})
        except Exception as e:
            print(f"[ERROR] Error setting prompt template: {e}")

    # ─── Vector Store Direct Access ───────────────────────────────────────────

    def invoke_vectorstore(self, bot_id: str, query: str) -> list:
        """Retrieve similar documents directly from the vector store."""
        try:
            return self.bot_data[bot_id]["ensemble_retriever"].invoke(query)
        except Exception as e:
            print(f"[ERROR] Error invoking vector store: {e}")
            return []

    # ─── Chat History ─────────────────────────────────────────────────────────

    def list_chats(self, bot_id: str) -> dict:
        """List all chat and vision chat IDs for a bot."""
        return self._storage.list_chats(bot_id)

    def get_chat_by_id(self, chat_id: str, order: str = "newest") -> Optional[list[dict]]:
        """Retrieve full chat history for a chat session."""
        return self._storage.get_chat_by_id(chat_id, order)

    def get_vision_chat_by_id(
        self, vision_chat_id: str, order: str = "newest"
    ) -> Optional[list[dict]]:
        """Retrieve full vision chat history."""
        return self._storage.get_vision_chat_by_id(vision_chat_id, order)

    # ─── Train on Chats ───────────────────────────────────────────────────────

    def train_chats(self, bot_id: str) -> dict:
        """Train the bot on its own unprocessed chat history."""
        try:
            new_chats = self._storage.find_untrained_chats(bot_id)
            if not new_chats:
                return {"message": "No new chats found for training.", "csv_path": None}

            if self.encrypt_chats:
                for chat in new_chats:
                    chat["question"] = self._storage.decrypt_data(chat["question"])
                    chat["answer"] = self._storage.decrypt_data(chat["answer"])

            df = pd.DataFrame(new_chats, columns=["question", "answer"])
            df.columns = ["Question", "Answer"]

            csv_path = self._storage.export_chats_to_csv(df, bot_id)

            chat_ids = [chat["_id"] for chat in new_chats]
            self._storage.mark_chats_trained(chat_ids)

            self._doc_manager.add_document_from_path(path=csv_path, bot_id=bot_id)
            self.create_bot(bot_id=bot_id)

            return {"message": "Trainer updated with new chat history.", "csv_path": csv_path}
        except Exception as e:
            print(f"[ERROR] Error training on chats: {e}")
            return {"message": f"Error: {e}", "csv_path": None}

    # ─── Internal Helpers (preserved for backwards compatibility) ──────────────

    def _web_search(self, query: str) -> str:
        """Perform a DuckDuckGo web search."""
        return ChatManager._web_search(query)

    def _extract_web_links(self, text: str) -> list[str]:
        """Extract links from web search results text."""
        return ChatManager._extract_web_links(text)

    def _store_chat(self, **kwargs) -> None:
        """Store a chat exchange in MongoDB."""
        self._storage.store_chat(**kwargs)

    def _store_vision_chat(self, **kwargs) -> None:
        """Store a vision chat exchange in MongoDB."""
        self._storage.store_vision_chat(**kwargs)

    def _encrypt_data(self, data: str) -> str:
        """Encrypt a string using Fernet."""
        return self._storage.encrypt_data(data)

    def _decrypt_data(self, data: str) -> str:
        """Decrypt a Fernet-encrypted string."""
        return self._storage.decrypt_data(data)

    def _export_chats_to_csv(self, dataframe: pd.DataFrame, bot_id: str) -> str:
        """Export a DataFrame of chats to a CSV file."""
        return self._storage.export_chats_to_csv(dataframe, bot_id)
