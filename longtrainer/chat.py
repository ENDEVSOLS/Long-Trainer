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
        enable_tracer: Whether LongTracer tracing is active.
        tracer_verify: Run CitationVerifier for hallucination detection.
        tracer_threshold: Hallucination detection threshold.
    """

    def __init__(
        self,
        storage: MongoStorage,
        llm: BaseChatModel,
        max_token_limit: int = 32000,
        enable_tracer: bool = False,
        tracer_verify: bool = True,
        tracer_threshold: float = 0.5,
        rate_limiter=None,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.enable_tracer = enable_tracer
        self.tracer_verify = tracer_verify
        self.tracer_threshold = tracer_threshold
        self._rate_limiter = rate_limiter

    # ─── Tracer Helpers ──────────────────────────────────────────────────────

    def _build_tracer_config(
        self,
        bot_id: str,
        chat_id: str,
        chat_type: str,
        is_agent: bool,
    ) -> tuple[Optional[dict], Optional[object]]:
        """Build a LangChain config dict with the appropriate tracer handler.

        Uses the default tracer (no args) so that callback handler spans
        and manual root runs target the same Tracer instance.

        Args:
            bot_id: The bot's unique identifier.
            chat_id: The chat session's unique identifier.
            chat_type: "rag" or "agent".
            is_agent: Whether this is an agent bot.

        Returns:
            Tuple of (config_dict_or_None, tracer_or_None).
            For AgentBot: tracer is None (handler auto-manages root).
            For RAGBot: tracer is returned (caller manages root lifecycle).
        """
        if not self.enable_tracer:
            return None, None
        try:
            from longtracer import LongTracer as LT

            tracer = LT.get_tracer()  # default tracer — same one handlers use
            if not tracer:
                return None, None

            if is_agent:
                from longtracer.adapters.langgraph_handler import LongTracerAgentHandler
                handler = LongTracerAgentHandler(threshold=self.tracer_threshold)
                return {"callbacks": [handler]}, None  # handler auto-manages root
            else:
                from longtracer.adapters.langchain_handler import LongTracerCallbackHandler
                handler = LongTracerCallbackHandler()
                tracer.start_root(inputs={
                    "bot_id": bot_id,
                    "chat_id": chat_id,
                    "chat_type": chat_type,
                    "query_preview": "",
                })
                return {"callbacks": [handler]}, tracer  # caller manages root
        except ImportError:
            return None, None
        except Exception as e:
            print(f"[WARN] Tracer setup failed: {e}")
            return None, None

    def _inject_trace_metadata(
        self,
        tracer: Optional[object],
        bot_id: str,
        chat_id: str,
        chat_type: str,
    ) -> None:
        """Inject bot_id/chat_id metadata into the current root trace run.

        Uses ``tracer._safe_update_run`` (internal API) as a workaround
        for handlers that auto-manage root without caller metadata.
        Wrapped in try/except — never propagates.

        Long-term: replace with public ``tracer.update_root_metadata()``
        when LongTracer exposes one.
        """
        try:
            if tracer and hasattr(tracer, "root_run") and tracer.root_run:
                tracer._safe_update_run(
                    tracer.root_run["run_id"],
                    {
                        "inputs.bot_id": bot_id,
                        "inputs.chat_id": chat_id,
                        "inputs.chat_type": chat_type,
                    },
                )
        except Exception:
            pass  # Never crash the pipeline for metadata injection

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
        schema: Optional[dict] = None,
    ) -> Union[tuple, Iterator[str]]:
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
                if self._rate_limiter:
                    self._rate_limiter.check_and_consume("tool_calls", bot_id)
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

            # ── Tracer setup ────────────────────────────────────────────────────
            is_agent = bot_data.get("agent_mode", False)

            # ── Structured output path (manual tracing, no callback handler) ─────
            # invoke_structured bypasses self.chain, so callback-based tracing
            # cannot capture it. Use manual start_root/span/end_root instead.
            # Checked before standard tracer setup so we don't create an unused
            # callback handler + root trace.
            if schema and not is_agent and hasattr(bot_instance, "invoke_structured"):
                structured_tracer = None
                if self.enable_tracer:
                    try:
                        from longtracer import LongTracer as LT
                        structured_tracer = LT.get_tracer()
                        if structured_tracer:
                            structured_tracer.start_root(inputs={
                                "bot_id": bot_id,
                                "chat_id": chat_id,
                                "chat_type": "structured",
                                "query_preview": query[:300],
                            })
                    except ImportError:
                        pass
                    except Exception as te:
                        print(f"[WARN] Structured tracer setup failed: {te}")
                        structured_tracer = None

                try:
                    if structured_tracer:
                        with structured_tracer.span(
                            "invoke_structured",
                            run_type="llm",
                            inputs={
                                "query_preview": query[:300],
                                "schema_keys": list(schema.get("properties", {}).keys()) if isinstance(schema, dict) else [],
                            },
                        ) as span:
                            structured = bot_instance.invoke_structured(final_query, schema)
                            span.set_output({
                                "result_status": structured.get("status", "unknown"),
                                "answer_preview": str(structured.get("data", ""))[:500],
                            })
                    else:
                        structured = bot_instance.invoke_structured(final_query, schema)
                finally:
                    if structured_tracer:
                        try:
                            structured_tracer.end_root(outputs={
                                "answer_preview": str(structured.get("data", ""))[:300],
                            })
                        except Exception:
                            pass

                self.storage.store_chat(
                    bot_id=bot_id,
                    chat_id=chat_id,
                    query=query,
                    answer=str(structured.get("data", "")),
                    web_source=web_source,
                    uploaded_files=uploaded_files,
                )
                return structured, web_source

            # ── RAG / Agent path (callback-based tracing) ─────────────────────────
            config, tracer = self._build_tracer_config(
                bot_id, chat_id, "agent" if is_agent else "rag", is_agent,
            )

            if stream:
                return self._stream_response(
                    final_query, bot_id, chat_id, bot_instance, query, web_source,
                    config=config, tracer=tracer,
                )

            # ── Standard path ────────────────────────────────────────────────────
            if self._rate_limiter:
                self._rate_limiter.check_and_consume("llm_calls", bot_id)
            answer = bot_instance.invoke(final_query, config=config)

            # Agent path: inject metadata (handler auto-managed root)
            if is_agent:
                try:
                    from longtracer import LongTracer as LT
                    self._inject_trace_metadata(
                        LT.get_tracer(), bot_id, chat_id, "agent",
                    )
                except Exception:
                    pass

            # RAG path: caller manages root lifecycle
            if tracer:
                tracer.end_root(outputs={"answer_preview": answer[:300]})

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
            # Re-raise rate limit errors — they must propagate to API/CLI
            from longtrainer.rate_limiter import LongTrainerRateLimitError
            if isinstance(e, LongTrainerRateLimitError):
                raise
            # Ensure tracer root is closed even on error
            try:
                if tracer:
                    tracer.end_root(outputs={"error": str(e)})
            except Exception:
                pass
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
        config: Optional[dict] = None,
        tracer: Optional[object] = None,
    ) -> Iterator[str]:
        """Internal streaming response generator."""
        full_response = ""
        try:
            if self._rate_limiter:
                self._rate_limiter.check_and_consume("llm_calls", bot_id)
            for chunk in bot_instance.stream(final_query, config=config):
                full_response += chunk
                yield chunk
        finally:
            # Close tracer root after streaming completes
            try:
                if tracer:
                    tracer.end_root(outputs={"answer_preview": full_response[:300]})
            except Exception:
                pass
            # Agent metadata injection
            try:
                if config and not tracer:  # agent path: handler managed root, tracer is None
                    from longtracer import LongTracer as LT
                    self._inject_trace_metadata(
                        LT.get_tracer(), bot_id, chat_id, "agent",
                    )
            except Exception:
                pass
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
                if self._rate_limiter:
                    self._rate_limiter.check_and_consume("tool_calls", bot_id)
                webdata = self._web_search(query)
                final_query = f"{query}\n\nAdditional web context:\n{webdata}"

            if uploaded_files:
                file_details = "\n".join(
                    f"File: {f['name']} (Type: {f['type']})\n"
                    f"Extracted Text: {f.get('extracted_text', 'N/A')}"
                    for f in uploaded_files
                )
                final_query = f"Uploaded Files:\n{file_details}\n\nQuestion:\n{final_query}"

            # ── Tracer setup ────────────────────────────────────────────────────
            is_agent = bot_data.get("agent_mode", False)
            config, tracer = self._build_tracer_config(
                bot_id, chat_id, "agent" if is_agent else "rag", is_agent,
            )

            full_response = ""
            try:
                if self._rate_limiter:
                    self._rate_limiter.check_and_consume("llm_calls", bot_id)
                async for chunk in bot_instance.astream(final_query, config=config):
                    full_response += chunk
                    yield chunk
            finally:
                # Close tracer root after streaming completes
                try:
                    if tracer:
                        tracer.end_root(outputs={"answer_preview": full_response[:300]})
                except Exception:
                    pass
                # Agent metadata injection
                try:
                    if config and not tracer:
                        from longtracer import LongTracer as LT
                        self._inject_trace_metadata(
                            LT.get_tracer(), bot_id, chat_id, "agent",
                        )
                except Exception:
                    pass
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
                if self._rate_limiter:
                    self._rate_limiter.check_and_consume("tool_calls", bot_id)
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

            prompt, doc_sources, raw_docs = assistant.get_answer(final_query, web_text)
            if self._rate_limiter:
                self._rate_limiter.check_and_consume("llm_calls", bot_id)
            vision = VisionBot(prompt_template=prompt, llm=self.llm)
            vision.create_vision_bot(image_paths)
            vision_response = vision.get_response(query)
            assistant.save_chat_history(query, vision_response)

            # ── Post-hoc tracing ──────────────────────────────────────────────
            if self.enable_tracer:
                try:
                    from longtracer import LongTracer as LT
                    tracer = LT.get_tracer()
                    if tracer:
                        tracer.start_root(inputs={
                            "bot_id": bot_id,
                            "chat_id": vision_chat_id,
                            "chat_type": "vision",
                            "query_preview": query[:300],
                            "image_count": len(image_paths),
                        })
                        # Retrieval span
                        with tracer.span("retrieval", run_type="retriever") as span:
                            span.set_output({
                                "count": len(raw_docs),
                                "sources": doc_sources,
                            })
                        # LLM span
                        with tracer.span("llm_call", run_type="llm") as span:
                            span.set_output({
                                "answer_preview": vision_response[:500],
                            })
                        # Verification (only if tracer_verify=True and sources exist)
                        if self.tracer_verify and raw_docs:
                            try:
                                from longtracer.guard.verifier import CitationVerifier
                                source_texts = [d.page_content for d in raw_docs]
                                result = CitationVerifier(
                                    threshold=self.tracer_threshold,
                                ).verify_parallel(vision_response, source_texts)
                                with tracer.span("grounding", run_type="chain") as span:
                                    span.set_output({
                                        "trust_score": result.trust_score,
                                        "verdict": result.verdict,
                                        "summary": result.summary,
                                        "hallucination_count": result.hallucination_count,
                                    })
                            except Exception as ve:
                                print(f"[WARN] Vision verification failed: {ve}")
                        tracer.end_root(outputs={"answer_preview": vision_response[:300]})
                except ImportError:
                    pass
                except Exception as te:
                    print(f"[WARN] Vision tracer failed: {te}")

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
