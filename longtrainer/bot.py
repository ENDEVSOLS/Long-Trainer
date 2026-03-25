"""Bot modules for LongTrainer V2.

Provides two bot implementations:
- RAGBot: LCEL-based RAG chain for simple retrieval Q&A (default)
- AgentBot: LangGraph-based agent with tool calling (optional, requires `longtrainer[agent]`)
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough


def _format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGBot:
    """LCEL-based RAG chain with streaming support.

    This is the default bot mode for simple retrieval-augmented Q&A
    without tool calling.

    Args:
        retriever: Document retriever (FAISS or ensemble).
        llm: Language model for generating responses.
        prompt: ChatPromptTemplate for the conversation.
        token_limit: Maximum token limit for conversation buffer (used for trimming).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        token_limit: int = 32000,
    ) -> None:
        try:
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
            self.token_limit = token_limit

            self.chat_history = InMemoryChatMessageHistory()
            self.chain = self._build_chain()
        except Exception as e:
            print(f"[ERROR] Error initializing RAGBot: {e}")

    def _build_chain(self):
        """Build the LCEL RAG chain."""

        def get_context(query: str) -> str:
            docs = self.retriever.invoke(query)
            return _format_docs(docs)

        chain = (
            RunnablePassthrough.assign(
                context=lambda x: get_context(x["question"]),
                chat_history=lambda _: self.chat_history.messages,
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def save_context(self, query: str, answer: str) -> None:
        """Save a query-answer pair to chat history.

        Args:
            query: The user's question.
            answer: The AI's response.
        """
        self.chat_history.add_message(HumanMessage(content=query))
        self.chat_history.add_message(AIMessage(content=answer))

    def invoke(self, query: str) -> str:
        """Get a complete response for a query.

        Args:
            query: The user's question.

        Returns:
            The assistant's response string.
        """
        try:
            result = self.chain.invoke({"question": query})
            self.save_context(query, result)
            return result
        except Exception as e:
            print(f"[ERROR] Error in RAGBot invoke: {e}")
            return ""

    def invoke_structured(self, query: str, schema: dict):
        """Single LLM call: retrieve docs → validate output against JSON schema.

        Correct message ordering for all LLM providers:
          [SystemMessage (prompt + context)] → [chat history H/A pairs] → [HumanMessage]
        SystemMessage is ALWAYS first — Anthropic/OpenAI reject SystemMessage mid-array.

        Includes a 1-retry self-correction loop when validation fails.

        Args:
            query: The user's question.
            schema: JSON Schema dict to validate the LLM response against.

        Returns:
            dict with keys: status, data, raw_llm_output, error.
        """
        import json
        import logging
        import jsonschema

        logger = logging.getLogger(__name__)

        def _validate(llm_output: str, schema: dict) -> dict:
            """Parse and validate LLM output against a JSON Schema."""
            cleaned = llm_output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            parsed = json.loads(cleaned)
            jsonschema.validate(instance=parsed, schema=schema)
            return parsed

        def _truncate_error(error: Exception) -> str:
            """Extract field path + error type — never the full broken payload."""
            if isinstance(error, jsonschema.ValidationError):
                path = " -> ".join(str(p) for p in error.absolute_path) or "root"
                return f"Field '{path}' failed validation: {error.message[:200]}"
            elif isinstance(error, json.JSONDecodeError):
                return f"JSON parse error at position {error.pos}: {error.msg}"
            return str(error)[:200]

        _partial = lambda msg, raw=None: {
            "status": "partial_success",
            "data": None,
            "raw_llm_output": raw,
            "error": msg,
        }

        try:
            # Step 1: Retrieve relevant docs (no LLM call — pure vector search)
            docs = self.retriever.invoke(query)
            context = _format_docs(docs)

            # Step 2: Build correctly ordered message array
            try:
                original_system = self.prompt.messages[0].prompt.template
            except (AttributeError, IndexError):
                original_system = "You are a helpful assistant."

            system_msg = SystemMessage(
                content=f"{original_system}\n\nRelevant context from knowledge base:\n{context}"
            )
            history_messages = [
                m for m in self.chat_history.messages
                if not isinstance(m, SystemMessage)
            ]

            schema_instruction = (
                f"You MUST respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                f"Return ONLY the JSON object, no markdown fences, no extra text."
            )
            messages = [system_msg] + history_messages + [
                SystemMessage(content=schema_instruction) if not history_messages else HumanMessage(content=query),
            ]
            # Correct: system prompt first, then history, then schema instruction merged into user msg
            messages = (
                [system_msg]
                + history_messages
                + [HumanMessage(content=f"{query}\n\n{schema_instruction}")]
            )

            # Step 3: Attempt 1 — single LLM call
            try:
                response = self.llm.invoke(messages)
                raw_output = response.content if isinstance(response, AIMessage) else str(response)
                parsed = _validate(raw_output, schema)
                self.save_context(query, str(parsed))
                return {"status": "success", "data": parsed, "raw_llm_output": None, "error": None}
            except (json.JSONDecodeError, jsonschema.ValidationError) as first_error:
                logger.warning("Structured output attempt 1 failed: %s", first_error)

                # Step 4: Attempt 2 — retry with error feedback (scratchpad, never mutates history)
                error_context = _truncate_error(first_error)
                retry_messages = list(messages) + [
                    HumanMessage(content=f"Your previous response was invalid. {error_context} Regenerate valid JSON only.")
                ]
                try:
                    retry_response = self.llm.invoke(retry_messages)
                    raw_retry = retry_response.content if isinstance(retry_response, AIMessage) else str(retry_response)
                    parsed = _validate(raw_retry, schema)
                    self.save_context(query, str(parsed))
                    return {"status": "success", "data": parsed, "raw_llm_output": None, "error": None}
                except (json.JSONDecodeError, jsonschema.ValidationError) as second_error:
                    logger.warning("Structured output attempt 2 failed: %s", second_error)
                    raw_final = raw_retry if "raw_retry" in dir() else raw_output
                    return _partial("The AI model failed to generate a strictly formatted response.", raw_final)

        except Exception as e:
            print(f"[ERROR] Error in RAGBot invoke_structured: {e}")
            return _partial(f"Unexpected error: {str(e)}")


    def stream(self, query: str) -> Iterator[str]:
        """Stream response tokens for a query.

        Args:
            query: The user's question.

        Yields:
            Response tokens as strings.
        """
        try:
            full_response = ""
            for chunk in self.chain.stream({"question": query}):
                full_response += chunk
                yield chunk
            self.save_context(query, full_response)
        except Exception as e:
            print(f"[ERROR] Error in RAGBot stream: {e}")

    async def astream(self, query: str) -> AsyncIterator[str]:
        """Async stream response tokens for a query.

        Args:
            query: The user's question.

        Yields:
            Response tokens as strings.
        """
        try:
            full_response = ""
            async for chunk in self.chain.astream({"question": query}):
                full_response += chunk
                yield chunk
            self.save_context(query, full_response)
        except Exception as e:
            print(f"[ERROR] Error in RAGBot astream: {e}")

    @property
    def memory(self):
        """Backward-compatible memory accessor (returns self for save_context calls)."""
        return self


class AgentBot:
    """LangGraph-based agent with tool calling and streaming.

    Requires the `longtrainer[agent]` extra: ``pip install longtrainer[agent]``

    Args:
        llm: Language model for the agent.
        tools: List of LangChain tools for the agent to use.
        system_prompt: System prompt string for the agent.
        token_limit: Maximum token limit for conversation buffer.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list,
        system_prompt: str,
        token_limit: int = 32000,
    ) -> None:
        try:
            try:
                from langgraph.prebuilt import create_react_agent
            except ImportError:
                raise ImportError(
                    "Agent mode requires the 'langgraph' package. "
                    "Install it with: pip install longtrainer[agent]"
                )

            self.llm = llm
            self.tools = tools
            self.system_prompt = system_prompt
            self.token_limit = token_limit

            self.chat_history = InMemoryChatMessageHistory()

            self.agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=SystemMessage(content=system_prompt),
            )
        except ImportError:
            raise
        except Exception as e:
            print(f"[ERROR] Error initializing AgentBot: {e}")

    def save_context(self, query: str, answer: str) -> None:
        """Save a query-answer pair to chat history.

        Args:
            query: The user's question.
            answer: The AI's response.
        """
        self.chat_history.add_message(HumanMessage(content=query))
        self.chat_history.add_message(AIMessage(content=answer))

    def invoke(self, query: str) -> str:
        """Get a complete response using the agent.

        Args:
            query: The user's question.

        Returns:
            The agent's final response string.
        """
        try:
            messages = list(self.chat_history.messages) + [
                HumanMessage(content=query),
            ]
            result = self.agent.invoke({"messages": messages})
            answer = result["messages"][-1].content if result.get("messages") else ""
            self.save_context(query, answer)
            return answer
        except Exception as e:
            print(f"[ERROR] Error in AgentBot invoke: {e}")
            return ""

    def stream(self, query: str) -> Iterator[str]:
        """Stream response tokens from the agent.

        Args:
            query: The user's question.

        Yields:
            Response tokens as strings.
        """
        try:
            messages = list(self.chat_history.messages) + [
                HumanMessage(content=query),
            ]
            full_response = ""
            for chunk in self.agent.stream(
                {"messages": messages},
                stream_mode="messages",
            ):
                if hasattr(chunk, "__iter__") and len(chunk) == 2:
                    msg, meta = chunk
                    if hasattr(msg, "content") and msg.content:
                        full_response += msg.content
                        yield msg.content
                elif hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            self.save_context(query, full_response)
        except Exception as e:
            print(f"[ERROR] Error in AgentBot stream: {e}")

    async def astream(self, query: str) -> AsyncIterator[str]:
        """Async stream response tokens from the agent.

        Args:
            query: The user's question.

        Yields:
            Response tokens as strings.
        """
        try:
            messages = list(self.chat_history.messages) + [
                HumanMessage(content=query),
            ]
            full_response = ""
            async for chunk in self.agent.astream(
                {"messages": messages},
                stream_mode="messages",
            ):
                if hasattr(chunk, "__iter__") and len(chunk) == 2:
                    msg, meta = chunk
                    if hasattr(msg, "content") and msg.content:
                        full_response += msg.content
                        yield msg.content
                elif hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            self.save_context(query, full_response)
        except Exception as e:
            print(f"[ERROR] Error in AgentBot astream: {e}")

    @property
    def memory(self):
        """Backward-compatible memory accessor."""
        return self
