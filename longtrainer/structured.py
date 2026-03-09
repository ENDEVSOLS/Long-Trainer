"""Structured Output for LongTrainer — P3-1.

Provides JSON Schema validation on LLM outputs with a 1-retry
self-correction loop using a scratchpad clone (never mutates real history).

Two paths:
- SDK: Static Pydantic class → `llm.with_structured_output(schema)` wrapped
  in try/except OutputParserException → returns StructuredResponse.
- API: JSON Schema dict → LLM tools/response_format → jsonschema.validate().
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import jsonschema
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─── Response Model ───────────────────────────────────────────────────────────


class StructuredResponse(BaseModel):
    """Unified response shape for both SDK and REST API paths.

    - status: "success" | "partial_success"
    - data: The validated parsed output (None on failure)
    - raw_llm_output: The raw LLM text when validation fails
    - error: Human-readable error message (None on success)
    """

    status: str  # "success" | "partial_success"
    data: Optional[Any] = None
    raw_llm_output: Optional[str] = None
    error: Optional[str] = None


# ─── Validation ───────────────────────────────────────────────────────────────


def validate_structured_output(llm_output: str, schema: dict) -> dict:
    """Parse and validate LLM output against a JSON Schema.

    Args:
        llm_output: Raw string from the LLM (expected to be JSON).
        schema: A valid JSON Schema dict.

    Returns:
        The parsed dict if validation succeeds.

    Raises:
        jsonschema.ValidationError: If the output doesn't match the schema.
        json.JSONDecodeError: If the output is not valid JSON.
    """
    cleaned_output = llm_output.strip()
    if cleaned_output.startswith("```"):
        # Strip the first line (e.g., "```json")
        cleaned_output = cleaned_output.split("\n", 1)[-1]
        # Strip the trailing markdown block
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()

    parsed = json.loads(cleaned_output)
    jsonschema.validate(instance=parsed, schema=schema)
    return parsed


# ─── Self-Correction Loop ────────────────────────────────────────────────────


def get_structured_response(
    llm: BaseChatModel,
    messages: list,
    schema: dict,
) -> StructuredResponse:
    """Call LLM and validate output against schema with 1-retry self-correction.

    Uses a scratchpad clone for the retry — NEVER mutates the real message
    history. On success, only the valid AIMessage is persisted.

    Args:
        llm: A LangChain ChatModel instance.
        messages: The conversation messages (will NOT be mutated).
        schema: JSON Schema dict to validate against.

    Returns:
        StructuredResponse with status="success" or "partial_success".
    """
    schema_instruction = (
        f"You MUST respond with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        f"Return ONLY the JSON object, no markdown fences, no extra text."
    )

    # Inject schema instruction as a system message
    augmented_messages = list(messages) + [SystemMessage(content=schema_instruction)]

    # ── Attempt 1 ──
    try:
        response = llm.invoke(augmented_messages)
        raw_output = response.content if isinstance(response, AIMessage) else str(response)
        parsed = validate_structured_output(raw_output, schema)
        return StructuredResponse(status="success", data=parsed)
    except (json.JSONDecodeError, jsonschema.ValidationError) as first_error:
        logger.warning("Structured output attempt 1 failed: %s", first_error)

        # ── Attempt 2: Scratchpad clone with truncated error context ──
        # Extract only the error path — never feed the full broken output back
        error_context = _truncate_error(first_error)

        retry_messages = list(augmented_messages)  # shallow copy = scratchpad
        retry_messages.append(
            SystemMessage(content=f"Your previous response was invalid. {error_context} Regenerate valid JSON only.")
        )

        try:
            retry_response = llm.invoke(retry_messages)
            raw_retry = retry_response.content if isinstance(retry_response, AIMessage) else str(retry_response)
            parsed = validate_structured_output(raw_retry, schema)
            return StructuredResponse(status="success", data=parsed)
        except (json.JSONDecodeError, jsonschema.ValidationError) as second_error:
            logger.warning("Structured output attempt 2 failed: %s", second_error)
            raw_final = raw_retry if "raw_retry" in dir() else raw_output
            return StructuredResponse(
                status="partial_success",
                data=None,
                raw_llm_output=raw_final,
                error="The AI model failed to generate a strictly formatted response.",
            )
    except Exception as e:
        logger.error("Unexpected error in structured output: %s", e)
        return StructuredResponse(
            status="partial_success",
            data=None,
            raw_llm_output=None,
            error=f"Unexpected error: {str(e)}",
        )


# ─── SDK Wrapper ──────────────────────────────────────────────────────────────


def get_structured_response_sdk(
    llm: BaseChatModel,
    messages: list,
    pydantic_schema,
) -> StructuredResponse:
    """SDK path: Uses LangChain's with_structured_output() with OutputParserException safety.

    Wraps the raw LangChain method so SDK callers never see an uncaught
    OutputParserException crash.

    Args:
        llm: A LangChain ChatModel instance.
        messages: The conversation messages.
        pydantic_schema: A Pydantic BaseModel class for structured output.

    Returns:
        StructuredResponse with the parsed Pydantic model in .data or partial_success.
    """
    try:
        from langchain_core.exceptions import OutputParserException
    except ImportError:
        OutputParserException = Exception  # Fallback for older LangChain

    try:
        structured_llm = llm.with_structured_output(pydantic_schema)
        result = structured_llm.invoke(messages)
        return StructuredResponse(
            status="success",
            data=result.model_dump() if hasattr(result, "model_dump") else result,
        )
    except OutputParserException as e:
        logger.warning("SDK structured output failed: %s", e)
        return StructuredResponse(
            status="partial_success",
            data=None,
            raw_llm_output=str(e),
            error="The AI model failed to generate a strictly formatted response.",
        )
    except Exception as e:
        logger.error("Unexpected SDK structured output error: %s", e)
        return StructuredResponse(
            status="partial_success",
            data=None,
            raw_llm_output=None,
            error=f"Unexpected error: {str(e)}",
        )


# ─── Schema Registry Helpers ─────────────────────────────────────────────────


def compute_schema_hash(schema: dict) -> str:
    """Compute a deterministic SHA-256 hash for a JSON schema."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─── Internal Helpers ─────────────────────────────────────────────────────────


def _truncate_error(error: Exception) -> str:
    """Extract only the field path and error type — never the full broken payload.

    Prevents context doubling when the error message is fed back to the LLM.
    """
    if isinstance(error, jsonschema.ValidationError):
        path = " -> ".join(str(p) for p in error.absolute_path) or "root"
        return f"Field '{path}' failed validation: {error.message[:200]}"
    elif isinstance(error, json.JSONDecodeError):
        return f"JSON parse error at position {error.pos}: {error.msg}"
    return str(error)[:200]
