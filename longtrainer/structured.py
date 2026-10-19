"""JSON-schema validation and bounded response repair for structured chats."""

import hashlib
import json
import logging
from typing import Any, Literal, Optional

import jsonschema
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StructuredResponse(BaseModel):
    """Internal response model; bot callers receive its dictionary representation."""

    status: Literal["success", "partial_success"] = "partial_success"
    data: Any = None
    raw_llm_output: Optional[str] = None
    error: Optional[str] = None


def validate_structured_output(raw_output: str, schema: dict) -> Any:
    """Parse JSON, tolerating markdown fences, and validate against the schema."""
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    parsed = json.loads(cleaned)
    jsonschema.validate(instance=parsed, schema=schema)
    return parsed


def compute_schema_hash(schema: dict) -> str:
    """Return a deterministic SHA-256 digest independent of dictionary key order."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _truncate_error(error: Exception) -> str:
    """Describe the failing field or JSON position without dumping the payload."""
    if isinstance(error, jsonschema.ValidationError):
        path = " -> ".join(str(part) for part in error.absolute_path) or "root"
        return f"Field '{path}' failed validation: {error.message[:200]}"
    if isinstance(error, json.JSONDecodeError):
        return f"JSON parse error at position {error.pos}: {error.msg}"
    return str(error)[:200]


def log_unexpected_error(error: Exception) -> None:
    """Log a structured-call failure using this module's logger."""
    logger.error("Unexpected structured output error: %s", _truncate_error(error))


def get_structured_response(llm: Any, messages: list, schema: dict,
                            config: Optional[dict] = None) -> StructuredResponse:
    """Validate a response, retrying invalid output once on a copied message list.

    Transport and other unexpected failures return the legacy partial response
    without retrying. Successful responses omit raw output; final validation
    failures retain the second output. Caller messages and history are untouched.
    """
    scratchpad = list(messages)
    try:
        for attempt in range(2):
            response = llm.invoke(scratchpad, config=config) if config is not None else llm.invoke(scratchpad)
            raw_output = StrOutputParser().invoke(response) if isinstance(response, AIMessage) else str(response)
            try:
                parsed = validate_structured_output(raw_output, schema)
                return StructuredResponse(status="success", data=parsed)
            except (json.JSONDecodeError, jsonschema.ValidationError) as error:
                feedback = _truncate_error(error)
                logger.warning("Structured output attempt %s failed: %s", attempt + 1, feedback)
                if attempt == 1:
                    return StructuredResponse(
                        raw_llm_output=raw_output,
                        error="The AI model failed to generate a strictly formatted response.",
                    )
                scratchpad = scratchpad + [HumanMessage(
                    content=f"Your previous response was invalid. {feedback} Regenerate valid JSON only."
                )]
    except Exception as error:
        log_unexpected_error(error)
        return StructuredResponse(error=f"Unexpected error: {error}")
