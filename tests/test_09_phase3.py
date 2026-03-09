"""Tests for P3-1 Structured Output, P3-2 GET /bots/{bot_id},
P3-3 Config Lock Guard, and P3-6 Vision Normalizer."""

import json

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from longtrainer.structured import (
    StructuredResponse,
    validate_structured_output,
    get_structured_response,
    compute_schema_hash,
    _truncate_error,
)


# ─── P3-1: validate_structured_output ─────────────────────────────────────────


TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"],
}


def test_validate_valid_output():
    """Valid JSON matching schema should return parsed dict."""
    raw = json.dumps({"answer": "Hello", "confidence": 0.95})
    result = validate_structured_output(raw, TEST_SCHEMA)
    assert result["answer"] == "Hello"
    assert result["confidence"] == 0.95


def test_validate_missing_field():
    """Missing required field should raise ValidationError."""
    import jsonschema
    raw = json.dumps({"answer": "Hello"})
    with pytest.raises(jsonschema.ValidationError):
        validate_structured_output(raw, TEST_SCHEMA)


def test_validate_invalid_json():
    """Non-JSON string should raise JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        validate_structured_output("not json at all", TEST_SCHEMA)


# ─── P3-1: StructuredResponse ─────────────────────────────────────────────────


def test_structured_response_success():
    """StructuredResponse should serialize correctly for success."""
    resp = StructuredResponse(status="success", data={"answer": "Hi", "confidence": 0.9})
    d = resp.model_dump()
    assert d["status"] == "success"
    assert d["data"]["answer"] == "Hi"
    assert d["error"] is None


def test_structured_response_partial():
    """StructuredResponse should serialize correctly for partial_success."""
    resp = StructuredResponse(
        status="partial_success",
        data=None,
        raw_llm_output="broken json",
        error="Validation failed",
    )
    d = resp.model_dump()
    assert d["status"] == "partial_success"
    assert d["data"] is None
    assert d["raw_llm_output"] == "broken json"


# ─── P3-1: get_structured_response (mocked LLM) ─────────────────────────────


def _make_mock_llm(responses):
    """Create a mock LLM that returns the given strings in sequence."""
    mock = MagicMock()
    from langchain_core.messages import AIMessage
    side_effects = [AIMessage(content=r) for r in responses]
    mock.invoke = MagicMock(side_effect=side_effects)
    return mock


def test_structured_response_first_attempt_success():
    """LLM returns valid JSON on first try → success."""
    valid_json = json.dumps({"answer": "Hello", "confidence": 0.95})
    mock_llm = _make_mock_llm([valid_json])
    result = get_structured_response(mock_llm, [], TEST_SCHEMA)
    assert result.status == "success"
    assert result.data["confidence"] == 0.95
    assert mock_llm.invoke.call_count == 1


def test_structured_response_retry_success():
    """LLM fails first try, succeeds on retry → success."""
    bad_json = json.dumps({"answer": "Hi"})  # missing 'confidence'
    good_json = json.dumps({"answer": "Hi", "confidence": 0.8})
    mock_llm = _make_mock_llm([bad_json, good_json])
    result = get_structured_response(mock_llm, [], TEST_SCHEMA)
    assert result.status == "success"
    assert result.data["confidence"] == 0.8
    assert mock_llm.invoke.call_count == 2


def test_structured_response_both_fail():
    """LLM fails both attempts → partial_success with raw output."""
    bad1 = json.dumps({"answer": "Hi"})
    bad2 = json.dumps({"answer": "Hello"})
    mock_llm = _make_mock_llm([bad1, bad2])
    result = get_structured_response(mock_llm, [], TEST_SCHEMA)
    assert result.status == "partial_success"
    assert result.data is None
    assert result.error is not None


def test_scratchpad_does_not_mutate_original():
    """The retry must NOT mutate the original messages list."""
    from langchain_core.messages import HumanMessage
    original = [HumanMessage(content="test")]
    original_len = len(original)

    bad = json.dumps({"answer": "Hi"})
    good = json.dumps({"answer": "Hi", "confidence": 0.8})
    mock_llm = _make_mock_llm([bad, good])

    get_structured_response(mock_llm, original, TEST_SCHEMA)
    assert len(original) == original_len, "Original messages list was mutated!"


# ─── P3-1: compute_schema_hash ────────────────────────────────────────────────


def test_schema_hash_deterministic():
    """Same schema content should always produce the same hash."""
    schema1 = {"type": "object", "properties": {"x": {"type": "number"}}}
    schema2 = {"properties": {"x": {"type": "number"}}, "type": "object"}
    assert compute_schema_hash(schema1) == compute_schema_hash(schema2)


def test_schema_hash_different():
    """Different schemas should produce different hashes."""
    s1 = {"type": "object", "properties": {"x": {"type": "number"}}}
    s2 = {"type": "object", "properties": {"y": {"type": "string"}}}
    assert compute_schema_hash(s1) != compute_schema_hash(s2)


# ─── P3-1: _truncate_error ───────────────────────────────────────────────────


def test_truncate_json_error():
    """JSON decode errors should include position info."""
    try:
        json.loads("{bad")
    except json.JSONDecodeError as e:
        msg = _truncate_error(e)
        assert "JSON parse error" in msg


def test_truncate_validation_error():
    """Validation errors should include field path."""
    import jsonschema
    try:
        jsonschema.validate({"x": 1}, {"type": "object", "required": ["y"]})
    except jsonschema.ValidationError as e:
        msg = _truncate_error(e)
        assert "failed validation" in msg


# ─── P3-6: Vision Normalizer ─────────────────────────────────────────────────


def test_normalize_openai_passthrough():
    """OpenAI should pass through image_url unchanged."""
    from longtrainer.vision_normalizer import normalize_image_messages
    messages = [{"content": [{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}]}]
    result = normalize_image_messages(messages, "openai")
    assert result[0]["content"][0]["type"] == "image_url"
    assert result[0]["content"][0]["image_url"]["url"] == "https://example.com/img.jpg"


def test_normalize_anthropic_base64():
    """Anthropic should convert to base64 format when bytes are available."""
    from longtrainer.vision_normalizer import normalize_image_messages
    test_bytes = b"\xff\xd8\xff"  # JPEG magic bytes (truncated)
    messages = [{"content": [{"type": "image_url", "image_url": {"url": "/tmp/test.jpg"}}]}]
    result = normalize_image_messages(
        messages, "anthropic",
        image_bytes_map={"/tmp/test.jpg": test_bytes},
    )
    assert result[0]["content"][0]["type"] == "image"
    assert result[0]["content"][0]["source"]["type"] == "base64"


def test_normalize_text_only_passthrough():
    """Messages without image content should pass through unchanged."""
    from longtrainer.vision_normalizer import normalize_image_messages
    messages = [{"content": "Hello, world!"}]
    result = normalize_image_messages(messages, "anthropic")
    assert result[0]["content"] == "Hello, world!"


# ─── P3-6: downscale_image ───────────────────────────────────────────────────


def test_downscale_small_image():
    """A small image should be processed without error."""
    from longtrainer.vision_normalizer import downscale_image
    from PIL import Image
    import io

    # Create a small test image
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()

    result = downscale_image(raw, max_dim=50)
    assert len(result) > 0
    # Verify it was actually resized
    result_img = Image.open(io.BytesIO(result))
    assert max(result_img.size) <= 50
