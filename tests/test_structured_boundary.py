"""Compatibility checks for structured bot, chat and API boundaries."""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from longtrainer.bot import RAGBot
from longtrainer.chat import ChatManager, build_chat_prompt
from longtrainer.structured import get_structured_response, validate_structured_output

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
VALID = {"answer": "Hello"}


def make_bot(outputs):
    bot = RAGBot.__new__(RAGBot)
    bot.llm = MagicMock()
    bot.llm.invoke.side_effect = outputs
    bot.retriever = MagicMock()
    bot.retriever.invoke.return_value = [Document(page_content="Knowledge")]
    bot.prompt = build_chat_prompt("Helpful assistant {context}")
    bot.chat_history = InMemoryChatMessageHistory()
    return bot


@pytest.mark.parametrize("outputs,success", [
    ([AIMessage(content=json.dumps(VALID))], True),
    ([AIMessage(content="{}"), AIMessage(content=json.dumps(VALID))], True),
    ([AIMessage(content="{}"), AIMessage(content="broken")], False),
    ([RuntimeError("offline")], False),
    ([AIMessage(content="{}"), RuntimeError("offline")], False),
])
def test_bot_contract_and_history(outputs, success):
    bot = make_bot(outputs)
    history = [SystemMessage(content="old system"), HumanMessage(content="old query"), AIMessage(content="old answer")]
    bot.chat_history.add_messages(history)
    config = {"tags": ["structured"]}
    result = bot.invoke_structured("Question", SCHEMA, config=config)
    assert set(result) == {"status", "data", "raw_llm_output", "error"}
    assert result["status"] == ("success" if success else "partial_success")
    assert result["data"] == (VALID if success else None)
    assert bot.chat_history.messages[:3] == history
    assert len(bot.chat_history.messages) == (5 if success else 3)
    if success:
        assert bot.chat_history.messages[-2:] == [HumanMessage(content="Question"), AIMessage(content=str(VALID))]
        assert result["raw_llm_output"] is None and result["error"] is None
    elif isinstance(outputs[-1], AIMessage):
        assert result["raw_llm_output"] == "broken"
        assert result["error"] == "The AI model failed to generate a strictly formatted response."
    else:
        assert result["raw_llm_output"] is None
        assert result["error"] == "Unexpected error: offline"
    first = bot.llm.invoke.call_args_list[0].args[0]
    assert isinstance(first[0], SystemMessage)
    assert "Knowledge" in first[0].content
    assert first[1:3] == history[1:]
    assert isinstance(first[-1], HumanMessage) and "Question" in first[-1].content
    assert len(first) == 4
    for call in bot.llm.invoke.call_args_list:
        assert call.kwargs == {"config": config}
    if len(outputs) == 2:
        retry = bot.llm.invoke.call_args_list[1].args[0]
        assert retry[:-1] == first
        assert "previous response was invalid" in retry[-1].content


def test_retrieval_failure_returns_legacy_partial():
    bot = make_bot([])
    bot.retriever.invoke.side_effect = RuntimeError("retrieval unavailable")
    result = bot.invoke_structured("Question", SCHEMA)
    assert result == {"status": "partial_success", "data": None, "raw_llm_output": None,
                      "error": "Unexpected error: retrieval unavailable"}
    bot.llm.invoke.assert_not_called()
    assert bot.chat_history.messages == []


def test_fences_and_non_object_json():
    assert validate_structured_output('```json\n{"answer":"Hello"}\n```', SCHEMA) == VALID
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='[1,2]')
    result = get_structured_response(llm, [], {"type": "array"})
    assert result.data == [1, 2]


def test_chat_structured_path_persists_legacy_answer():
    bot = make_bot([AIMessage(content=json.dumps(VALID))])
    storage = MagicMock()
    manager = ChatManager(storage, bot.llm)
    result, sources = manager.get_response("Question", "bot", "chat", {"chains": {"chat": bot}}, schema=SCHEMA)
    assert result == {"status": "success", "data": VALID, "raw_llm_output": None, "error": None}
    assert sources == []
    storage.store_chat.assert_called_once_with(bot_id="bot", chat_id="chat", query="Question",
                                              answer=str(VALID), web_source=[], uploaded_files=None)


def test_structured_content_blocks_ignore_non_text_metadata():
    response = AIMessage(content=[
        {"type": "text", "text": '{"answer":'},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": '"Hello"}'},
    ])
    bot = make_bot([response])
    result = bot.invoke_structured("Question", SCHEMA)
    assert result["status"] == "success"
    assert result["data"] == VALID
