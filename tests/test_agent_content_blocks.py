"""Provider content blocks must preserve the public string response contract."""
from unittest.mock import MagicMock

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, AIMessageChunk

from longtrainer.bot import AgentBot


def make_agent():
    bot = AgentBot.__new__(AgentBot)
    bot.agent = MagicMock()
    bot.chat_history = InMemoryChatMessageHistory()
    return bot


def test_agent_invoke_extracts_text_blocks():
    bot = make_agent()
    bot.agent.invoke.return_value = {"messages": [AIMessage(content=[
        {"type": "text", "text": "The answer is 437."},
    ])]}
    assert bot.invoke("Calculate") == "The answer is 437."
    assert bot.chat_history.messages[-1].content == "The answer is 437."


def test_agent_stream_extracts_text_blocks():
    bot = make_agent()
    bot.agent.stream.return_value = [
        (AIMessageChunk(content=[{"type": "text", "text": "The answer "}]), {}),
        (AIMessageChunk(content=[{"type": "text", "text": "is 437."}]), {}),
    ]
    assert "".join(bot.stream("Calculate")) == "The answer is 437."
    assert bot.chat_history.messages[-1].content == "The answer is 437."


@pytest.mark.asyncio
async def test_agent_async_stream_extracts_text_blocks():
    bot = make_agent()

    async def chunks(*args, **kwargs):
        yield (AIMessageChunk(content=[{"type": "text", "text": "437"}]), {})

    bot.agent.astream = chunks
    assert "".join([part async for part in bot.astream("Calculate")]) == "437"
    assert bot.chat_history.messages[-1].content == "437"
