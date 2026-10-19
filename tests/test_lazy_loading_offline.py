"""Lazy-loading regressions with real bot memory and mocked external services."""

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from longtrainer.bot import RAGBot
from longtrainer.tools import get_builtin_tools
from longtrainer.trainer import LongTrainer
from longtrainer.vision_bot import VisionMemory


class _Retriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Stored policy context.")]


@pytest.fixture
def trainer(monkeypatch):
    """Construct real managers using local model/embeddings and mocked persistence."""
    storage = MagicMock()
    storage.find_bot.return_value = {
        "bot_id": "bot-test",
        "db_path": "db_bot-test",
        "prompt_template": "Answer from these documents: {context}",
        "agent_mode": False,
    }
    storage.get_chat_by_id.return_value = []
    storage.get_vision_chat_by_id.return_value = []
    monkeypatch.setattr("longtrainer.trainer.MongoStorage", lambda config: storage)
    vectorstore = MagicMock()
    vectorstore.as_retriever.return_value = _Retriever()
    monkeypatch.setattr("longtrainer.trainer.get_vectorstore", lambda **kwargs: vectorstore)
    instance = LongTrainer(
        llm=FakeListChatModel(responses=["Test answer"]),
        embedding_model=FakeEmbeddings(size=8),
    )
    return instance


def test_load_bot_defers_all_chat_history(trainer):
    trainer.load_bot("bot-test")

    bot = trainer.bot_data["bot-test"]
    assert bot["chains"] == {}
    assert bot["assistants"] == {}
    assert bot["vectorstore"] is not None
    assert bot["retriever"] is not None
    trainer._storage.get_chat_by_id.assert_not_called()
    trainer._storage.get_vision_chat_by_id.assert_not_called()


def test_load_bot_restores_legacy_path_and_dynamic_tools(trainer):
    config = trainer._storage.find_bot.return_value
    config.pop("db_path")
    config["faiss_path"] = "legacy-index"
    config["agent_mode"] = True
    config["dynamic_tools"] = ["wikipedia"]
    tool = get_builtin_tools()[0]

    with patch("longtrainer.tools.load_dynamic_tools", return_value=[tool]) as load_tools:
        trainer.load_bot("bot-test")

    bot = trainer.bot_data["bot-test"]
    assert bot["db_path"] == "legacy-index"
    assert bot["agent_mode"] is True
    assert bot["tools"].get_tools() == [tool]
    load_tools.assert_called_once_with(["wikipedia"])


def test_lazy_rag_chat_replays_only_requested_history(trainer):
    trainer.load_bot("bot-test")
    trainer._storage.get_chat_by_id.return_value = [
        {"question": "First question", "answer": "First answer"},
        {"question": "Second question", "answer": "Second answer"},
    ]

    trainer._ensure_chat_loaded("bot-test", "chat-test")

    chains = trainer.bot_data["bot-test"]["chains"]
    assert list(chains) == ["chat-test"]
    assert isinstance(chains["chat-test"], RAGBot)
    assert [m.content for m in chains["chat-test"].chat_history.messages] == [
        "First question", "First answer", "Second question", "Second answer",
    ]
    trainer._storage.get_chat_by_id.assert_called_once_with("chat-test", "oldest")


def test_cached_chat_is_not_rebuilt_or_replayed(trainer):
    trainer.load_bot("bot-test")
    trainer._storage.get_chat_by_id.return_value = [{"question": "Hello", "answer": "Hi"}]
    trainer._ensure_chat_loaded("bot-test", "chat-test")
    cached = trainer.bot_data["bot-test"]["chains"]["chat-test"]

    trainer._ensure_chat_loaded("bot-test", "chat-test")

    assert trainer.bot_data["bot-test"]["chains"]["chat-test"] is cached
    assert len(cached.chat_history.messages) == 2
    trainer._storage.get_chat_by_id.assert_called_once()


def test_chat_without_history_starts_empty(trainer):
    trainer.load_bot("bot-test")
    trainer._storage.get_chat_by_id.return_value = None

    trainer._ensure_chat_loaded("bot-test", "fresh-chat")

    assert trainer.bot_data["bot-test"]["chains"]["fresh-chat"].chat_history.messages == []


def test_lazy_agent_chat_receives_tools_and_replays_history(trainer):
    trainer._storage.find_bot.return_value["agent_mode"] = True
    trainer.load_bot("bot-test")
    global_tool, bot_tool = get_builtin_tools()
    trainer.add_tool(global_tool)
    trainer.add_tool(bot_tool, "bot-test")
    trainer._storage.get_chat_by_id.return_value = [{"question": "Hello", "answer": "Hi"}]

    with patch("longtrainer.trainer.AgentBot") as agent_class:
        trainer._ensure_chat_loaded("bot-test", "agent-chat")

    agent_class.assert_called_once_with(
        llm=trainer.llm,
        tools=[global_tool, bot_tool],
        system_prompt=trainer.bot_data["bot-test"]["prompt_template"],
        token_limit=trainer.max_token_limit,
    )
    agent_class.return_value.save_context.assert_called_once_with("Hello", "Hi")
    assert trainer.bot_data["bot-test"]["chains"]["agent-chat"] is agent_class.return_value


def test_lazy_vision_chat_replays_history_and_uses_cache(trainer):
    trainer.load_bot("bot-test")
    trainer._storage.get_vision_chat_by_id.return_value = [
        {"question": "Describe image", "response": "An image description"},
    ]

    trainer._ensure_vision_chat_loaded("bot-test", "vision-test")
    cached = trainer.bot_data["bot-test"]["assistants"]["vision-test"]
    trainer._ensure_vision_chat_loaded("bot-test", "vision-test")

    assert isinstance(cached, VisionMemory)
    assert trainer.bot_data["bot-test"]["assistants"]["vision-test"] is cached
    assert [m.content for m in cached.chat_history_store.messages] == [
        "Describe image", "An image description",
    ]
    trainer._storage.get_vision_chat_by_id.assert_called_once_with("vision-test", "oldest")


def test_vision_chat_without_history_starts_empty(trainer):
    trainer.load_bot("bot-test")

    trainer._ensure_vision_chat_loaded("bot-test", "fresh-vision")

    assert trainer.bot_data["bot-test"]["assistants"]["fresh-vision"].chat_history_store.messages == []


@pytest.mark.parametrize("method", ["_ensure_chat_loaded", "_ensure_vision_chat_loaded"])
def test_lazy_load_rejects_unknown_bot(trainer, method):
    with pytest.raises(ValueError, match="not found"):
        getattr(trainer, method)("unknown-bot", "chat")
    trainer._storage.get_chat_by_id.assert_not_called()
    trainer._storage.get_vision_chat_by_id.assert_not_called()


def test_sync_response_lazy_loads_then_saves_new_exchange(trainer):
    trainer.load_bot("bot-test")
    trainer._storage.get_chat_by_id.return_value = [{"question": "Earlier", "answer": "Earlier answer"}]

    answer, sources = trainer.get_response("New question", "bot-test", "chat-test")

    assert answer == "Test answer"
    assert sources == []
    history = trainer.bot_data["bot-test"]["chains"]["chat-test"].chat_history.messages
    assert [m.content for m in history] == ["Earlier", "Earlier answer", "New question", "Test answer"]
    trainer._storage.get_chat_by_id.assert_called_once_with("chat-test", "oldest")
    trainer._storage.store_chat.assert_called_once()


async def test_async_response_lazy_loads_chat(trainer):
    trainer.load_bot("bot-test")

    chunks = [chunk async for chunk in trainer.aget_response("Question", "bot-test", "chat-test")]

    assert "".join(chunks) == "Test answer"
    trainer._storage.get_chat_by_id.assert_called_once_with("chat-test", "oldest")
    assert len(trainer.bot_data["bot-test"]["chains"]["chat-test"].chat_history.messages) == 2


def test_vision_response_loads_memory_before_delegating(trainer):
    trainer.load_bot("bot-test")

    def respond(*args):
        assert "vision-test" in trainer.bot_data["bot-test"]["assistants"]
        return "Vision answer", []

    with patch.object(trainer._chat_manager, "get_vision_response", side_effect=respond):
        assert trainer.get_vision_response("Question", [], "bot-test", "vision-test") == ("Vision answer", [])
    trainer._storage.get_vision_chat_by_id.assert_called_once_with("vision-test", "oldest")


def test_new_chats_after_reload_start_with_empty_histories(trainer):
    trainer.load_bot("bot-test")
    chat_id = trainer.new_chat("bot-test")
    vision_id = trainer.new_vision_chat("bot-test")

    assert chat_id
    assert vision_id
    assert trainer.bot_data["bot-test"]["chains"][chat_id].chat_history.messages == []
    assert trainer.bot_data["bot-test"]["assistants"][vision_id].chat_history_store.messages == []
    trainer._storage.get_chat_by_id.assert_not_called()
    trainer._storage.get_vision_chat_by_id.assert_not_called()


def test_reloaded_bot_can_fetch_history_and_update_prompt(trainer):
    trainer.load_bot("bot-test")
    trainer.list_chats("bot-test")
    trainer.get_chat_by_id("chat-test")
    trainer.get_vision_chat_by_id("vision-test")
    trainer.set_custom_prompt_template("bot-test", "Updated context: {context}")

    trainer._storage.list_chats.assert_called_once_with("bot-test")
    trainer._storage.get_chat_by_id.assert_called_once_with("chat-test", "newest")
    trainer._storage.get_vision_chat_by_id.assert_called_once_with("vision-test", "newest")
    assert trainer.bot_data["bot-test"]["prompt_template"] == "Updated context: {context}"
    trainer._storage.update_bot.assert_called_once_with(
        "bot-test", {"prompt_template": "Updated context: {context}"},
    )


def test_reload_discards_cached_sessions_and_replays_on_demand(trainer):
    trainer.load_bot("bot-test")
    trainer._ensure_chat_loaded("bot-test", "chat-test")
    previous = trainer.bot_data["bot-test"]["chains"]["chat-test"]
    trainer._storage.get_chat_by_id.reset_mock()

    trainer.load_bot("bot-test")

    assert trainer.bot_data["bot-test"]["chains"] == {}
    trainer._storage.get_chat_by_id.assert_not_called()
    trainer._ensure_chat_loaded("bot-test", "chat-test")
    assert trainer.bot_data["bot-test"]["chains"]["chat-test"] is not previous
    assert trainer._storage.get_chat_by_id.call_args_list == [call("chat-test", "oldest")]
