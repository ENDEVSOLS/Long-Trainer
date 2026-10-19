"""Offline purpose selection, persistence and runtime compatibility."""
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from longtrainer import AgentTypeRegistry, LongTrainer
from longtrainer.agent_types import resolve_tools
from longtrainer.agent_types.base import AgentTypeConfig
from longtrainer.bot import RAGBot


@tool
def example_tool(query: str) -> str:
    """Return a deterministic test answer."""
    return query


class Retriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Support policy")]


@pytest.fixture
def trainer(monkeypatch):
    storage = MagicMock()
    storage.find_bot.return_value = {"db_path": "unused"}
    storage.get_chat_by_id.return_value = []
    monkeypatch.setattr("longtrainer.trainer.MongoStorage", lambda config: storage)
    store = MagicMock()
    store.as_retriever.return_value = Retriever()
    monkeypatch.setattr("longtrainer.trainer.get_vectorstore", lambda **kwargs: store)
    instance = LongTrainer(llm=FakeListChatModel(responses=["Supported answer"]),
                           embedding_model=FakeEmbeddings(size=8), ensemble=False)
    instance.load_bot("bot")
    instance._doc_manager = MagicMock()
    instance._doc_manager.get_documents.return_value = []
    return instance


def test_registry():
    assert AgentTypeRegistry.names() == ["coding", "customer_support", "financial", "research"]
    with pytest.raises(ValueError, match="Supported types:.*research"):
        AgentTypeRegistry.get("sql")
    config = AgentTypeConfig(name="test", system_prompt="Test {context}", mode="rag")
    try:
        AgentTypeRegistry.register(config)
        assert AgentTypeRegistry.get("test") == config
    finally:
        AgentTypeRegistry._configs.pop("test", None)


@pytest.mark.parametrize("name", ["research", "coding", "financial", "customer_support"])
def test_defaults_load_and_rebuild(trainer, name):
    config = AgentTypeRegistry.get(name)
    with patch("longtrainer.trainer.resolve_tools", return_value=[example_tool] if config.default_tools else []) as resolve:
        trainer.create_bot("bot", agent_type=name, agent_mode=True)
        assert resolve.call_args.args[0] == list(config.default_tools)
        bot = trainer.bot_data["bot"]
        assert bot["agent_mode"] == (config.mode == "agent")
        assert bot["prompt_template"] == config.system_prompt
        saved = trainer._storage.update_bot.call_args.args[1]
        assert saved["agent_type"] == name and saved["type_tools_override"] is False
        trainer._storage.find_bot.return_value = {"db_path": "unused", **saved}
        trainer.load_bot("bot")
        assert trainer.bot_data["bot"]["prompt_template"] == config.system_prompt
        assert trainer.bot_data["bot"]["agent_type"] == name
        trainer._rebuild_bot("bot")
        assert trainer.bot_data["bot"]["agent_mode"] == (config.mode == "agent")


@pytest.mark.parametrize("specs", [[], [example_tool], ["wikipedia"]])
def test_overrides_exclude_globals_and_previous_tools(trainer, specs):
    trainer._global_tools.register(example_tool)
    trainer.bot_data["bot"]["tools"].register(example_tool)
    with patch("longtrainer.tools.load_dynamic_tools", return_value=[example_tool]):
        trainer.create_bot("bot", agent_type="research", tools=specs, prompt_template="Custom {context}")
    assert trainer.bot_data["bot"]["prompt_template"] == "Custom {context}"
    assert trainer._get_bot_tools("bot") == ([example_tool] if specs else [])
    assert trainer.list_tools("bot") == ([example_tool.name] if specs else [])
    assert trainer._storage.update_bot.call_args.args[1]["type_tools_override"] is True
    trainer._rebuild_bot("bot")
    assert trainer.bot_data["bot"]["prompt_template"] == "Custom {context}"


def test_invalid_name_and_missing_default_fail_loudly(trainer):
    with pytest.raises(ValueError, match="Unknown agent type"):
        trainer.create_bot("bot", agent_type="invalid")
    with patch("longtrainer.tools.get_tavily_search_tool", return_value=None):
        with pytest.raises(ValueError, match="Cannot load tool 'tavily'"):
            trainer.create_bot("bot", agent_type="research")


@pytest.mark.parametrize("name", ["research", "coding", "financial"])
def test_new_and_lazy_agent_use_matching_tools(trainer, name):
    trainer._global_tools.register(example_tool)
    trainer.create_bot("bot", agent_type=name, tools=[])
    with patch("longtrainer.chat.AgentBot") as new, patch("longtrainer.trainer.AgentBot") as lazy:
        trainer.new_chat("bot")
        trainer._ensure_chat_loaded("bot", "saved-chat")
        assert new.call_args.kwargs == lazy.call_args.kwargs
        assert new.call_args.kwargs["tools"] == []


def test_support_rag_response_and_augmentation_policy(trainer):
    trainer.create_bot("bot", agent_type="customer_support")
    chat = trainer.new_chat("bot")
    assert isinstance(trainer.bot_data["bot"]["chains"][chat], RAGBot)
    assert trainer.get_response("Question", "bot", chat)[0] == "Supported answer"
    trainer._ensure_chat_loaded("bot", "old")
    assert isinstance(trainer.bot_data["bot"]["chains"]["old"], RAGBot)
    for kwargs in ({"web_search": True}, {"uploaded_files": [{"name": "file"}]}):
        with pytest.raises(ValueError, match="knowledge-base"):
            trainer.get_response("Question", "bot", chat, **kwargs)


def test_factories_and_loader_identifiers():
    with patch("longtrainer.tools.get_tavily_search_tool", return_value=example_tool) as tavily, \
         patch("longtrainer.tools.get_python_repl_tool", return_value=example_tool) as python, \
         patch("longtrainer.tools.get_yahoo_finance_tool", return_value=example_tool) as yahoo, \
         patch("longtrainer.tools.load_dynamic_tools", return_value=[example_tool]) as loader, \
         patch("importlib.import_module", return_value=MagicMock()):
        assert len(resolve_tools(["tavily", "python_repl", "yahoo_finance_news", "wikipedia", "arxiv"])) == 5
        tavily.assert_called_once()
        python.assert_called_once()
        yahoo.assert_called_once()
        assert [call.args[0] for call in loader.call_args_list] == [["wikipedia"], ["arxiv"]]


def test_optional_imports_stay_lazy():
    script = '''import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'langgraph', 'langchain_experimental', 'tavily', 'arxiv', 'yfinance'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
import longtrainer
assert len(longtrainer.AgentTypeRegistry.names()) == 4
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from longtrainer.bot import RAGBot
from longtrainer.chat import build_chat_prompt
class Retriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Policy")]
bot = RAGBot(Retriever(), FakeListChatModel(responses=["Local RAG answer"]), build_chat_prompt("Use {context}"))
assert bot.invoke("Question") == "Local RAG answer"
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_legacy_positional_mode_and_prompt(trainer):
    trainer.create_bot("bot", "Legacy {context}", True, [])
    assert trainer.bot_data["bot"]["agent_mode"] is True
    assert trainer.bot_data["bot"]["prompt_template"] == "Legacy {context}"
    assert not trainer.bot_data["bot"].get("agent_type")


@pytest.mark.parametrize("name", ["research", "coding", "financial"])
def test_real_agent_runtime_responds_offline(trainer, name):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from longtrainer.bot import AgentBot

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    llm = ToolModel(responses=[AIMessage(content="Verified local answer")])
    trainer.llm = llm
    trainer._chat_manager.llm = llm
    with patch("longtrainer.trainer.resolve_tools", return_value=[example_tool]):
        trainer.create_bot("bot", agent_type=name)
    chat = trainer.new_chat("bot")
    assert isinstance(trainer.bot_data["bot"]["chains"][chat], AgentBot)
    assert trainer.get_response("Question", "bot", chat)[0] == "Verified local answer"
    trainer._ensure_chat_loaded("bot", "reloaded-chat")
    assert trainer.get_response("Question", "bot", "reloaded-chat")[0] == "Verified local answer"


def test_empty_override_survives_reload(trainer):
    trainer.create_bot("bot", agent_type="research", tools=[], prompt_template="Custom")
    trainer._storage.find_bot.return_value = {"db_path": "unused", **trainer._storage.update_bot.call_args.args[1]}
    trainer.load_bot("bot")
    assert trainer._get_bot_tools("bot") == []
    assert trainer.bot_data["bot"]["type_tools_override"] is True
    trainer._rebuild_bot("bot")
    assert trainer._get_bot_tools("bot") == []
    assert trainer.bot_data["bot"]["prompt_template"] == "Custom"


def test_update_and_training_preserve_named_configuration(trainer):
    trainer.create_bot("bot", agent_type="customer_support", prompt_template="Custom {context}")
    trainer.update_chatbot([], "bot")
    assert trainer.bot_data["bot"]["agent_type"] == "customer_support"
    trainer._storage.find_untrained_chats.return_value = [{"_id": "one", "question": "q", "answer": "a"}]
    trainer._storage.export_chats_to_csv.return_value = "mocked.csv"
    result = trainer.train_chats("bot")
    assert result["csv_path"] == "mocked.csv"
    assert trainer.bot_data["bot"]["prompt_template"] == "Custom {context}"
    assert trainer.bot_data["bot"]["agent_mode"] is False


@pytest.mark.asyncio
async def test_support_async_augmentation_policy(trainer):
    trainer.create_bot("bot", agent_type="customer_support")
    with pytest.raises(ValueError, match="knowledge-base"):
        async for _ in trainer.aget_response("q", "bot", "chat", web_search=True):
            pass


def test_financial_missing_dependency_fails_at_construction():
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        with pytest.raises(ValueError, match="Cannot load tool 'yahoo_finance_news'"):
            resolve_tools(["yahoo_finance_news"])


def test_support_sources_reach_prompt_without_mutating_docs(trainer):
    trainer.create_bot("bot", agent_type="customer_support")
    docs = trainer.bot_data["bot"]["ensemble_retriever"].invoke("q")
    assert "[Document 1]" in docs[0].page_content
    assert "Support policy" in docs[0].page_content
    assert Retriever().invoke("q")[0].page_content == "Support policy"
    with pytest.raises(ValueError, match="vision augmentation"):
        trainer.get_vision_response("q", [], "bot", "vision")


@pytest.mark.parametrize("name", ["research", "customer_support"])
def test_named_chat_retains_rate_limiter_and_tracer_config(trainer, name):
    trainer.create_bot("bot", agent_type=name, tools=[])
    bot = MagicMock()
    bot.invoke.return_value = "answer"
    trainer.bot_data["bot"]["chains"]["chat"] = bot
    limiter = MagicMock()
    trainer._chat_manager._rate_limiter = limiter
    config = {"tags": ["trace-test"]}
    with patch.object(trainer._chat_manager, "_build_tracer_config", return_value=(config, None)) as trace:
        assert trainer.get_response("q", "bot", "chat")[0] == "answer"
    limiter.check_and_consume.assert_called_once_with("llm_calls", "bot")
    bot.invoke.assert_called_once_with("q", config=config)
    assert trace.call_args.args[-1] == (name == "research")


def test_cli_ingestion_keeps_named_bot(trainer, tmp_path):
    from click.testing import CliRunner
    from longtrainer.cli import cli

    trainer.create_bot("bot", agent_type="customer_support", tools=[], prompt_template="Custom {context}")
    trainer._storage.find_bot.return_value = {"db_path": "unused", **trainer._storage.update_bot.call_args.args[1]}
    document = tmp_path / "policy.txt"
    document.write_text("Policy")
    with patch("longtrainer.cli._get_trainer", return_value=trainer):
        result = CliRunner().invoke(cli, ["add-doc", "bot", str(document)])
    assert result.exit_code == 0, result.output
    assert trainer.bot_data["bot"]["agent_type"] == "customer_support"
    assert trainer.bot_data["bot"]["prompt_template"] == "Custom {context}"


def test_load_unknown_type_or_unavailable_tools_fails_visibly(trainer):
    trainer._storage.find_bot.return_value = {"agent_type": "missing"}
    with pytest.raises(ValueError, match="Unknown agent type"):
        trainer.load_bot("bot")
    trainer._storage.find_bot.return_value = {"agent_type": "research"}
    with patch("longtrainer.tools.get_tavily_search_tool", return_value=None):
        with pytest.raises(ValueError, match="Cannot load tool"):
            trainer.load_bot("bot")


def test_missing_agent_runtime_fails_at_configuration(trainer):
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        with pytest.raises(ValueError, match="requires LangGraph"):
            trainer.create_bot("bot", agent_type="research", tools=[])
