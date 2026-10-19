"""API/CLI purpose selection, compatibility and configuration boundaries."""
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from longtrainer.api import app
from longtrainer.cli import cli


@pytest.fixture
def trainer():
    instance = MagicMock()
    instance.initialize_bot_id.return_value = "new-bot"
    instance.bot_data = {"existing": {"agent_type": "customer_support", "prompt_template": "Stored {context}"}}
    return instance


@pytest.mark.parametrize("name", ["research", "coding", "financial", "customer_support"])
def test_api_purpose_forwarding(trainer, name):
    with patch("longtrainer.api._get_trainer", return_value=trainer):
        response = TestClient(app).post('/bots/existing/build', json={"agent_type": name, "tools": [],
                                                                    "prompt_template": "Custom", "agent_mode": True})
    assert response.status_code == 200
    trainer.create_bot.assert_called_once_with(bot_id="existing", agent_type=name, tools=[],
                                              prompt_template="Custom", agent_mode=True)


def test_api_invalid_purpose_before_storage_access(trainer):
    with patch("longtrainer.api._get_trainer", return_value=trainer) as get:
        response = TestClient(app).post('/bots/existing/build', json={"agent_type": "unknown"})
    assert response.status_code == 400
    assert all(name in response.json()['detail'] for name in ('research', 'coding', 'financial', 'customer_support'))
    get.assert_not_called()


def test_api_legacy_payload_and_missing_dependency(trainer):
    with patch("longtrainer.api._get_trainer", return_value=trainer):
        assert TestClient(app).post('/bots/existing/build', json={}).status_code == 200
        trainer.create_bot.assert_called_once_with(bot_id="existing", tools=None, prompt_template=None, agent_mode=False)
        trainer.create_bot.side_effect = ValueError("Required tool unavailable")
        response = TestClient(app).post('/bots/existing/build', json={"agent_type": "research"})
    assert response.status_code == 400
    assert response.json()['detail'] == 'Required tool unavailable'


@pytest.mark.parametrize("command", [["build"], ["bot", "create"]])
@pytest.mark.parametrize("tools,expected", [(None, None), ("", []), ("wikipedia, arxiv", ["wikipedia", "arxiv"])])
def test_cli_tools_and_purpose(trainer, command, tools, expected):
    args = command + ['--agent-type', 'research', '--prompt', 'Custom']
    if tools is not None:
        args += ['--tools', tools]
    with patch('longtrainer.cli._get_trainer', return_value=trainer):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    trainer.create_bot.assert_called_once_with(bot_id='new-bot', agent_type='research', tools=expected,
                                              prompt_template='Custom', agent_mode=False)


@pytest.mark.parametrize("command", [["build"], ["bot", "create"]])
def test_cli_invalid_type_no_allocation(trainer, command):
    with patch('longtrainer.cli._get_trainer', return_value=trainer) as get:
        result = CliRunner().invoke(cli, command + ['--agent-type', 'unknown'])
    assert result.exit_code != 0
    assert 'Supported types:' in result.output
    get.assert_not_called()


def test_cli_existing_rebuild_and_override(trainer):
    with patch('longtrainer.cli._get_trainer', return_value=trainer):
        result = CliRunner().invoke(cli, ['build', 'existing'])
        assert result.exit_code == 0, result.output
        trainer._rebuild_bot.assert_called_once_with('existing', prompt_template=None)
        result = CliRunner().invoke(cli, ['build', 'existing', '--tools', ''])
    assert result.exit_code == 0, result.output
    trainer.create_bot.assert_called_once_with(bot_id='existing', agent_type='customer_support', tools=[],
                                              prompt_template='Stored {context}', agent_mode=False)


def test_cli_legacy_create(trainer):
    with patch('longtrainer.cli._get_trainer', return_value=trainer):
        result = CliRunner().invoke(cli, ['bot', 'create', '--agent', '--tools', 'wikipedia'])
    assert result.exit_code == 0, result.output
    trainer.create_bot.assert_called_once_with(bot_id='new-bot', tools=['wikipedia'], prompt_template=None, agent_mode=True)


@pytest.mark.parametrize("command", [["build"], ["bot", "create"]])
def test_cli_failed_creation_is_visible(trainer, command):
    trainer.create_bot.side_effect = ValueError('Tool dependency missing')
    with patch('longtrainer.cli._get_trainer', return_value=trainer):
        result = CliRunner().invoke(cli, command + ['--agent-type', 'research'])
    assert result.exit_code != 0
    assert 'Tool dependency missing' in result.output
    assert 'successfully' not in result.output


def test_yaml_excludes_purpose_configuration(tmp_path):
    from longtrainer.cli import _get_trainer
    config = tmp_path / 'config.yaml'
    config.write_text('mongo_endpoint: mongodb://localhost:27017/\nagent_type: research\ntools: [python_repl]\n')
    with patch('longtrainer.trainer.LongTrainer') as constructor:
        _get_trainer(str(config))
    assert 'agent_type' not in constructor.call_args.kwargs
    assert 'tools' not in constructor.call_args.kwargs


@pytest.mark.parametrize('surface', ['api', 'build', 'create'])
@pytest.mark.parametrize('name', ['research', 'customer_support'])
def test_surfaces_select_real_runtime(monkeypatch, surface, name):
    from langchain_core.documents import Document
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.retrievers import BaseRetriever
    from longtrainer import LongTrainer
    from longtrainer.bot import RAGBot, AgentBot

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class Retriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return [Document(page_content='Policy')]

    storage = MagicMock()
    storage.find_bot.return_value = {'db_path': 'unused'}
    store = MagicMock()
    store.as_retriever.return_value = Retriever()
    monkeypatch.setattr('longtrainer.trainer.MongoStorage', lambda config: storage)
    monkeypatch.setattr('longtrainer.trainer.get_vectorstore', lambda **kwargs: store)
    instance = LongTrainer(llm=Model(responses=[AIMessage(content='Local answer')]), embedding_model=FakeEmbeddings(size=8))
    instance.load_bot('existing')
    instance._doc_manager = MagicMock()
    instance._doc_manager.get_documents.return_value = []
    instance.initialize_bot_id = MagicMock(return_value='existing')
    if surface == 'api':
        with patch('longtrainer.api._get_trainer', return_value=instance):
            result = TestClient(app).post('/bots/existing/build', json={'agent_type': name, 'tools': [], 'agent_mode': True})
        assert result.status_code == 200
    else:
        args = ['build'] if surface == 'build' else ['bot', 'create']
        with patch('longtrainer.cli._get_trainer', return_value=instance):
            result = CliRunner().invoke(cli, args + ['--agent-type', name, '--tools', '', '--agent'])
        assert result.exit_code == 0, result.output
    assert instance.bot_data['existing']['agent_type'] == name
    assert storage.update_bot.call_args.args[1]['agent_mode'] == (name == 'research')
    chat = instance.new_chat('existing')
    expected = AgentBot if name == 'research' else RAGBot
    assert isinstance(instance.bot_data['existing']['chains'][chat], expected)
    assert instance.get_response('Question', 'existing', chat)[0] == 'Local answer'
