import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from longtrainer.api import app


@pytest.fixture
def mock_trainer():
    with patch("longtrainer.api._get_trainer") as mock_get_trainer:
        mock_instance = MagicMock()
        mock_instance.initialize_bot_id.return_value = "bot_123"
        mock_instance.new_chat.return_value = "chat_456"
        mock_instance.list_chats.return_value = ["chat_456", "chat_789"]
        mock_get_trainer.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Test the /health endpoint without initializing the full LongTrainer."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_create_bot_endpoint(client, mock_trainer):
    """Test the POST /bots endpoint."""
    response = client.post("/bots")
    assert response.status_code == 200
    assert response.json() == {"bot_id": "bot_123"}
    mock_trainer.initialize_bot_id.assert_called_once()


def test_new_chat_endpoint(client, mock_trainer):
    """Test the POST /bots/{bot_id}/chats endpoint."""
    response = client.post("/bots/bot_123/chats")
    assert response.status_code == 200
    assert response.json() == {"chat_id": "chat_456"}
    mock_trainer.new_chat.assert_called_once_with("bot_123")


def test_list_chats_endpoint(client, mock_trainer):
    """Test the GET /bots/{bot_id}/chats endpoint."""
    response = client.get("/bots/bot_123/chats")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 2
    assert "chat_456" in response.json()
    mock_trainer.list_chats.assert_called_once_with("bot_123")


def test_chat_sync_endpoint(client, mock_trainer):
    """Test the POST /bots/{bot_id}/chats/{chat_id} endpoint (non-streaming)."""
    mock_trainer.get_response.return_value = ("Test answer", ["source1.txt"])
    
    response = client.post(
        "/bots/bot_123/chats/chat_456",
        json={"query": "Hello world", "stream": False, "web_search": False}
    )
    
    assert response.status_code == 200
    assert response.json() == {"answer": "Test answer", "web_sources": ["source1.txt"]}
    mock_trainer.get_response.assert_called_once_with(
        query="Hello world",
        bot_id="bot_123",
        chat_id="chat_456",
        stream=False,
        uploaded_files=None,
        web_search=False
    )
