"""Vision providers may return content blocks instead of plain text."""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from longtrainer.vision_bot import VisionBot


def test_vision_response_extracts_provider_text_blocks():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=[
        {"type": "text", "text": "Red"},
        {"type": "image_url", "image_url": {"url": "https://example.com/unused.png"}},
    ])
    bot = VisionBot(llm, "Describe the image.")
    assert bot.get_response("What color?") == "Red"
