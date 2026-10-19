"""Yahoo ticker news can be empty while its news-search endpoint works."""
from unittest.mock import MagicMock

import pytest


def setup_tool(monkeypatch, primary):
    yf = pytest.importorskip("yfinance")
    from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
    from longtrainer.tools import get_yahoo_finance_tool

    monkeypatch.setattr(YahooFinanceNewsTool, "_run", lambda self, query, run_manager=None: primary)
    search = MagicMock()
    monkeypatch.setattr(yf, "Search", search)
    return get_yahoo_finance_tool(), search


def test_yahoo_preserves_existing_article_response(monkeypatch):
    tool, search = setup_tool(monkeypatch, "Existing article text")
    assert tool.invoke("MSFT") == "Existing article text"
    search.assert_not_called()


def test_yahoo_falls_back_to_actual_news_search_interface(monkeypatch):
    tool, search = setup_tool(monkeypatch, "No news found for company that searched with MSFT ticker.")
    search.return_value.news = [{"title": "Microsoft announcement", "publisher": "Example Publisher",
                                 "link": "https://example.com/news", "providerPublishTime": 1700000000}]
    response = tool.invoke("MSFT")
    assert "Microsoft announcement" in response
    assert "Example Publisher" in response
    assert "https://example.com/news" in response
    assert "2023-11-14" in response
    assert "headlines" in response
    search.assert_called_once_with("MSFT", news_count=5, timeout=10)


def test_yahoo_empty_fallback_retains_no_news_result(monkeypatch):
    primary = "No news found for company that searched with MSFT ticker."
    tool, search = setup_tool(monkeypatch, primary)
    search.return_value.news = []
    assert tool.invoke("MSFT") == primary
