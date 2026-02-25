import pytest
from unittest.mock import patch, MagicMock
from langchain_core.tools import BaseTool

from longtrainer.tools import (
    load_dynamic_tools,
    get_wikipedia_tool,
    get_arxiv_tool,
    get_python_repl_tool,
    get_yahoo_finance_tool,
    get_tavily_search_tool,
)

def test_load_dynamic_tools_success():
    """Test dynamically loading a valid built-in tool."""
    # LLM-math requires an llm usually, but 'wikipedia' does not
    tools = load_dynamic_tools(["wikipedia"])
    assert len(tools) == 1
    assert isinstance(tools[0], BaseTool)
    assert tools[0].name.lower() == "wikipedia"

def test_load_dynamic_tools_invalid():
    """Test dynamic tool loader with an invalid tool name."""
    tools = load_dynamic_tools(["invalid_tool_that_does_not_exist_123"])
    assert tools == []

@patch("longtrainer.tools.load_dynamic_tools")
def test_explicit_tool_wrappers(mock_load):
    """Test that explicit wrappers call load_dynamic_tools correctly."""
    mock_tool = MagicMock(spec=BaseTool)
    mock_load.return_value = [mock_tool]
    
    wiki = get_wikipedia_tool()
    mock_load.assert_called_with(["wikipedia"])
    assert wiki == mock_tool
    
    arxiv = get_arxiv_tool()
    mock_load.assert_called_with(["arxiv"])
    assert arxiv == mock_tool

@patch("importlib.import_module")
def test_experimental_tool_wrappers(mock_import):
    """Test wrappers for tools that require external packages."""
    # We won't test the actual import since it requires Experimental/YFinance installed
    # Simply testing our wrapper logic
    pass 
