"""Built-in tools and tool registry for LongTrainer V2.

Provides a ToolRegistry for managing per-bot tools, plus built-in tools
for web search and document reading.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool
from duckduckgo_search import DDGS


class ToolRegistry:
    """Registry for managing tools attached to bots.

    Each bot can have its own set of tools. The registry maintains
    a name → tool mapping for easy add/remove/list operations.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, t: BaseTool) -> None:
        """Register a tool.

        Args:
            t: A LangChain-compatible tool to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if t.name in self._tools:
            raise ValueError(f"Tool '{t.name}' is already registered.")
        self._tools[t.name] = t

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name.

        Args:
            name: The name of the tool to remove.

        Raises:
            KeyError: If the tool name is not found.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        del self._tools[name]

    def get_tools(self) -> list[BaseTool]:
        """Get all registered tools as a list.

        Returns:
            List of registered tools.
        """
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name to check.

        Returns:
            True if the tool exists in the registry.
        """
        return name in self._tools

    def list_tool_names(self) -> list[str]:
        """List the names of all registered tools.

        Returns:
            List of tool name strings.
        """
        return list(self._tools.keys())



@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results.

    Args:
        query: The search query string.

    Returns:
        Formatted string of search results with titles, snippets, and links.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)
        if not results:
            return "No results found."

        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')}\n"
                f"Link: {r.get('href', 'N/A')}"
            )
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def document_reader(file_path: str) -> str:
    """Read and extract text content from a document file.

    Supports PDF, DOCX, TXT, MD, CSV, and HTML files.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content from the file.
    """
    try:
        from longtrainer.loaders import DocumentLoader

        loader = DocumentLoader()
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

        if ext == "pdf":
            docs = loader.load_pdf(file_path)
        elif ext in ("docx", "doc"):
            docs = loader.load_doc(file_path)
        elif ext in ("md", "markdown", "txt"):
            docs = loader.load_markdown(file_path)
        elif ext == "csv":
            docs = loader.load_csv(file_path)
        elif ext in ("html", "htm"):
            docs = loader.load_text_from_html(file_path)
        else:
            docs = loader.load_unstructured(file_path)

        if not docs:
            return f"No content could be extracted from {file_path}"

        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Error reading document: {e}"


def get_builtin_tools() -> list[BaseTool]:
    """Get the list of built-in tools.

    Returns:
        List containing web_search and document_reader tools.
    """
    return [web_search, document_reader]
