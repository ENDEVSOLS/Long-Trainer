"""LongTrainer V2 — Production-Ready RAG Framework.

Multi-tenant bots, streaming, tools, and persistent memory — all batteries included.
"""

from longtrainer.trainer import LongTrainer
from longtrainer.tools import ToolRegistry, web_search, document_reader, get_builtin_tools

__all__ = [
    "LongTrainer",
    "ToolRegistry",
    "web_search",
    "document_reader",
    "get_builtin_tools",
]

__version__ = "1.1.0"
