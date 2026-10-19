"""Named bot purposes with lazy tool construction."""
from .base import AgentTypeConfig
from . import research, coding, financial, customer_support


class AgentTypeRegistry:
    """Registry separate from the runtime ToolRegistry."""

    _configs: dict[str, AgentTypeConfig] = {}

    @classmethod
    def register(cls, config: AgentTypeConfig) -> None:
        """Register or replace a purpose configuration."""
        cls._configs[config.name] = config

    @classmethod
    def names(cls) -> list[str]:
        """Return supported names in deterministic order."""
        return sorted(cls._configs)

    @classmethod
    def get(cls, name: str) -> AgentTypeConfig:
        """Look up a purpose or raise an actionable error."""
        if name not in cls._configs:
            raise ValueError(f"Unknown agent type {name!r}. Supported types: {', '.join(cls.names())}")
        return cls._configs[name]


for _module, _mode in ((research, "agent"), (coding, "agent"), (financial, "agent"), (customer_support, "rag")):
    AgentTypeRegistry.register(AgentTypeConfig(
        name=_module.__name__.rsplit(".", 1)[-1], system_prompt=_module.SYSTEM_PROMPT,
        default_tools=_module.DEFAULT_TOOLS, mode=_mode,
    ))


def resolve_tools(specs: list) -> list:
    """Resolve explicit loader identifiers and the minimal built-in factory adapter.

    A missing dependency, key, or tool is a construction error, never an empty
    successful default. Tool objects are accepted without serialization.
    """
    from importlib import import_module
    from longtrainer import tools

    factories = {"tavily": tools.get_tavily_search_tool,
                 "python_repl": tools.get_python_repl_tool,
                 "yahoo_finance_news": tools.get_yahoo_finance_tool}
    resolved = []
    for spec in specs:
        if not isinstance(spec, str):
            resolved.append(spec)
            continue
        try:
            if spec == "yahoo_finance_news":
                import_module("yfinance")  # The existing factory defers this import until execution.
            loaded = [factories[spec]()] if spec in factories else tools.load_dynamic_tools([spec])
            if not loaded or any(tool is None for tool in loaded):
                raise RuntimeError("tool factory returned no tool")
            resolved.extend(loaded)
        except Exception as error:
            raise ValueError(f"Cannot load tool {spec!r}; install the purpose extra and configure required keys.") from error
    return resolved


__all__ = ["AgentTypeConfig", "AgentTypeRegistry"]
