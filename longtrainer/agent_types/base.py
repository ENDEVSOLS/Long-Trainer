"""Purpose configuration, independent of optional tool dependencies."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AgentTypeConfig(BaseModel):
    """Immutable purpose defaults for bot construction."""

    model_config = ConfigDict(frozen=True)
    name: str
    system_prompt: str
    default_tools: tuple[str, ...] = ()
    mode: Literal["agent", "rag"] = "agent"
