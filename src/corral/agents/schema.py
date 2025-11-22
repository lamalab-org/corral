from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    """Represents an action to be taken"""

    tool_name: str
    arguments: dict[str, Any]


@dataclass
class Thought:
    """Represents agent's reasoning step"""

    content: str
