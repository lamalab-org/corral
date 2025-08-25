from dataclasses import dataclass

from corral.agents.base_agent import BaseAgent
from corral.router.routes import CorralRouter

TypeAgent = BaseAgent
TypeRouter = CorralRouter


@dataclass
class ToolResponse:
    """Response from a tool execution"""

    success: bool
    result: str | None
    error: str | None


class BenchmarkError(Exception):
    """Base exception for all benchmark-related errors"""


class TaskNotFoundError(BenchmarkError):
    """Raised when a task ID is not found in the benchmark results"""


class InsufficientTrialsError(BenchmarkError):
    """Raised when there aren't enough trials for a calculation"""


class NoResultsError(BenchmarkError):
    """Raised when trying to calculate metrics without results"""
