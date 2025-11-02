"""
Reflection components for the Reflexion agent architecture.

This module implements the Self-Reflection module (Msr) and long-term episodic memory
components of the Reflexion architecture as described in the paper.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents.prompt_utils import get_prompt
from corral.agents.utils import LiteLLMMessage, llm_call


@dataclass
class Reflection:
    """
    A single reflection from a failed attempt.

    Represents a verbal lesson learned from analyzing a trajectory
    that did not achieve the desired outcome.

    Attributes:
        trial_index (int): The index of the trial that generated this reflection
        task_id (str): The task being attempted
        error_signal (str): Error message or feedback from the evaluator
        trajectory (list[LiteLLMMessage]): Copy of the agent's message history during the attempt
        reflection_text (str): The LLM-generated reflection text (actionable insight)
        score (float): The score achieved in this attempt
        timestamp (datetime): When this reflection was created
    """

    trial_index: int
    task_id: str
    trajectory: list[LiteLLMMessage]
    reflection_text: str
    score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert reflection to dictionary for serialization."""
        return {
            "trial_index": self.trial_index,
            "task_id": self.task_id,
            "trajectory": [dict(msg) for msg in self.trajectory],
            "reflection_text": self.reflection_text,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reflection":
        """Create reflection from dictionary."""
        return cls(
            trial_index=data["trial_index"],
            task_id=data["task_id"],
            trajectory=[LiteLLMMessage(**msg) for msg in data["trajectory"]],
            reflection_text=data["reflection_text"],
            score=data["score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class ReflectionMemory:
    """
    Manages long-term episodic memory for the Reflexion agent.

    Stores reflections from previous failed attempts and formats them
    for injection into the actor's prompt. Uses a FIFO strategy to
    maintain a bounded memory size.

    Args:
        reflections (list[Reflection]): List of stored reflections
        max_size (int): Maximum number of reflections to keep
    """

    def __init__(self, max_size: int = 3):
        self.reflections: list[Reflection] = []
        self.max_size = max_size

    def add_reflection(self, reflection: Reflection) -> None:
        """
        Add a reflection to memory, removing oldest if at capacity.

        Args:
            reflection (Reflection): The reflection to add
        """
        self.reflections.append(reflection)

        # Maintain max size with FIFO policy
        if len(self.reflections) > self.max_size:
            removed = self.reflections.pop(0)
            logger.debug(
                f"Removed oldest reflection from attempt {removed.trial_index} "
                f"to maintain max_size={self.max_size}"
            )

    def clear(self) -> None:
        """Clear all reflections from memory."""
        self.reflections.clear()

    def format_for_prompt(self) -> str:
        """
        Format all reflections as text for injection into the actor's prompt.

        Returns:
            str: Formatted string with all reflections, or empty string if no reflections
        """
        if not self.reflections:
            return ""

        formatted_reflections = []
        for _i, reflection in enumerate(self.reflections, 1):
            formatted_reflections.append(
                f"Attempt {reflection.trial_index + 1} (Score: {reflection.score:.2f}):\n"
                f"{reflection.reflection_text}"
            )

        return "\n\n".join(formatted_reflections)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "max_size": self.max_size,
            "reflections": [r.to_dict() for r in self.reflections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionMemory":
        """Create memory from dictionary."""
        memory = cls(max_size=data["max_size"])
        memory.reflections = [Reflection.from_dict(r) for r in data["reflections"]]
        return memory


class ReflectionModule:
    """
    Self-Reflection module that generates verbal reflections.

    Converts sparse evaluator feedback (scores, errors) and trajectories
    into actionable textual insights that the actor can use to improve
    performance on subsequent attempts.

    This is the core component that implements "reinforcement via self-reflection"
    by linguistically amplifying the evaluator's feedback signal.

    Args:
        model (str): The LLM model to use for generating reflections
        reflection_prompt (str | Any): The prompt template for generating reflections
        temperature (float): Temperature for reflection generation
        prompt_store (PromptStore | None): The prompt store instance to use
    """

    # Default reflection prompt ID - will be created in prompt store
    DEFAULT_REFLECTION_PROMPT_ID = "reflexion-default-v1"

    def __init__(
        self,
        model: str,
        reflection_prompt: str | Any | None = None,
        reflection_prompt_id: str | None = None,
        prompt_store: PromptStore | None = None,
        temperature: float = 0.0,  # Use deterministic reflection
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        self.prompt_store = prompt_store

        # Load or use provided prompt
        if reflection_prompt is not None:
            self.reflection_prompt = reflection_prompt
        elif reflection_prompt_id:
            if not prompt_store:
                raise ValueError(
                    "prompt_store must be provided when using reflection_prompt_id"
                )
            self.reflection_prompt = get_prompt(
                prompt_store, None, reflection_prompt_id
            )
        else:
            # Use default inline prompt if no store available
            self.reflection_prompt = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default reflection prompt as inline string."""
        return """You are a reflection module analyzing a failed attempt at solving a task.

Your goal is to provide a concise, actionable reflection that will help improve performance on the next attempt.

Task:
{task_id} - {trial_id}

- Score achieved: {score}
- Error/Feedback: {error_message}

Trajectory (key steps):
{trajectory_summary}

Please provide a brief reflection (2-5 sentences) that:
1. Identifies what went wrong in this attempt
2. Suggests a specific strategy to avoid this mistake in the next attempt
3. Is actionable and directly applicable to solving this task
"""

    def generate_reflection(
        self,
        task_id: str,
        trial_id: str,
        trajectory: list[LiteLLMMessage],
        score: float,
    ) -> tuple[str, dict[str, int]]:
        """
        Generate a reflection from a failed attempt.

        Args:
            task_id (str): The task being attempted
            trial_id (str): The specific trial/attempt identifier
            trajectory (list[LiteLLMMessage]): The agent's message history during the attempt
            score (float): The score achieved (typically low for failed attempts)

        Returns:
            tuple[str, dict[str, int]]: A tuple of (reflection_text, token_usage_dict) where
                token_usage_dict contains 'prompt_tokens', 'completion_tokens', and 'total_tokens'
        """
        logger.info(f"Generating reflection for attempt with score {score:.2f}")

        # Summarize trajectory (take key messages to avoid context overflow)
        trajectory_summary = self._summarize_trajectory(trajectory)

        # Prepare prompt
        if hasattr(self.reflection_prompt, "fill"):
            # Use Jinja template
            prompt_text = self.reflection_prompt.fill(
                {
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "score": score,
                    "trajectory_summary": trajectory_summary,
                }
            )
        else:
            # Use string format
            prompt_text = self.reflection_prompt.format(
                task_id=task_id,
                trial_id=trial_id,
                score=score,
                trajectory_summary=trajectory_summary,
            )

        # Generate reflection
        messages = [LiteLLMMessage(role="user", content=prompt_text)]

        try:
            response, usage_info = llm_call(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                return_usage=True,
                **self.kwargs,
            )

            reflection_text = response.content.strip()
            logger.debug(f"Generated reflection: {reflection_text}")

            # usage_info is a dict with prompt_tokens, completion_tokens, total_tokens
            return reflection_text, usage_info

        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            # Return a basic reflection on error with zero token usage
            fallback_reflection = (
                f"Previous attempt achieved score {score:.3f}. "
                "Consider reviewing the approach and trying a different strategy."
            )
            return fallback_reflection, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    def _summarize_trajectory(self, trajectory: list[LiteLLMMessage]) -> str:
        """
        Summarize trajectory to avoid context overflow.

        Preserves all the messages except for the tool results that are summarized.

        Args:
            trajectory (list[LiteLLMMessage]): The full message history

        Returns:
            str: Formatted summary of the trajectory
        """
        return self._format_messages(trajectory)

    def _format_messages(self, messages: list[LiteLLMMessage]) -> str:
        """Format messages for inclusion in reflection prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Check if this is a tool-related message
            is_tool_message = role == "tool" or (
                role == "user"
                and isinstance(content, str)
                and content.startswith("Observation:")
            )

            if is_tool_message:
                # Reduce tool message content to essential metadata
                summarized_content = self._summarize_tool_message(msg, content)
                formatted.append(f"{role.upper()}: {summarized_content}")
            else:
                formatted.append(f"{role.upper()}: {content}")

        return "\n\n".join(formatted)

    def _summarize_tool_message(self, msg: LiteLLMMessage, content: str) -> str:
        """
        Summarize tool message to include only essential metadata.

        Extracts: status, error_message, duration, arguments, tool_name
        and a brief indication of result without the full content.

        Args:
            msg (LiteLLMMessage): The message object
            content (str): The message content

        Returns:
            str: Summarized tool message string
        """
        try:
            # Try to parse content as JSON for tool messages
            if content.startswith("Observation:"):
                # Extract JSON from "Observation: {json}" format
                json_str = content.replace("Observation:", "").strip()
                tool_data = json.loads(json_str)
            else:
                # For role="tool", content is typically the JSON directly
                tool_data = json.loads(content) if isinstance(content, str) else content

            # Extract essential fields
            tool_name = tool_data.get("tool_name") or msg.get("name", "unknown_tool")
            status = tool_data.get("status", "unknown")
            error_message = tool_data.get("error_message", "")
            duration = tool_data.get("duration", "N/A")
            arguments = tool_data.get("arguments", {})

            # Build summarized content
            parts = [f"Tool: {tool_name}", f"Status: {status}"]

            if duration != "N/A":
                parts.append(f"Duration: {duration}s")

            if error_message:
                parts.append(f"Error: {error_message}")

            if arguments:
                # Truncate arguments if very long
                args_str = json.dumps(arguments)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "...}"
                parts.append(f"Arguments: {args_str}")

            # Add success/failure indicator without full result
            if status == "success" or tool_data.get("success", False):
                result = tool_data.get("result", "")
                if result:
                    # Just indicate result type/size, not full content
                    if isinstance(result, str):
                        result_info = f"<returned {len(result)} characters>"
                    elif isinstance(result, list | dict):
                        result_info = f"<returned {type(result).__name__}>"
                    else:
                        result_info = "<returned result>"
                    parts.append(f"Result: {result_info}")

            return " | ".join(parts)

        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            # If parsing fails, return truncated original content
            logger.debug(f"Failed to parse tool message, using truncated content: {e}")
            truncated = content[:200] if len(content) > 200 else content
            if len(content) > 200:
                truncated += "..."
            return truncated


def create_reflexion_history(memory: ReflectionMemory) -> list[LiteLLMMessage] | None:
    """
    Convert reflection memory into history messages for actor prompt injection.

    Args:
        memory (ReflectionMemory): The reflection memory to convert

    Returns:
        list[LiteLLMMessage]: List with a single system message containing formatted reflections,
        or None if no reflections are available
    """
    if not memory.reflections:
        return None

    reflection_text = (
        "=== LESSONS FROM PREVIOUS ATTEMPTS ===\n\n"
        "You have attempted this task before. Here are reflections from those attempts "
        "that can help you succeed this time:\n\n"
        + memory.format_for_prompt()
        + "\n\n=== END OF LESSONS ===\n\n"
        "Use these insights to guide your approach. Avoid the mistakes identified above."
    )

    return [
        LiteLLMMessage(
            role="system",
            content=reflection_text,
            name="reflexion-memory",
        )
    ]
