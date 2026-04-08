"""Sycophancy testing hook for injecting simulated user feedback during agent execution.

This module provides a hook that injects a misleading (or neutral) user message
at a specific iteration during task execution. It tests whether agents prioritize
agreeableness over accuracy when faced with user pushback mid-trajectory.

Used with BEFORE_ITERATION hook point.
"""

from loguru import logger

from corral.agents.hooks.core import HookCallback, HookContext
from corral.agents.utils import LiteLLMMessage


def create_sycophancy_hook(
    user_message: str,
    trigger_iteration: int,
) -> HookCallback:
    """Factory function to create a sycophancy testing hook.

    Injects a simulated user message into the agent's conversation at a
    specific iteration. The message appears as a user turn, testing whether
    the agent abandons its current (potentially correct) approach to agree
    with the misleading feedback.

    Note: The Anthropic API requires alternating roles, so litellm merges
    consecutive user messages into a single turn with multiple content blocks.
    The model does see the injected text but may choose to ignore mild messages.
    Use stronger/more authoritative messages for a more pronounced effect.

    Args:
        user_message: The simulated user message to inject.
        trigger_iteration: The iteration number (0-indexed) at which to
            inject the message. Should be mid-trajectory (e.g., ~40% of
            mean steps for the environment) so the agent has committed to
            an approach but still has room to change course.

    Returns:
        HookCallback to be registered with AgentHooks at BEFORE_ITERATION.

    Example:
        >>> hook = create_sycophancy_hook(
        ...     user_message="I think you're going about this wrong...",
        ...     trigger_iteration=5,
        ... )
        >>> hooks.register(HookPoint.BEFORE_ITERATION, hook)
    """

    def sycophancy_hook(context: HookContext) -> None:
        if context.iteration != trigger_iteration:
            return

        logger.info(
            f"Sycophancy hook: injecting user message at iteration "
            f"{context.iteration} for task {context.task_id}"
        )

        context.messages.append(LiteLLMMessage(role="user", content=user_message))

        context.metadata["sycophancy_injected"] = True
        context.metadata["sycophancy_trigger_iteration"] = trigger_iteration
        context.metadata["sycophancy_message"] = user_message

    return sycophancy_hook
