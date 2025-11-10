"""
ReflexionAgent: A wrapper agent that adds self-reflection capabilities.

This implements the Reflexion architecture (https://arxiv.org/abs/2303.11366)
by wrapping existing agents with a trial-reflect-retry loop.
"""

from typing import Any

from loguru import logger
from promptstore import PromptStore

from corral.agents import BaseAgent
from corral.agents.reflection import (
    Reflection,
    ReflectionMemory,
    ReflectionModule,
    create_reflexion_history,
)
from corral.agents.utils import LiteLLMMessage
from corral.router.routes import CorralRouter


class ReflexionAgent(BaseAgent):
    """
    Agent that uses the Reflexion framework (https://arxiv.org/abs/2303.11366) to improve through self-reflection.

    The ReflexionAgent wraps an existing agent (the "Actor") and adds:
    1. A Self-Reflection module that generates verbal insights from failures
    2. Long-term episodic memory that stores reflections across attempts
    3. A trial-reflect-retry loop that learns from mistakes

    The architecture follows: Actor + Evaluator (external) + Self-Reflection
    where the Evaluator is provided by the environment's scoring function.

    ## Usage Example

    ```python
    # Create base agent (the Actor)
    base_agent = ReActAgent(model="gpt-4o", max_iterations=10)

    # Wrap with Reflexion capabilities
    reflexion_agent = ReflexionAgent(actor=base_agent, reflection_model="gpt-4o")

    # Use like any other agent - framework handles trials
    runner = CorralRunner(interface, reflexion_agent)
    result = runner.bench(task_ids=["task_1"], trials_per_task=5)
    ```

    Args:
        actor (BaseAgent): The base agent to wrap (ReActAgent, ToolCallingAgent, etc.)
        reflection_model (str): Model to use for generating reflections
        reflection_prompt (str | Any | None): Custom reflection prompt
        reflection_prompt_id (str | None): ID of reflection prompt in prompt store
        reflection_temperature (float | None): Temperature for reflection generation (default: 0.0)
        prompt_store (PromptStore | None): Prompt store instance for reflection prompts
        **kwargs: Additional arguments passed to the actor
    """

    def __init__(
        self,
        actor: BaseAgent,
        reflection_model: str,
        reflection_prompt: str | Any | None = None,
        reflection_prompt_id: str | None = None,
        reflection_temperature: float | None = None,
        prompt_store: PromptStore | None = None,
        **kwargs,
    ):
        super().__init__(
            model=actor.model,
            max_iterations=actor.max_iterations,
            api_endpoint=actor.api_endpoint,
            temperature=actor.temperature,
            prompt_store=prompt_store or actor.store,
            **kwargs,
        )

        # Store the actor (the actual agent doing the work)
        self.actor = actor

        # Determine reflection parameters (prioritize explicitly passed values)
        final_reflection_model = (
            reflection_model if reflection_model is not None else actor.model
        )
        final_reflection_temperature = (
            reflection_temperature if reflection_temperature is not None else 0.0
        )

        # Initialize reflection components
        self.memory = ReflectionMemory(max_size=3)
        self.reflection_module = ReflectionModule(
            model=final_reflection_model,
            reflection_prompt=reflection_prompt,
            reflection_prompt_id=reflection_prompt_id,
            prompt_store=prompt_store or actor.store,
            temperature=final_reflection_temperature,
        )

        # Track current task
        self._current_task_id = None

        # Store messages from previous trial for reflection generation
        self._previous_messages: list[LiteLLMMessage] = []

        # Track token usage for reflection generation
        self.reflection_token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """
        Run the agent with reflexion capabilities.

        This works with the framework's trial system:
        1. Retrieve the last score (if this is not the first trial)
        2. If previous trial exists, generate reflection and add to memory
        3. Inject reflections from memory into history
        4. Run the actor with enriched history
        5. Return the answer (framework will submit and score it)

        Args:
            interface (CorralRouter): The interface to use
            task_id (str): The task ID to solve
            history (list[LiteLLMMessage] | None): Initial history items (optional)
            task_prompt (str | None): Custom task prompt (optional)
            examples (list[str] | None): Few-shot examples (optional)

        Returns:
            str: The final answer from the actor
        """
        logger.info(f"Starting ReflexionAgent for task {task_id}")

        # Try to get last score from previous trial
        last_score_data = None
        try:
            last_score_data = interface.get_last_score(task_id)
            logger.info(
                f"Retrieved last score: {last_score_data['score']} from trial {last_score_data['trial_id']}"
            )
        except Exception as e:
            # First trial - no previous score exists
            logger.info(
                f"No previous trial found (this is likely the first trial): {e}"
            )

        # If we have a previous score, generate reflection from previous attempt
        if last_score_data is not None:
            score = last_score_data.get("score", 0.0)
            trial_id = last_score_data.get("trial_id", "unknown")

            # Generate and store reflection from previous trial
            logger.info(f"Generating reflection from previous trial (score: {score})")
            if self.messages:
                self._generate_and_store_reflection(
                    task_id=task_id,
                    trial_id=trial_id,
                    score=score,
                )

        # Inject reflections from memory into history
        reflexion_history = create_reflexion_history(self.memory)

        # Combine with any provided history
        combined_history = []
        if history:
            combined_history.extend(history)
        if reflexion_history:
            combined_history.extend(reflexion_history)

        final_history = combined_history or None

        # Run the actor with enriched history
        try:
            answer = self.actor.run(
                interface=interface,
                task_id=task_id,
                history=final_history,
                task_prompt=task_prompt,
                examples=examples,
            )

            # Copy actor's messages and token usage for tracking
            self.messages = self.actor.messages.copy()
            self.token_usage = self.actor.token_usage.copy()

            # Store messages for next trial's reflection generation
            self._previous_messages = self.messages.copy()

            logger.info(f"ReflexionAgent completed task {task_id}")
            return answer

        except Exception as e:
            logger.error(f"Error in ReflexionAgent: {e}")
            raise

    def _generate_and_store_reflection(
        self,
        task_id: str,
        trial_id: str,
        score: float,
    ) -> None:
        """
        Generate a reflection from the previous trial and store it in memory.

        This method uses the trajectory stored in self._previous_messages from
        the previous trial and generates a reflection based on the score achieved.

        Args:
            task_id (str): The task ID
            trial_id (str): The trial ID from the previous attempt
            score (float): The score achieved in the previous trial
        """
        # Use the messages stored from the previous trial
        trajectory = self._previous_messages

        if not trajectory:
            logger.warning(
                f"No previous messages found for task {task_id}. "
                "Skipping reflection generation."
            )
            return

        # Generate reflection using the in-memory trajectory directly
        reflection_text, token_usage = self.reflection_module.generate_reflection(
            task_id=task_id,
            trial_id=trial_id,
            trajectory=trajectory,
            score=score,
        )

        # Accumulate reflection token usage
        self.reflection_token_usage["prompt_tokens"] = token_usage.get(
            "prompt_tokens", 0
        )
        self.reflection_token_usage["completion_tokens"] = token_usage.get(
            "completion_tokens", 0
        )
        self.reflection_token_usage["total_tokens"] = token_usage.get("total_tokens", 0)

        # Store the reflection with the full trajectory
        reflection = Reflection(
            trial_index=len(self.memory.reflections),
            task_id=task_id,
            trajectory=trajectory,
            reflection_text=reflection_text,
            score=score,
        )

        # Store in memory
        self.memory.add_reflection(reflection)

        logger.info(
            f"Stored reflection from trial {trial_id} (score: {score:.3f}): "
            f"{reflection_text[:100]}..."
        )

    def get_total_token_usage(self) -> dict[str, int]:
        """
        Get total token usage including actor and reflection module.

        Returns:
            dict[str, int]: Dictionary with token usage statistics including reflection generation tokens
        """
        # Get actor's token usage
        actor_usage = (
            self.actor.get_total_token_usage()
            if hasattr(self.actor, "get_total_token_usage")
            else self.token_usage
        )

        # Combine actor usage with reflection usage
        total_usage = {
            "prompt_tokens": actor_usage.get("prompt_tokens", 0)
            + self.reflection_token_usage.get("prompt_tokens", 0),
            "completion_tokens": actor_usage.get("completion_tokens", 0)
            + self.reflection_token_usage.get("completion_tokens", 0),
            "total_tokens": actor_usage.get("total_tokens", 0)
            + self.reflection_token_usage.get("total_tokens", 0),
        }

        # Add breakdown if desired
        total_usage["actor_tokens"] = actor_usage.get("total_tokens", 0)
        total_usage["reflection_tokens"] = self.reflection_token_usage.get(
            "total_tokens", 0
        )

        return total_usage
