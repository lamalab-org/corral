"""An Inspect solver that replays answers a policy already gave."""

from __future__ import annotations

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.solver._multiple_choice import (
    parse_answers,
    set_choices_based_on_generated_response,
)

__all__ = ["replay_solver"]


@solver
def replay_solver(answers: dict[str, str]) -> Solver:
    """Put the policy's completion on each sample, as if it had just generated it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate  # the policy ran earlier, in the jail
        completion = answers.get(str(state.sample_id), "")
        state.output = ModelOutput.from_content(model="policy", content=completion)
        state.messages.append(ChatMessageAssistant(content=completion))
        metadata = state.metadata or {}
        answer_format = str(metadata.get("answer_format", "text"))
        # chembench's scorer parses its own markers; choice() reads marked options.
        if (
            metadata.get("benchmark") != "chembench"
            and answer_format.startswith("mcq")
            and state.choices
        ):
            set_choices_based_on_generated_response(
                state,
                parse_answers(state, multiple_correct=answer_format == "mcq_multi"),
            )
        return state

    return solve
