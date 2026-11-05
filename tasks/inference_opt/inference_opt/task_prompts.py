"""Build the prompt shown to the teacher agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corral.core.environment import Environment
    from corral.core.state import ExecutionState

__all__ = ["task_prompt"]

_TEMPLATE = """\
{description}

Please write a reusable Python policy in `{policy_dir}/policy.py` using the
`{policy_api}` API described in `guide/policy_api.md`. Don't modify model weights;
the approach you use around the model is up to you. I have {n_train} development
questions available, with labelled examples accessible a few at a time.

My software needs to extract each answer from the policy's response. Please use
`ANSWER: <answer>`, or `[ANSWER]<answer>[/ANSWER]` for ChemBench, with any explanation
before the marker. Follow the question's answer type and choice labels so I can
use the results directly.

{budget}

You can use `dry_run_policy` to check a small sample before spending more calls
on a full run with `evaluate_candidate`.

When you're ready, or running low on calls, please call
`submit_policy('{policy_dir}')`, then call `submit_answer('submission.json')`.
That gives me a bundle pointing to the policy code and configuration. Return
that filename as your final answer so I can run the policy on new questions.
"""


def _budget_summary(config: dict) -> str:
    budget = dict(config.get("budget") or {})
    summary = (
        f"I can allow {budget.get('max_experiments', 20)} full development runs, "
        f"{budget.get('max_debug_runs', budget.get('max_debug_questions', 10))} dry runs, "
        f"and {budget.get('max_student_calls', 250)} student model calls in total. "
        f"You can request labelled examples {budget.get('max_reveals', 4)} times, "
        f"with {budget.get('reveal_batch', 5)} questions in each batch."
    )
    final_calls = config.get("final_max_student_calls")
    if final_calls:
        summary += (
            f" For subsequent use, I need the policy to handle batches of "
            f"{config.get('n_test', 30)} new questions within {final_calls} "
            "student calls per model."
        )
    return summary


def task_prompt(env: Environment, state: ExecutionState) -> str:
    """Build the prompt for one inference-optimization task."""
    del state
    task = env.current_task
    config = dict(task.initial_input)
    return _TEMPLATE.format(
        description=task.description,
        policy_api=config.get("policy_api", "primitive"),
        n_train=config.get("n_train", 30),
        budget=_budget_summary(config),
        policy_dir="policy",
    )
