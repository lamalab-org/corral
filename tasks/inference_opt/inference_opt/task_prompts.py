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
`run(questions, ctx)` API described in `guide/policy_api.md`. Don't modify model weights;
the approach you use around the frozen student model is up to you. I have
{n_train} development questions available, with separate labelled examples
accessible a few at a time. The policy will later receive {n_test} new questions.

My software needs to extract each answer from the policy's response. Please put
`ANSWER: <answer>` on the final line for MATH questions. For ChemBench, use
`[ANSWER]<answer>[/ANSWER]`. For GSM-Symbolic, leave the final number as the answer.
Put any explanation before the final answer so I can use the results directly.

The dataset is {benchmark}. Labelled examples are given to the policy as
`ctx.examples`; development questions are available through `evaluate_candidate`.

{budget}

You can use `dry_run_policy` to check a small sample before spending more calls
on a full run with `evaluate_candidate`.

When you're ready, or running low on calls, please call
`submit_policy('{policy_dir}')`, then call `submit_answer('submission.json')`.
Only one submission is allowed. The text passed to `submit_answer` is a completion
signal; the policy you submitted is what I will run on new questions.
"""


def _budget_summary(config: dict) -> str:
    budget = dict(config.get("budget") or {})
    summary = (
        f"I can allow {budget.get('max_experiments', 20)} full development runs, "
        f"{budget.get('max_debug_runs', budget.get('max_debug_questions', 10))} dry runs, "
        f"and {budget.get('max_student_calls', 250)} student model calls in total. "
        f"You can request labelled examples {budget.get('max_reveals', 4)} times, "
        f"with {budget.get('reveal_batch', 5)} questions in each batch. "
        "Each student call may generate at most 8192 tokens, reasoning included."
    )
    final_calls = config.get("final_max_student_calls")
    if final_calls:
        summary += (
            " For subsequent use, I need the policy to handle batches of "
            f"{config.get('n_test', 30)} new questions within {final_calls} "
            "student calls per model."
        )
    return summary


def task_prompt(env: Environment, state: ExecutionState) -> str:
    """Build the request for one inference-optimization task."""
    del state
    task = env.current_task
    config = dict(task.initial_input)
    return _TEMPLATE.format(
        description=task.description,
        policy_dir="policy",
        benchmark=config.get("benchmark", "?"),
        n_train=config.get("n_train", 30),
        n_test=config.get("n_test", 30),
        budget=_budget_summary(config),
    )
