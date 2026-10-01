"""Build the prompt shown to the teacher agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corral.core.environment import Environment
    from corral.core.state import ExecutionState

__all__ = ["task_prompt"]

_TEMPLATE = """\
{description}

## The setup

A frozen student model is served for you. Implement any strategy that does not
modify model weights. You can change the prompts, how many samples you draw, how
they are combined, whether one call checks another, what is remembered between
questions, or any strategy you can come up with.

Write it as a Python policy in `{policy_dir}/policy.py`. The API is documented in `guide/policy_api.md`.

Return the answer marker required by the benchmark scorer: use `ANSWER: <answer>` for every benchmark except ChemBench, which uses `[ANSWER]<answer>[/ANSWER]`.
The scorer parses and grades these markers.

Benchmark: {benchmark}
Student model(s): {models}
Policy API: {policy_api}
Train questions: {n_train} (labels revealed a few at a time)
Test questions: {n_test} (held out; you never should see these or their answers)

## Scoring

{scoring}


## Budget

{budget}

Every experiment costs real inference. `dry_run_policy` is much cheaper than
`evaluate_candidate` and catches the errors that would otherwise waste one.

## Submitting

When you are done - or when your budget is nearly gone - call `submit_policy('{policy_dir}')`, then pass the exact string it returns to `submit_answer`.
`submit_policy` runs your policy once on the held-out test questions and keeps the result hidden until the episode is scored.
You have {max_submissions} submissions; the last one counts. A policy that is never submitted scores nothing, so submit early and again if you improve on it.
"""


def _budget_summary(config: dict) -> str:
    budget = dict(config.get("budget") or {})
    lines = [
        f"- {budget.get('max_experiments', 20)} full evaluations on the train split",
        f"- {budget.get('max_debug_runs', budget.get('max_debug_questions', 10))} dry runs",
        f"- {budget.get('max_student_calls', 250)} student model calls in total",
        f"- {budget.get('max_reveals', 4)} reveals of {budget.get('reveal_batch', 5)} "
        "labelled train questions each",
    ]
    final_calls = config.get("final_max_student_calls")
    if final_calls:
        lines.append(
            f"- at test time your policy may make {final_calls} student calls in "
            f"total, per model - plan for that, not for unlimited sampling"
        )
    return "\n".join(lines)


def _scoring_summary(config: dict) -> str:
    """Describe the pass rule the scorer will apply to this task."""
    rule = dict(config.get("pass_rule") or {"kind": "headroom", "min_closed": 0.5})
    joint = len(config.get("models", [])) > 1
    if rule.get("kind") == "continuous":
        lines = [
            "- Your submitted policy is run on the held-out test split, and your score "
            "is the improvement over the student's measured zero-shot baseline on "
            "those same questions.",
        ]
        if joint:
            lines.append("- For two student models, the score is the smaller improvement.")
        return "\n".join(lines)
    share = f"{float(rule.get('min_closed', 0.5)):.0%}"
    lines = [
        "- Your submitted policy is run on the held-out test split. The task passes "
        f"(score 1) if the policy answers correctly at least {share} of the test "
        "questions the student's measured zero-shot baseline gets wrong; otherwise "
        "it scores 0.",
    ]
    if joint:
        lines.append("- With two student models, both must reach this to pass.")
    return "\n".join(lines)


def task_prompt(env: Environment, state: ExecutionState) -> str:
    """Build the prompt for one inference-optimization task."""
    del state
    task = env.current_task
    config = dict(task.initial_input)
    return _TEMPLATE.format(
        description=task.description,
        benchmark=config.get("benchmark", "?"),
        models=(
            "two frozen student models"
            if len(config.get("models", [])) > 1
            else "one frozen student model"
        ),
        policy_api=config.get("policy_api", "primitive"),
        n_train=config.get("n_train", 30),
        n_test=config.get("n_test", 30),
        budget=_budget_summary(config),
        scoring=_scoring_summary(config),
        policy_dir="policy",
        max_submissions=int((config.get("budget") or {}).get("max_submissions", 3)),
    )
