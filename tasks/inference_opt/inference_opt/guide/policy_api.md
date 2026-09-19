# Writing a policy

Your policy lives in `policy/policy.py`. It is imported once and then asked to
answer each question. You may add other files next to it (helper modules, prompt
text, demonstration sets) and import them normally.

## The contract

```python
MANIFEST = {"name": "my-policy"}          # optional, see below

class Policy:
    def setup(self, ctx):                 # optional, runs once before any question
        ...

    def solve(self, question, ctx):       # required, runs per question
        return "Reasoning...\nANSWER: B"
```

`solve` returns the student's completion. Inspect Evals owns parsing and grading.
Use `ANSWER: <answer>` for all benchmarks except ChemBench, which requires
`[ANSWER]<answer>[/ANSWER]`. A structured `Answer(final=..., ...)` is also
accepted and is formatted for the selected benchmark by the runner.

### `question`

| field | meaning |
| --- | --- |
| `id` | stable item id |
| `text` | the question |
| `benchmark` | which benchmark it came from |
| `answer_type` | `"mcq"`, `"numeric"` or `"text"` |
| `choices` | tuple of option texts, or `None` |
| `choice_labels` | `("A", "B", ...)` matching `choices` |
| `topic` | subject/category, when the benchmark has one |
| `index`, `total` | position in the run, so you can pace your budget |

`question.rendered_choices()` gives you `"A) Paris\nB) Berlin"` if you want it.

### `ctx` in `solve`

| member | meaning |
| --- | --- |
| `ctx.student.generate(prompt, system=..., temperature=..., max_tokens=...)` | one completion, costs 1 call; ask the student to use the benchmark's answer marker |
| `ctx.student.sample(prompt, n=5, temperature=0.8)` | `n` completions, costs `n` calls |
| `ctx.student.batch([p1, p2, p3])` | several prompts concurrently, costs 3 calls |
| `ctx.student.calls_remaining` | what this question may still spend |
| `ctx.question_budget` | same number, before you start |
| `ctx.budget_remaining` | what the whole run has left |
| `ctx.memory` | state shared across questions (needs `memory="shared"`) |
| `ctx.artifacts` | your policy directory, readable — put prompt files here |
| `ctx.log("...")` | a line you will see in `inspect_failures` |
| `ctx.scratch["fallback"] = "A"` | used as your answer if you later crash or run out of budget |
| `ctx.component("critic")` | context manager tagging calls, for per-component diagnostics |

`prompt` may be a string or a list of `{"role": ..., "content": ...}` dicts.

### `ctx` in `setup`

Everything above except the per-question fields, plus
`ctx.train_examples` — the questions you unlocked with `reveal_train_questions`,
each carrying `.question`, `.answer` (the gold answer), `.baseline_completion`
(what the student said zero-shot) and `.baseline_correct`. This is the only place
gold answers appear, and it is where prompt fitting and demonstration selection
belong. Calls made in `setup` come from a separate allowance.

## MANIFEST

All keys optional; unknown keys are rejected so a typo cannot silently do nothing.

| key | default | meaning |
| --- | --- | --- |
| `name` | `"policy"` | label in diagnostics |
| `memory` | `"none"` | `"shared"` enables `ctx.memory` |
| `max_calls_per_question` | `8` | your own cap; the task cap still applies |
| `setup_calls` | `0` | calls reserved for `setup` |
| `max_tokens_per_call` | `2048` | clamped by the task |
| `components` | `()` | `[{"name": "critic", "kind": "critic"}]`, for diagnostics |
| `config` | `{}` | passed through to `ctx.config` in `setup` |

## Imports and model access

Teacher policy code is trusted inside the Docker trial, so it is not restricted by
an AST validator or a Corral worker sandbox. The policy should use the
provided `ctx.student` client as its model interface; other model endpoints are
outside the task's intended scope. The policy directory may contain normal helper
modules and prompt files.

`dry_run_policy` imports and executes the policy through the real evaluation path,
so use it to catch import, contract, and runtime errors before spending an
experiment.

## Budget

Calls are charged before the request is made. Each question gets a guaranteed
reserve plus access to a shared pool, so spending heavily on early questions does
not starve later ones — but the per-question cap is hard.

When the budget runs out, `ctx.student` raises `BudgetExhausted` (a `RuntimeError`).
Remaining questions still run, and every call raises immediately, so a policy that
degrades gracefully still scores:

```python
def solve(self, question, ctx):
    ctx.scratch["fallback"] = "ANSWER: A"  # used if anything below fails
    try:
        return self._full_strategy(question, ctx)
    except Exception:
        return ctx.scratch["fallback"]
```

Unattempted questions count as wrong — the denominator is always the whole split.

## A complete policy example

```python
from collections import Counter

MANIFEST = {
    "name": "vote-then-check",
    "memory": "shared",
    "max_calls_per_question": 6,
    "setup_calls": 4,
    "components": [{"name": "proposer", "kind": "sampler"},
                   {"name": "checker", "kind": "critic"}],
}

class Policy:
    def __init__(self):
        self._prefix = ""

    def setup(self, ctx):
        wrong = [e for e in ctx.train_examples if not e.baseline_correct]
        ctx.log(f"{len(wrong)} of {len(ctx.train_examples)} revealed items were wrong")
        if wrong:
            self._prefix = "Work step by step, then state the final answer.\n\n"
        ctx.memory.set("seen", 0)

    def solve(self, question, ctx):
        ctx.scratch["fallback"] = (question.choice_labels or ("A",))[0]
        prompt = self._prefix + question.text
        if question.choices:
            prompt += "\n\n" + question.rendered_choices()

        n = 3 if ctx.question_budget >= 4 else 1
        with ctx.component("proposer"):
            drafts = ctx.student.sample(prompt, n=n, temperature=0.7)

        tally = Counter(d.strip()[:200] for d in drafts)
        best, votes = tally.most_common(1)[0]
        if votes == 1 and ctx.question_budget >= 1:
            with ctx.component("checker"):
                best = ctx.student.generate(
                    f"{prompt}\n\nA draft answer was:\n{best}\n\n"
                    "If it is wrong, give the corrected answer; otherwise repeat it."
                )
        ctx.memory.set("seen", ctx.memory.get("seen", 0) + 1)
        return best
```
