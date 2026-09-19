# Policy API

Write the policy and its optional configuration in `policy/policy.py`. There is
no separate manifest file; the module is imported once.

```python
MANIFEST = {"name": "my-policy"}  # optional

class Policy:
    def setup(self, ctx):       # optional
        pass

    def solve(self, question, ctx):  # required
        return ctx.student.generate(question.text)
```

`solve()` returns the student's completion. Use `ANSWER: <answer>` for every
benchmark except ChemBench, which uses `[ANSWER]<answer>[/ANSWER]`. You may also
return `Answer(final=...)` from `inference_opt.api`.

## Question

`question` provides:

| Field | Description |
| --- | --- |
| `id`, `benchmark` | Stable item ID and benchmark name |
| `text` | Question text |
| `answer_type` | `mcq`, `numeric`, or `text` |
| `choices` | Option texts, or `None` |
| `choice_labels` | Labels matching `choices` |
| `topic` | Optional category |
| `index`, `total` | Position in the run |

Use `question.rendered_choices()` to format options.

## Context

In `solve()`, the required model interface is `ctx.student.generate(...)`.

```python
ctx.student.generate(
    prompt, *, system=None, temperature=0.0, max_tokens=1024,
    stop=None, seed=None, component=None
) -> str                         # costs 1 call
```

- `ctx.student.calls_remaining`: calls available for this question.
- `ctx.log(text)`: adds a line to run diagnostics.
- `ctx.scratch`: per-question state; set `fallback` for error recovery.

Prompts may be strings or message dictionaries with `role` and `content`.

In optional `setup()`, the context also provides `student`, `artifacts`,
`budget_remaining`, `log`, and `config`. `ctx.train_examples`
contains the revealed examples and their answers. Setup calls use the declared
`setup_calls` allowance. A policy may also prepare state in `__init__()`.

## Manifest

All fields are optional. Unknown fields are rejected.

| Field | Default | Description |
| --- | --- | --- |
| `name` | `"policy"` | Diagnostic label |
| `version` | `1` | Manifest version |
| `max_calls_per_question` | `8` | Policy's own per-question cap |
| `setup_calls` | `0` | Calls reserved for `setup()` |
| `max_tokens_per_call` | `2048` | Maximum output tokens per call |
| `config` | `{}` | Values passed to `setup()` |

The policy may import helper modules from its directory and must use
`ctx.student` for model access. Calls raise `BudgetExhausted` when the available
allowance is exhausted. A fallback in `ctx.scratch["fallback"]` is used when a
question fails.

```python
MANIFEST = {
    "name": "vote",
    "max_calls_per_question": 12,
}

class Policy:
    def solve(self, question, ctx):
        ctx.scratch["fallback"] = (question.choice_labels or ("A",))[0]
        prompt = question.text
        if question.choices:
            prompt += "\n\n" + question.rendered_choices()
        return ctx.student.generate(prompt)
```

<!-- enhanced:start -->
## Enhanced API

Enhanced tasks also expose these optional helpers.

```python
ctx.student.sample(
    prompt, *, n, temperature=0.8, system=None,
    max_tokens=1024, component=None
) -> list[str]                   # costs n calls

ctx.student.batch(
    prompts, *, system=None, temperature=0.0,
    max_tokens=1024, component=None
) -> list[str]                   # costs one call per prompt
```

`sample()` and `batch()` are convenience methods. The policy can call
`generate()` repeatedly instead.

Set `MANIFEST["memory"] = "shared"` to use shared memory:

```python
ctx.memory.get(key, default)
ctx.memory.set(key, value)
ctx.memory.append(key, value)
ctx.memory.items()
```

The same memory is available to `setup()` and `solve()`. A policy can instead
keep state on its `Policy` instance.

Set `MANIFEST["components"]` to name diagnostic components. Wrap calls with
`ctx.component(name)` to attribute them:

```python
with ctx.component("critic"):
    answer = ctx.student.generate(prompt)
```
<!-- enhanced:end -->
