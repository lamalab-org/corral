# Policy API

Put one policy in `policy/policy.py`. Define a `Policy` class with `solve()`.

```python
class Policy:
    def solve(self, question, ctx):
        return ctx.student.generate(question.text)
```

Return `ANSWER: <answer>` for most tasks and `[ANSWER]<answer>[/ANSWER]` for
ChemBench. You can also return `Answer(final=...)` from `inference_opt.api`.

## Question

`question` has `id`, `text`, `benchmark`, `answer_type`, `choices`,
`choice_labels`, `topic`, `index`, and `total`. Use
`question.rendered_choices()` for multiple-choice prompts.

## Primitive API

The default context has:

- `ctx.student.generate(prompt, *, system=None, temperature=0.0,
  max_tokens=1024, stop=None, seed=None, component=None)`: one metered call.
- `ctx.scratch`: state for the current question.
- `ctx.log(text)`: add a diagnostic line.

`ctx.student.calls_remaining` reports the current allowance. Set
`ctx.scratch["fallback"]` to a safe answer for error recovery.
Prompts may be strings or message dictionaries with `role` and `content`.

## Optional manifest

Keep configuration in the same `policy.py`; no second manifest file is needed.
All fields are optional:

```python
MANIFEST = {
    "name": "my-policy",
    "max_calls_per_question": 8,
    "max_tokens_per_call": 2048,
}
```

`max_calls_per_question` is the policy's per-question cap. The task also has a
total run budget. `setup_calls` reserves calls for optional `setup(ctx)`.
`config` is passed to setup. `version` and `components` are diagnostic fields.
Unknown fields are rejected.

<!-- enhanced:start -->
## Enhanced API

Enhanced tasks also provide:

```python
ctx.student.sample(prompt, *, n, temperature=0.8, system=None,
                   max_tokens=1024, component=None)
ctx.student.batch(prompts, *, system=None, temperature=0.0,
                  max_tokens=1024, component=None)
```

`sample()` costs `n` calls; `batch()` costs one call per prompt. Repeated
`generate()` calls are equivalent.

Set `MANIFEST["memory"] = "shared"` to enable state across questions:

```python
ctx.memory.get(key, default)
ctx.memory.set(key, value)
ctx.memory.append(key, value)
ctx.memory.items()
```

`setup(ctx)` receives the revealed labelled training examples. Enhanced solve
contexts also provide `memory`, `artifacts`, `budget_remaining`,
`question_budget`, and `component`. Use `with ctx.component("name"):` to label
model calls.

Policies may keep private state on their `Policy` instance. State shared through
`ctx.memory` requires the `shared` manifest setting.
<!-- enhanced:end -->
