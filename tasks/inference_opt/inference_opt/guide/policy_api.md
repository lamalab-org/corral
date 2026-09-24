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
  max_tokens=None, stop=None, seed=None, component=None)`: one metered call.
  `max_tokens=None` uses the manifest's `max_tokens_per_call`.
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
    "max_tokens_per_call": 8192,
    "concurrent": True,
}
```

`max_calls_per_question` is the policy's per-question cap. `max_tokens_per_call`
(default 8192) is the policy's per-call output limit and the default for calls
that omit `max_tokens`; the task sets no token cap of its own. The task also has
a total run budget. `setup_calls` reserves calls for optional `setup(ctx)`.
Questions are solved concurrently, so `solve` may run for several questions at
once; `ctx.scratch` is per question. Set `"concurrent": False` if a question
depends on earlier ones, for example through `ctx.memory` or state kept on the
policy object between calls. `config` is passed to setup. `version` and
`components` are diagnostic fields. Unknown fields are rejected.

<!-- enhanced:start -->
## Enhanced API

Enhanced tasks also provide:

```python
ctx.student.sample(prompt, *, n, temperature=0.8, system=None,
                   max_tokens=None, component=None)
ctx.student.batch(prompts, *, system=None, temperature=0.0,
                  max_tokens=None, component=None)
```

`sample()` costs `n` calls; `batch()` sends its prompts concurrently and costs
one call per prompt. Repeated `generate()` calls are equivalent.

`ctx.memory` holds state shared across all questions of a run:

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

Policies may also keep private state on their `Policy` instance. If a question
must see what earlier questions stored, set `"concurrent": False` so they run in
order.
<!-- enhanced:end -->
