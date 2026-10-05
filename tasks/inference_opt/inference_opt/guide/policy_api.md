# Policy API

Write one policy in `policy/policy.py`.
It defines `run(questions, ctx)`, which gets every question of a run at once and records an answer for each:

```python
class Policy:
    def run(self, questions, ctx):
        texts = ctx.student.generate([q.text for q in questions])  # sent in batch
        ctx.answer({q.id: text for q, text in zip(questions, texts)})
```

`run` may also be a plain module-level function.
Everything else can be defined by you: the order of questions, what is sent side by side, what is remembered between questions, and how the call budget is spread.
Keep any state you need in ordinary Python variables.

Answers are text ending in `ANSWER: <answer>`, or `[ANSWER]<answer>[/ANSWER]` for ChemBench.
You can also give `Answer(final=...)` from `inference_opt.api`, which adds the right marker. `inference_opt.api` is the only `inference_opt` module a policy may import.

## Calling the student

`ctx.student.generate(prompt, *, system=None, temperature=0.0, max_tokens=None, stop=None, seed=None, question_id=None, component=None)`

- **One prompt** (a string, or a list of `{"role": ..., "content": ...}` messages) returns one completion as a `str`. A failed request raises.
- **A list of prompts** returns a list of completions, in the same order, and sends them in a batch.
A failed entry comes back as `""` and is logged.
Each entry is a prompt, or a dict `{"prompt": ..., **settings}` that sets its own `system`, `temperature`, `max_tokens`, `stop`, `seed`, `question_id` or `component`.
The keyword arguments apply to every entry that does not.

Sampling one question several times is a list with that prompt repeated:

```python
votes = ctx.student.generate(
    [{"prompt": q.text, "seed": s} for s in range(5)], temperature=0.8
)
```

Every completion costs one call of the run's budget; `ctx.budget` says how many are left.
A batch larger than what is left sends what fits and returns `""` for the rest; a call when nothing is left raises `BudgetExhausted`.

No call generates more than 8192 tokens, reasoning included: the limit the zero-shot baseline was measured with.
`max_tokens=None` uses the manifest's `max_tokens_per_call`; a larger request is cut to 8192.
How many requests run at once is set by the task, not the policy.
`question_id` attributes a call to a question in the run's diagnostics; `component` labels it in `student_calls.jsonl`.

## Recording answers

`ctx.answer(question_id, text)`, or `ctx.answer({question_id: text, ...})` for several at once, records answers as soon as you call it.
A later answer for the same question replaces the earlier one, and a question never answered counts as wrong.


## Question

`question` has `id`, `text`, `benchmark`, `answer_type`, `choices`,
`choice_labels`, `topic`, `index`, and `total`. Use
`question.rendered_choices()` for multiple-choice prompts.

## The context

- `ctx.student`: the student, as above.
- `ctx.examples`: the revealed train questions, as objects with the attributes `question`, `answer` (gold), `baseline_completion` and `baseline_correct`.
- `ctx.answer(...)`: record answers, as above.
- `ctx.budget`: student calls left for this run.
- `ctx.log(text, question_id=None)`: add a diagnostic line.

Your policy runs in a sandbox with no network: the student is reachable only through `ctx.student`, and every call is counted.

## Optional manifest

Keep configuration in the same `policy.py`. All fields are optional:

```python
MANIFEST = {
    "name": "my-policy",
    "version": 1,
    "max_tokens_per_call": 8192,
}
```

`max_tokens_per_call` (default 8192) is the default `max_tokens` for each call; it cannot raise the 8192 limit. Unknown fields are rejected.

## What each run writes

After a dry run or an experiment, `runs/<run_id>/<model>/` holds `predictions.jsonl` (your answer, whether it was right, calls and log per question), `student_calls.jsonl` (every prompt and completion) and `summary.json`.
Gold answers appear only through `ctx.examples` and `inspect_failures`, for questions you revealed.
