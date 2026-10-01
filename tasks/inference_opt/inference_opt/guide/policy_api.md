# Policy API

Put one policy in `policy/policy.py`. Define a `Policy` class whose `run()`
gets every question of a run at once and submits an answer for each.

```python
class Policy:
    def run(self, questions, ctx):
        for q in questions:
            answer = ctx.student.generate(q.text, question_id=q.id)
            ctx.submit(q.id, answer)
```

You decide the order, what runs side by side, what is remembered between
questions, and how the call budget is spread. Keep any state you need in
ordinary Python variables.

If you only need one question at a time, define `solve(question, ctx)` instead
and return the answer; each question is then solved on its own, several at
once.

Answers are text ending in `ANSWER: <answer>`, or `[ANSWER]<answer>[/ANSWER]`
for ChemBench. You can also submit `Answer(final=...)` from `inference_opt.api`,
which adds the right marker. `inference_opt.api` is the only `inference_opt`
module a policy may import.

## Submitting answers

`ctx.submit(question_id, answer)` records an answer as soon as you call it, and
a later submit for the same question replaces it. A question never submitted
counts as wrong. Submit a cheap guess early: if the budget or time runs out, or
`run()` raises, what you already submitted still counts. Returning a dict of
`{question_id: answer}` from `run()` submits those too.

## Question

`question` has `id`, `text`, `benchmark`, `answer_type`, `choices`,
`choice_labels`, `topic`, `index`, and `total`. Use
`question.rendered_choices()` for multiple-choice prompts.

## The context

- `ctx.student.generate(prompt, *, system=None, temperature=0.0,
  max_tokens=None, stop=None, seed=None, question_id=None, component=None)`:
  one metered call. `max_tokens=None` uses the manifest's `max_tokens_per_call`.
  `question_id` attributes the call to a question in the run's diagnostics.
- `ctx.examples`: the revealed train questions, each with `question`, `answer`
  (gold), `baseline_completion` and `baseline_correct`.
- `ctx.submit(question_id, answer)`: record an answer.
- `ctx.budget`: student calls left for this run.
- `ctx.log(text, question_id=None)`: add a diagnostic line.

Prompts may be strings or lists of message dictionaries with `role` and
`content`. When the budget is spent, student calls raise `BudgetExhausted`.

Your policy runs in a sandbox with no network: the student is reachable only
through `ctx.student`, and every call is counted.

## Optional manifest

Keep configuration in the same `policy.py`. All fields are optional:

```python
MANIFEST = {
    "name": "my-policy",
    "version": 1,
    "max_tokens_per_call": 16384,
}
```

`max_tokens_per_call` (default 16384) is the default and ceiling for
`max_tokens` on each call; the task caps it at 32768. Unknown fields are
rejected.

## What each run writes

After a dry run or an experiment, `runs/<run_id>/<model>/` holds
`predictions.jsonl` (your answer, whether it was right, calls and log per
question), `student_calls.jsonl` (every prompt and completion) and
`summary.json`. Gold answers appear only through `ctx.examples` and
`inspect_failures`, for questions you revealed.

<!-- enhanced:start -->
## Enhanced API

Enhanced tasks also provide:

```python
ctx.student.sample(prompt, *, n, temperature=0.8, system=None,
                   max_tokens=None, question_id=None, component=None)
ctx.student.batch(prompts, *, system=None, temperature=0.0,
                  max_tokens=None, question_id=None, component=None)
```

`sample()` costs `n` calls; `batch()` sends its prompts side by side and costs
one call per prompt. `component` labels calls in `student_calls.jsonl`.
<!-- enhanced:end -->
