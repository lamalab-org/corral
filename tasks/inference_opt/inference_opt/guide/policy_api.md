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

## How answers are graded

**MATH tasks** (`math_*`) are graded exactly as Inspect's MATH eval grades them (`expression_exact_match_sympy` in `inspect_evals`):

- The answer is read from a line `ANSWER: <answer>`, and that line must be the last line of the text.
- It is normalised and compared with the gold answer using sympy, so `ANSWER: \frac{1}{2}` and `ANSWER: 0.5` are the same answer.
- Text with no final `ANSWER:` line is wrong, even if it contains the right answer, for example in `\boxed{...}`.

Gold answers are numbers or expressions such as `\frac{3}{4}` or `1+2\sqrt{3}`.
`question.answer_type` is `"text"` for every MATH question, so it does not say which.

**GSM-Symbolic tasks** (`gsm_*`) are graded as GSM-Symbolic grades them: commas are removed, the **last number** in the text is the answer, and it is compared with the gold answer as a number.
Every gold answer is a number, so `question.answer_type` is `"numeric"`.
Each question is one instance of a template: the same problem with different names and numbers.
`question.topic` names the template, so instances of one template can be recognised; the score counts a template only if all of its instances are right.

You can also record `Answer(final=...)` from `inference_opt.api`, which writes an `ANSWER:` line for you; its value is then the last number for GSM-Symbolic too.
`inference_opt.api` is the only `inference_opt` module a policy may import.

## Calling the student

`ctx.student.generate(prompt, *, system=None, temperature=0.0, max_tokens=None, stop=None, seed=None, question_id=None, component=None)`

- **One prompt** (a string, or a list of `{"role": ..., "content": ...}` messages) returns one `Completion`. A failed request raises.
- **A list of prompts**, at most 1024, returns a list of `Completion`s, in the same order, and sends them in a batch.
A failed entry comes back empty and is logged.
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

A `Completion` is a `str` holding the answer text, so it works wherever text does, with two more attributes:

- `reasoning`: the student's thinking, when the server returns it separately from the answer; otherwise `""`.
- `finish_reason`: `"stop"`, or `"length"` when the call reached `max_tokens`; `completion.truncated` checks the latter.

A call cut off while the student is still thinking returns empty text, but its partial `reasoning` is kept.
A follow-up call can build on it, for example by passing that reasoning back and asking for the final answer only.
Run results count the calls that were cut off, and `student_calls.jsonl` records each call's reasoning and `finish_reason`.

`max_tokens=None` uses the manifest's `max_tokens_per_call`; a larger request is cut to 8192, and run results say how many were.
How many requests run at once is set by the task, not the policy.
`question_id` attributes a call to a question in the run's diagnostics; `component` labels it in `student_calls.jsonl`.

## Recording answers

`ctx.answer(question_id, text)`, or `ctx.answer({question_id: text, ...})` for several at once, records answers as soon as you call it.
A later answer for the same question replaces the earlier one, and a question never answered counts as wrong.


## Question

`question` has `id`, `text`, `benchmark`, `answer_type`, `choices`, `choice_labels`, `topic`, `index` and `total`.
`topic` is the MATH subject, or the GSM-Symbolic template; `index` and `total` give the question's position in the run.
`choices` and `choice_labels` are `None`: these questions have no options.

## The context

- `ctx.student`: the student, as above.
- `ctx.examples`: the example questions you revealed, as objects with the attributes `question`, `answer` (gold), `baseline_completion`, `baseline_reasoning` and `baseline_correct`.
  The baseline fields are this run's student answering zero-shot; `baseline_completion` is empty when it ran out of tokens, and all three are `None` if that reply was never measured.
  Examples are separate from the train and test questions.
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
Gold answers appear only for the example questions you revealed, through `reveal_examples` and `ctx.examples`.
