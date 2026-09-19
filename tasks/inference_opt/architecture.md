# Inference-opt architecture

Inference-opt asks an agent to write a Python policy that improves a frozen
student model. The agent edits `policy/policy.py`, tests it on labelled train
data, and submits it for scoring on held-out test data.

## Components

```text
Corral
  └─ InferenceOptEnvironment
       ├─ workspace tools
       ├─ inference tools
       └─ committed environment state

inference tools
  └─ PolicyEvaluator
       └─ Inspect AI task
            └─ StudentClient and benchmark scorer
```

Each task binds one benchmark and one or two student models. The task definition
contains immutable configuration, the teacher prompt, the workspace setup, and
the final scorer. It does not contain session history.

## Corral state

`InferenceOptEnvironment` is stateless. Corral passes the current
`ExecutionState` to each tool call. Inference-opt copies the hidden
`inference_state` mapping, passes the copy to the tool, and returns the updated
mapping in `ToolExecutionResult.environment`.

Corral commits that environment update with the tool result. This makes the
session ledger available after replay, restart, and task branching.

The session state is JSON-shaped:

```json
{
  "experiments": 0,
  "debug_runs": 0,
  "student_calls": 0,
  "reveals": 0,
  "probe_calls": 0,
  "revealed_ids": [],
  "runs": [],
  "best_run_id": null,
  "limits": {
    "max_experiments": 20,
    "max_debug_runs": 10,
    "max_student_calls": 250,
    "max_reveals": 4,
    "max_probe_calls": 30,
    "reveal_batch": 5
  }
}
```

`StateLedger` in `budget.py` provides the budget and run-record operations used
by the tools. The mapping in Corral state remains the source of truth.

## Workspace

The workspace holds source and detailed artifacts:

```text
policy/policy.py
guide/policy_api.md
notes.md
TODO.md
revealed/train_revealed.jsonl
runs/<run-id>/
  questions.jsonl
  predictions.jsonl
  summary.json
  log/
submission.json
```

The ledger stays in Corral state. Files hold the policy, revealed examples, and
run output because those artifacts are larger and useful to inspect directly.

## Tool flow

1. Corral materializes `ExecutionState` and injects hidden arguments.
2. `InferenceOptEnvironment.execute_tool()` copies `inference_state`.
3. The trusted tool reads or updates the copy and writes any workspace artifacts.
4. The environment returns the tool result and updated environment namespace.
5. Corral commits both as one transition.

The domain tools are trusted because they read private labels and run the
submitted policy. Policy code is also trusted inside the task container. Docker
is the isolation boundary; the evaluator does not create a second sandbox.

## Policy evaluation

`evaluate_candidate()` writes the train questions with hidden targets to a
task-local file, then runs `PolicyEvaluator`. The evaluator:

- loads the policy;
- creates Inspect samples and scorers;
- runs questions sequentially;
- meters calls through `StudentClient`;
- writes predictions, summaries, and Inspect logs.

The tool stores a compact `RunRecord` in the session ledger and keeps detailed
output under `runs/`. `dry_run_policy()` uses the same evaluator on a smaller set.

Final scoring runs the submitted policy on the held-out split and subtracts the
stored zero-shot baseline. Level-2 tasks score the smaller improvement across
the two student models.

Inspect handles model execution, sample execution, answer parsing, grading, and
logs. Inference-opt supplies the policy solver, policy API, client metering, and
task-specific data preparation.

## Runtime limits

The environment currently has no persistent policy process, background policy
jobs, file-backed active ledger, or restricted policy worker. Policy questions
run sequentially so shared memory, budgets, and artifacts are deterministic.

## Source map

| Responsibility | Source |
| --- | --- |
| Task and environment factory | `inference_opt/env.py` |
| Teacher tools and ledger updates | `inference_opt/tools.py`, `inference_opt/budget.py` |
| Policy contract and loading | `inference_opt/api.py`, `inference_opt/policy.py` |
| Inspect adapter | `inference_opt/eval_runner/` |
| Final scoring | `inference_opt/score.py` |
| Frozen data | `inference_opt/datasets.py` |
| Environment base class | `src/corral/core/environment.py` |
| State projection | `src/corral/core/state.py` |
| Tool transition | `src/corral/core/transition.py` |
