# Inference-time optimization environment

This environment asks a teacher agent to write a Python policy that improves a frozen student model at test time. The policy controls the strategy around the model and does not change model weights.

Level 1 evaluates one student per task. Level 2 evaluates one policy against two
students and scores the smaller improvement.
The final score is the accuracy improvement over the measured zero-shot baseline on held-out test questions.

Tasks use the `primitive` policy API by default. Tasks may opt into `enhanced`,
which adds sampling, batching, shared memory, setup, and component diagnostics.

## Run

```bash
cd tasks/inference_opt
uv sync
uv run python -m inference_opt.env --level 1
uv run pytest tests -q
```

Set `CORRAL_VLLM_URL` before starting Corral (or use a student-specific variable such as `CORRAL_VLLM_URL_STUDENT_A`).
The environment binds these values into the task at startup, so probes, dry runs, experiments, and final scoring use the same endpoint.
The committed tasks use placeholder baseline values until `scripts/measure_baselines.py` has been run.
Private labels are supplied separately through `CORRAL_INFERENCE_LABELS_PATH`.

## Agent tools

The domain tools are trusted Corral tools and use Corral's committed environment state. Workspace file tools create and edit the submitted policy.

| tool | purpose |
| --- | --- |
| `get_baseline` | Show the measured train baseline and topic breakdown. |
| `reveal_train_questions` | Spend a reveal to unlock labelled train examples. |
| `query_student` | Probe the student directly under the probe budget. |
| `dry_run_policy` | Run the policy on a small train sample. |
| `evaluate_candidate` | Evaluate and record a full train experiment. |
| `inspect_failures` | Inspect predictions and logs from an earlier run. |
| `compare_runs` | Show recorded experiments and the current best. |
| `get_budget` | Show remaining session budget. |
| `submit_policy` | Stage `submission.json` for final scoring. |

## State and execution

Corral owns the session state. Each trusted tool receives a JSON-shaped
`inference_state` namespace and returns its updates to Corral. No active budget
or run ledger is stored in the workspace filesystem.

Policy evaluation runs sequentially through `PolicyEvaluator` inside the trial.
Inspect AI handles execution and grading; a metered `StudentClient` handles model
access; predictions and Inspect logs are written to the run directory.
Policy code is trusted inside the Docker trial, which is the isolation boundary.
The client is the only supported model interface. There is no background job or
persistent policy REPL.

Running the task outside Docker does not provide the intended safety boundary.

## Source map

- `inference_opt/env.py`: Corral environment and state handoff.
- `inference_opt/tools.py`: trusted teacher-facing tools.
- `inference_opt/budget.py`: local meters and `StateLedger`.
- `inference_opt/policy.py` and `api.py`: policy loading and policy/client contract.
- `inference_opt/eval_runner/`: policy evaluator, Inspect adapter, runtime, and summaries.
- `inference_opt/outcomes.py`: Inspect-log outcome parsing.
- `inference_opt/score.py`: held-out scoring and baseline delta.
- `inference_opt/datasets.py`: frozen questions and evaluator-only targets.
