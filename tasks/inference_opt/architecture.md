# Inference-opt architecture

Inference-opt asks an agent to write a Python policy that improves a frozen
student model. The agent edits `policy/policy.py`, tests it on labelled train
data, and submits it; the submission runs once on held-out test data.

The rule the design keeps: agent-written code runs only in Corral's restricted
worker (the jail), and gold answers exist only in the controller.

## Components

```text
CONTROLLER (trusted, has the labels)          JAIL (agent code, no network)
  inference tools ── run_policy ──────────────▶ policy host
    │                 (one per student)          imports policy.py
    │                                            policy.run(questions, ctx)
    │   StudentGateway ◀── service channel ────── ctx.student / submit / log
    │     counts calls, stops at the budget,
    │     forwards to vLLM, records answers
    │
    └─ Grader: Inspect, replaying the answers,
       in its own process; its log stays outside the workspace
```

- `host.py` runs in the jail. It gets public questions, the revealed examples,
  and one service channel to the controller (`corral.runtime.service_channel`,
  passed in by `permissions.run_worker(service=...)`). Every `inference_opt`
  module a policy may use is imported before the jail closes.
- `gateway.py` answers the channel. Requests are agent-written, so it checks
  the budget, the question ids, the token ceiling and the allowed client
  methods. The vLLM address and key never enter the jail.
- `grading/` scores the submitted answers with Inspect's scorers. A replay
  solver puts each answer on its sample; the log, which holds the targets, is
  written beside the workspace, never inside it.
- `runner.py` ties these together. Without Docker permission enforcement the
  host runs as a plain subprocess: metering still holds, privacy does not.

## Corral state

The ledger is a Corral resource (`inference_state`, `InferenceLedgerAdapter`
in `env.py`), seeded through `resource_states` and committed after every tool
call. It holds the budget counters, revealed question ids, run records, the
submission count, and the hidden result of the last submission.

## Workspace

```text
policy/policy.py
guide/policy_api.md
notes.md
revealed/train_revealed.jsonl   copy for the agent; runs use the ledger
runs/<run-id>/<model>/          written by the controller, never a target
  predictions.jsonl
  student_calls.jsonl
  summary.json
submission.json                 what was submitted, with its hash
```

Each run folder is cleared at the start of its run, so a restart that reuses a
container never mixes old and new results.

## Scoring

`submit_policy` runs the policy on the test split exactly like an experiment,
computes the pass rule (`score.final_result`), and stores the result in the
ledger. The reply shows nothing about it. Corral's evaluation step calls
`StateScorer`, which only reads that stored result, so no agent code runs on the
host that scores the episode. A task with no submission scores 0; an
environment fault raises `HarnessError` instead of scoring.

## Known limits

- Per-question right/wrong is shown for train runs, so a multiple-choice answer
  marked correct reveals the gold option.
- Without Docker permission enforcement there is no jail; see `runner.py`.

## Source map

| Responsibility | Source |
| --- | --- |
| Task and environment factory, ledger resource | `inference_opt/env.py` |
| Teacher tools | `inference_opt/tools.py` |
| Ledger operations | `inference_opt/budget.py` |
| Policy contract and loading | `inference_opt/api.py`, `inference_opt/policy.py` |
| Jailed policy host | `inference_opt/host.py` |
| Student gateway | `inference_opt/gateway.py` |
| Run orchestration | `inference_opt/runner.py` |
| Grading | `inference_opt/grading/` |
| Pass rule and state scorer | `inference_opt/score.py` |
| Frozen data | `inference_opt/datasets.py` |
| Service channel | `src/corral/runtime/service_channel.py` |
| Restricted worker | `src/corral/runtime/permissions.py` |
