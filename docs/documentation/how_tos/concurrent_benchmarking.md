# How to Run Benchmarks Concurrently

Concurrency is part of the Temporal benchmark plan. `CorralRunner` does not
start tasks, create threads, or run an AnyIO scheduler; it builds a
`BenchmarkWorkflowInput` and delegates it to `TemporalBenchmarkExecutor`.

```python
result = await runner.run(
    "benchmark-2026-08-14",
    task_ids=["task_1", "task_2", "task_3"],
    trials_per_task=4,
    k_values=[1, 2, 4],
    max_parallel=16,
    max_parallel_per_task=4,
    max_parallel_by_model={"gpt-5": 8},
    max_parallel_by_environment={"wetlab": 1},
)
```

The limits are serialized into Temporal Workflow history:

| Argument | What it bounds |
|---|---|
| `max_parallel` | All running task attempts in the benchmark. |
| `max_parallel_per_task` | Attempts of the same task. |
| `max_parallel_by_model` | Attempts using each model ID. |
| `max_parallel_by_environment` | Attempts using each environment ID. |

These benchmark limits and the worker's optional
`max_concurrent_activities` setting are the concurrency controls. Corral does
not serialize requests that share an `agent_id`; a registered agent instance
must be reentrant and keep task-specific scratch in `AgentSession` or local
variables.

Per-task queue routing belongs to task metadata:

```python
from corral import BenchmarkTaskMetadata

tasks = {
    "wetlab_1": BenchmarkTaskMetadata(
        agent_id="react-gpt5",
        environment_id="wetlab",
        model="gpt-5",
        task_queue="wetlab-workers",
        max_iterations=20,
    )
}
```

Dependencies are metadata too. The runner makes a selected task set
dependency-closed automatically before starting the Workflow. Temporal waits
for same-round parents, propagates their runtime outputs, and marks descendants
unreachable when a parent produces no valid output. Pass
`include_dependencies=False` to request strict validation instead.

```python
tasks = {
    "prepare": BenchmarkTaskMetadata(
        agent_id="react-gpt5",
        environment_id="preparation",
    ),
    "analyze": BenchmarkTaskMetadata(
        agent_id="react-gpt5",
        environment_id="analysis",
        dependencies=("prepare",),
    ),
}
```

Retries, cancellation, progress tracking, and benchmark-batch Continue-As-New
are handled by Temporal. Each task is one Activity, while its StateStore history
contains task setup, before-tool, after-tool, and task-end checkpoints. The
execution head advances atomically, so a retry resumes the latest canonical
State and finishes an already-proposed Action before asking the agent for a new
decision. There is no local checkpoint directory or synchronous
`bench()`/asynchronous `abench()` split.
