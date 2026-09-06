# How to Run Benchmarks Concurrently

`CorralRunner` schedules trials with `asyncio` and bounds task execution and
evaluation with the configured concurrency limits.

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

The following limits apply throughout the benchmark:

| Argument | What it bounds |
|---|---|
| `max_parallel` | All running task attempts in the benchmark. |
| `max_parallel_per_task` | Attempts of the same task. |
| `max_parallel_by_model` | Attempts using each model ID. |
| `max_parallel_by_environment` | Attempts using each environment ID. |

These benchmark limits are the concurrency controls. Corral does
not serialize requests that share an `agent_id`; a registered agent instance
must be reentrant and keep task-specific scratch in `AgentSession` or local
variables.

Per-task model and environment IDs belong to task metadata:

```python
from corral import BenchmarkTaskMetadata

tasks = {
    "wetlab_1": BenchmarkTaskMetadata(
        agent_id="react-gpt5",
        environment_id="wetlab",
        model="gpt-5",
        max_iterations=20,
    )
}
```

Dependencies are metadata too. The runner makes a selected task set
dependency-closed automatically before starting trials. The runner waits
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

Task retries and cancellation are handled in the runner. Each task's commit
history contains typed setup, action, tool-effect, and terminal commits. The
execution head advances atomically, so a retry restores the latest projection
and finishes an already-proposed Action before asking the agent for a new
decision. Benchmark scheduling lives in the invoking process; task state is
persisted in the commit store.
