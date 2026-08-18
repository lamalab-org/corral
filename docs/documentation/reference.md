# Corral API reference

Corral's execution API is built around immutable State transitions and
Temporal orchestration. There is no benchmark HTTP server or router in the
runtime path.

## `BenchmarkTaskMetadata`

Describes one benchmark task using durable worker registry IDs:

```python
BenchmarkTaskMetadata(
    agent_id="react-gpt4o",
    environment_id="samplemath",
    dependencies=("upstream-task",),
    max_iterations=10,
    model="gpt-4o",
    task_queue="corral-cpu",
)
```

## `CorralRunner`

`CorralRunner` is a metadata adapter. It validates selected task metadata,
builds a `BenchmarkWorkflowInput`, delegates once to a benchmark executor, and
projects the durable result into reporting models.

```python
runner = CorralRunner(executor, tasks, state_store=state_store)
result = await runner.run(
    "benchmark-run-id",
    task_ids=["task-a", "task-b"],
    trials_per_task=3,
    k_values=[1, 2, 3],
    max_parallel=8,
    max_parallel_per_task=2,
)
```

The runner has no synchronous execution path, checkpoint scheduler, HTTP
router, or tool-verbosity parameter. Reporting records the fixed framework
default.

## `TaskRuntime`

`TaskRuntime.run(...)` starts from the execution's durable head. It persists the
deterministic initial State, configured State, both sides of every tool call,
and the completed State. Before/after action transitions have stable IDs, and
the append-only head moves atomically with each canonical snapshot. A restored
head containing a pending Action executes that exact Action before the agent
loop continues.

## Temporal executors

- `TemporalTaskExecutor.execute(TaskWorkflowInput)` executes one task attempt.
- `TemporalBenchmarkExecutor.execute(BenchmarkWorkflowInput)` executes a task
  DAG and its configured repetitions.
- `execute_task(executor, task)` is the thin standalone task helper.

Agents and environments are registered on workers with `RuntimeRegistry`.
Only serializable IDs and State references enter Workflow history.

## Core transition API

The canonical execution sequence is:

```python
proposed = propose_action(state, action)
# persist proposed and advance the execution head
next_state = execute_action(environment, proposed, action)
# persist next_state and advance the execution head
```

Every agent implements `run_session(AgentSession) -> AgentOutcome` and executes
actions through the session. `submit_answer` is present in every direct and MCP
tool catalog and is the only valid terminal action. Returning a completed
outcome without executing it is a protocol failure. Evaluation is performed
separately by a `Scorer` after execution.

`AgentSession.run_delegate(agent, max_iterations=remaining)` applies that same
lifecycle to a child agent on the current State, scopes it to the outer
scaffold's remaining task budget, and does not fold usage a second time.
`AgentSession.fork_branch()` creates an isolated, non-head-advancing speculative
session, and `AgentSession.adopt_branch(branch)` selects its current State for
the owning session's next canonical transition.

`AgentSession.get_agent_state(namespace)` and
`AgentSession.set_agent_state(namespace, value)` expose JSON-serializable,
namespaced algorithm state backed by immutable Corral State revisions. With
`previous=True`, `get_agent_state` reads from the explicitly supplied prior
State. This is the persistence boundary used by AI Scientist search provenance
and Reflexion memory; neither scaffold stores attempt data on its agent object.

## Reporting

`project_benchmark_result` converts `BenchmarkWorkflowResult` into
`BenchmarkResult`. Supplying a `StateStore` lets the projection include final
messages, token usage, duration, and tool statistics.
