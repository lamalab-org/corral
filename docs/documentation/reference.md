# Corral API reference

Corral's execution API is built around authored commits, materialized execution projections, and
Temporal orchestration. There is no benchmark HTTP server or router in the
runtime path.

## `BenchmarkTaskMetadata`

Describes one benchmark task using durable worker registry IDs. Most callers do
not need to construct it: `CorralRunner` infers it from an environment mapping.
Construct it explicitly only for per-task worker IDs, queues, models, or
budgets:

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

The default construction path infers task metadata and dependencies:

```python
runner = CorralRunner(
    executor,
    environments=environments,
    agent_id="tool-calling",
    model="openai/gpt-4o",
    max_iterations=10,
    state_store=state_store,
)
```

```python
result = await runner.run(
    "benchmark-run-id",
    task_ids=["task-b"],  # task-b's dependencies are included automatically
    trials_per_task=3,
    k_values=[1, 2, 3],
    max_parallel=8,
    max_parallel_per_task=2,
)
```

Set `include_dependencies=False` for strict validation instead of automatic
dependency expansion.

Pass `tasks={...}` with explicit `BenchmarkTaskMetadata` instead when the
automatic one-environment-ID-per-task convention is not suitable:

```python
runner = CorralRunner(executor, tasks=task_metadata, state_store=state_store)
```

`ActivityPolicy` defaults both `start_to_close_seconds` and
`heartbeat_timeout_seconds` to `None`. Temporal requires one closing timeout at
the protocol level, so Corral represents an unset Start-to-Close deadline with
a 100-year Schedule-to-Close timeout. Explicit positive values retain the
normal Temporal timeout behavior.

The runner has no synchronous execution path, checkpoint scheduler, HTTP
router, or tool-verbosity parameter. Reporting records the fixed framework
default.

## `TaskRuntime`

`TaskRuntime.run(...)` starts from the execution's durable branch head. It
appends typed configuration, agent, action, tool, submission, and terminal
events. A restored projection containing a pending Action resumes its stable
invocation before the agent loop continues.

## Commit store

`SQLiteCommitStore` is the authoritative persistence implementation. Its core
API is `append`, `head`, `iter_commits`, `materialize`, `get_commit`, and
`create_branch`. Agent and tool code obtains a trusted append capability with
`bind(actor, branch_id=...)`; runtime code cannot supply ordering or hash fields.

## Temporal executors

- `TemporalTaskExecutor.execute(TaskWorkflowInput)` executes one task attempt.
- `TemporalBenchmarkExecutor.execute(BenchmarkWorkflowInput)` executes a task
  DAG and its configured repetitions.
- `execute_task(executor, task)` is the thin standalone task helper.

Agents and environments are registered on workers with `RuntimeRegistry`.
Only serializable IDs and commit-backed projection references enter Workflow history.

## Core transition API

The low-level transition boundary produces events and effects:

```python
proposal_event = propose_action(action)
# append proposal_event, then materialize through that commit
effects = execute_action(environment, execution_state, action)
# append one atomic ToolCompleted event containing effects
```

Every agent implements `run_session(AgentSession) -> AgentOutcome` and executes
actions through the session. `submit_answer` is present in every direct and MCP
tool catalog and is the only valid terminal action. Returning a completed
outcome without executing it is a protocol failure. Evaluation is performed
separately by a `Scorer` after execution.

`AgentSession.run_delegate(agent, max_iterations=remaining)` applies that same
lifecycle to a composite delegate sharing the current identity and context,
scopes it to the outer scaffold's remaining task budget, and does not fold
usage a second time.
`AgentSession.spawn_subagent(...)` creates an isolated child identity and context
on the same linear branch. The parent sees an ordinary assistant tool call and
an automatic tool-style completion summary, just as it does for a regular tool.
Parents can additionally use `inspect_subagent()` to read the authorized private
descendant trace and `import_subagent_context()` to add selected trace details
to future model context. `AgentSession.fork_branch()` is reserved for a true
experimental history; there is no implicit branch adoption.

Inspection is model-facing only when the agent opts in:

```python
from corral.agents import AgentSessionCapabilities


class ResearchManager:
    session_capabilities = AgentSessionCapabilities(inspect_subagents=True)
```

The session then adds `inspect_subagent(child_run_id, include_commits=False,
limit=25)` to that agent's tool catalog. The result is committed and rendered
like any other tool observation. The capability is recorded on the agent run,
so retries reconstruct the same catalog. AI Scientist enables this capability;
other built-in agents do not.

`AgentSession.get_agent_state(namespace)` and
`AgentSession.set_agent_state(namespace, value)` expose JSON-serializable,
namespaced algorithm state backed by `agent.state_updated` commits. With
`previous=True`, `get_agent_state` reads from the explicitly supplied prior
projection. This is the persistence boundary used by AI Scientist search provenance
and Reflexion memory; neither scaffold stores attempt data on its agent object.

## Reporting

`project_benchmark_result` converts `BenchmarkWorkflowResult` into
`BenchmarkResult`. Supplying a `CommitStore` lets the projection include final
messages, token usage, duration, and tool statistics.
