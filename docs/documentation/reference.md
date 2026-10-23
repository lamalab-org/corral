# Corral API reference

Corral's execution API is built around authored commits, materialized execution projections, and
asyncio task scheduling. There is no benchmark HTTP server or router in the
runtime path.

## `BenchmarkTaskMetadata`

Describes one benchmark task using runtime registry IDs. Most callers do
not need to construct it: `CorralRunner` infers it from an environment mapping.
Construct it explicitly only for per-task resource IDs, models, or
budgets:

```python
BenchmarkTaskMetadata(
    agent_id="react-gpt4o",
    environment_id="samplemath",
    dependencies=("upstream-task",),
    max_iterations=10,
    model="gpt-4o",
)
```

## `CorralRunner`

`CorralRunner` validates selected task metadata, schedules trials with bounded
concurrency, and projects their persisted results into reporting models.

The default construction path infers task metadata and dependencies:

```python
runner = CorralRunner(
    registry,
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
    max_parallel_evaluations=2,
    max_parallel_total=10,
    max_parallel_evaluations_by_environment={"wetlab": 1},
)
```

Task execution and evaluation use separate concurrency controls. Task
execution parallelism defaults to `max_parallel=4`. Evaluation parallelism
defaults to `max_parallel`, and the runner waits for all queued
evaluations before returning. Set `max_parallel_evaluations` to override it.
`max_parallel_total` bounds executions and evaluations together and defaults
to the sum of their limits (`2 * max_parallel` with the defaults). Use
`max_parallel_evaluations_by_environment` for additional scorer-specific
limits; environments without an entry have no additional limit.
Sequential attempts of the same task wait for the preceding evaluation so
reflective agents receive its score and state. Opting into
`max_parallel_per_task > 1` allows those attempts to overlap, in which case
only evaluations completed before an attempt starts are available to it.

Set `include_dependencies=False` for strict validation instead of automatic
dependency expansion.

Pass `tasks={...}` with explicit `BenchmarkTaskMetadata` instead when the
automatic one-environment-ID-per-task convention is not suitable:

```python
runner = CorralRunner(registry, tasks=task_metadata, state_store=state_store)
```

`RetryPolicy` defaults to three attempts with exponential backoff. The runner
applies no time limit to task execution or evaluation. The CLI exposes
`--max-attempts`.

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
The store uses SQLAlchemy Core with aiosqlite. Database initialization happens
on first use; use `async with SQLiteCommitStore(...)` or `await store.aclose()`
to release its connections. `ShardedCommitStore` uses the same async cleanup API.

## Task execution

`execute_task(task=RunTaskInput(...), registry=registry, state_store=store)`
runs one task and returns a `StateRef` with its final status, submission, and
commit hash. Load the full projection with
`await store.for_execution(ref.execution_id).materialize(ref.branch_id, ref.commit_hash)`.
Agents and environments are registered with `RuntimeRegistry`. Benchmarks use
the same function for every trial, then evaluate the persisted result.

## Continue a finished trial

`corral continue` imports an exact historical commit into a new, independent
trial. Pass the source trial directory (for example `task-<id>/k-1`), not its
live workspace or the benchmark's top-level output directory:

```bash
corral continue \
  --source-trial .corral/runs/source-run/task-example/k-1 \
  --commit FULL_SHA256_COMMIT_HASH \
  --message "Your previous answer was wrong." \
  --trials 4 \
  --sandbox docker \
  --output-dir .corral/continuations
```

Use `--sandbox local` for local execution. Docker images must use runtime
protocol 5; rebuild older images. The command inherits saved agent and
environment runtime definitions when available. Older trials without these
definitions need explicit `--agent`, `--environment`, and any required
`--agent-kwargs` / `--env-kwargs`. Use `--source-branch` for a non-main branch,
`--source-execution` if the directory has no execution metadata, and
`--source-agent-run` to select an agent when the checkpoint is ambiguous.

The Python API uses the same import and execution path:

```python
from corral.orchestration import ContinuationSource, RunTaskInput, execute_task

source = ContinuationSource(
    trial="/path/to/source/trial",
    commit_hash="<full SHA-256 hash>",
    message="Recheck your last conclusion.",  # optional
)
result = await execute_task(
    task=RunTaskInput(
        execution_id="independent-child",
        task_id=task_id,
        environment_id=task_id,
        agent_id=agent_id,
        continuation=source,
        max_iterations=20,
    ),
    registry=registry,
    state_store=store,
)
```

For several continuations, pass
`continuation_by_task={task_id: source}` and `trials_per_task=4` to
`CorralRunner.run(...)`. Dependency outputs are inherited from the checkpoint;
upstream tasks are not rerun for the continued task. Each child has its own
execution ID, commit database, artifact store, and workspace. It does not depend
on the parent's files after import, and previous sibling evaluations are not
injected into its context.

The new `execution.forked` initialization event contains the historical state
and source identity. The importer verifies and copies referenced artifact blobs
before publishing that event. It reads a private copy of the source database
and WAL rather than opening the original database for writing. Existing commit
hashes and in-trial `create_branch()` behavior are unchanged. Lineage is
available in `state.runtime.metadata["lineage"]` and `StateRef.metadata`, and
therefore in the benchmark report. `launch-request.json` records the original
host-side runtime definitions, including paths before Docker mount translation.

Environment values, consumed resource budgets, workspace files, canonical
conversation messages, and stored agent algorithm state are inherited. The
agent implementation and task/tool definitions must remain compatible. The
optional message is recorded once, after any pending tool results and before
the next agent turn. Native tool-calling/ReAct agents reconstruct their prompts
and canonical history; external SDK adapters use their existing canonical
history import, not a saved opaque provider session.

A pending or running tool action resumes from its committed environment and
workspace in the child; completed tools are not replayed. If a successful
`submit_answer` is awaiting its acceptance event, acceptance is recovered
without another tool call. A checkpoint already submitted or terminated instead
starts a fresh agent lifecycle with the historical messages retained. Setup is
run only if the selected checkpoint precedes task configuration.

`--max-iterations` is the child's new run budget. New usage is counted separately
from cumulative `inherited_usage`; environment resource budgets are not reset.
Psychometrics restores its serialized REPL namespace and NumPy RNG state through
the existing worker, without replaying previous cells.

Live subagents and active background jobs are rejected explicitly: this API
copies durable state, not running processes or external services. Tools with
external side effects need an environment-specific recovery contract before
in-flight replay is safe. Workspace paths should use the portable `/workspace`
namespace or relative paths. Arbitrary host absolute paths, open OS resources,
and external SDK process memory cannot be made portable by copying a checkpoint.
Local mode retains its existing unrestricted execution semantics.

A reproducible integration check needs no model credentials:

```bash
PYTHONPATH=src:tasks/psychometrics/src python -m scripts.check_trial_continuation \
  --output /tmp/continuation-local

PYTHONPATH=src:tasks/psychometrics/src python -m scripts.check_trial_continuation \
  --sandbox docker --image corral-psychometrics:latest \
  --output /tmp/continuation-docker
```

Use a fresh output directory and an image built from this checkout. Both checks
run one psychometrics task and four continuations, verifying earlier file and
REPL state, the intervention message, independent children, and unchanged
source files. They retain all ledgers and snapshots plus `summary.json`.

## Restricted worker limits

Under Docker, agent and tool code runs in a restricted worker, and a worker
hands back two kinds of thing on two separate descriptors. Control replies are
JSON and stay small. Bulk payloads — a `PythonREPL` checkpoint, or anything else
whose size follows the agent's data rather than its output — travel beside them
and are spooled to a controller-private file as they arrive. Each channel has
its own ceiling:

| setting | default | bounds |
| --- | --- | --- |
| `CORRAL_MAX_WORKER_RESPONSE_BYTES` | 64 MiB | a worker's JSON control reply, and the session channel that carries state snapshots |
| `CORRAL_MAX_WORKER_BULK_BYTES` | 1 GiB | one bulk payload, such as a REPL checkpoint |

Both are read from the length prefix ahead of the body, so an oversize payload
is refused before the controller allocates for it. That is what keeps the reply
ceiling a memory-safety limit rather than a budget, and why it can stay low: an
environment that holds a whole dataset in its REPL session is bounded by the
bulk ceiling, which costs disk, not by the reply ceiling, which costs memory.
Neither limit needs raising for an ordinary session; raise the bulk ceiling for
an unusually large one, and lower the reply ceiling on a small host.
`create_python_repl_tool(max_response_bytes=...)` overrides the reply ceiling
for one tool, as `address_space_bytes` does for worker memory.

A tool that runs in a worker can send a bulk payload itself with
`permissions.send_bulk(name, payload)`, and the controller receives it from
`permissions.run_worker_with_bulk(...)`, which returns the JSON result and a
mapping of payload name to bytes. Prefer it to a large return value: a value
returned the ordinary way is JSON-encoded and held whole in controller memory.

Corral's launcher grants a trial container exactly the capabilities the worker
needs — `SETUID`, `SETGID`, `CHOWN`, `DAC_OVERRIDE`, `KILL`, `SYS_ADMIN` and
`SYS_CHROOT` over a `--cap-drop ALL` baseline, with `apparmor=unconfined` so the
trusted bootstrap can build the worker filesystem. Running a trial image by hand
needs the same grants, or the worker fails with `cannot create worker mount
namespace`:

```bash
docker run --rm --cap-add SYS_ADMIN --security-opt apparmor=unconfined \
  -e CORRAL_PERMISSION_TESTS=1 <image> \
  python -m pytest tests/runtime/test_permissions.py
```

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

`project_benchmark_result` converts `BenchmarkExecutionResult` into
`BenchmarkResult`. Supplying a `CommitStore` lets the projection include final
messages, token usage, duration, and tool statistics.
