# Architecture

Corral has three distinct execution values:

```text
Commit log          Immutable source of truth
ExecutionState      Materialized projection at one commit
AgentContext        Authorized view for one agent run
```

## Authored commit ledger

Every durable event is one small typed commit in SQLite. The store assigns the
timestamp, global execution sequence, branch sequence, `parent_hash`, and
content hash atomically. Runtime code submits only a `CommitRequest` containing
an idempotency key, selected branch, `based_on_hash`, bound actor, and event.

Ordinary activity always extends the selected branch head. A stale
`based_on_hash` records that an agent or tool worked from an older observation;
it does not create a fork. Divergence is possible only through the explicit
`SQLiteCommitStore.create_branch()` API.

```text
execution.started
  -> task.configured
  -> agent.started
  -> agent.turn_recorded
  -> tool.started
  -> tool.completed
  -> agent.completed
  -> execution.completed
```

The event reducer is deterministic and reconstructs `ExecutionState` from the
ledger. Occasional complete projections are stored only as replay accelerators.
They are not independent history records and replay never emits observability
events.

## Trusted authors and concurrency

Every commit carries an `ActorRef` for a runtime, agent run, or tool invocation.
Agent and tool commits use author-bound store capabilities, so model-produced
content cannot select its own durable author. The SQLite transaction serializes
concurrent commits and advances the branch head only after author checks,
history checks, event reduction, and shared-state preconditions succeed.

Private events can safely append from stale observations. Environment and
workspace effects carry expected revision numbers; an overlapping shared write
raises a retryable conflict instead of being silently rebased.

One model turn may propose an ordered parallel action group. Each tool gets a
stable invocation ID. Completion commits retain real completion order, while an
agent's next context presents tool results in declared action order.

## Runtime-selected tool transport

`Environment` owns the task's tool implementations, resolves the toolset, and
provides the workspace and execution guards. `AgentSession.tool_catalog` exposes
a validated schema snapshot derived from that environment, together with
applicable runtime/session actions such as submission and subagent inspection.
`session.tools` returns that catalog in provider format; the catalog's
`mcp_tools()` method supplies the MCP representation of the same schemas.

Agents request actions through `AgentSession`, which supplies agent identity,
records commits, and dispatches task-tool execution to the environment. The
session runner in `corral.agents.session` selects and provisions tool access for
every agent invocation:

- `tool_transport = "python"` uses direct `session.execute()` calls without an
  MCP binding. This is the default when an agent omits the declaration.
- `tool_transport = "mcp"` asks the runtime to expose the bound catalog through
  HTTP/MCP. Codex, Claude Code and OpenHands declare this transport.

The runtime supplies a `ToolConnection` through `session.tool_connection`.
An external adapter reads its URL and runs the harness:

```python
mcp_url = session.tool_connection.mcp_url
# Configure the harness with mcp_url and return its AgentOutcome.
```

The adapter owns no listener or server lifecycle. Run adapters through
`TaskRuntime` or `run_agent_session()` so the runtime can supply their connection.
Reading `mcp_url` without a provisioned MCP connection raises an explanatory error.

Each active `TaskRuntime.run()` attempt owns a host that starts one localhost
HTTP/MCP server when its first MCP invocation requests a binding. Python-only
executions allocate no listener or server task. An MCP delegate or subagent can
start the listener later; concurrent first bindings share the same startup.
Failed or cancelled startup releases its resources and permits a later binding
to retry. Already terminal executions follow the recovery path without creating
a host. The host is local to the call, so concurrent executions on one runtime
have independent listeners and state.

The backend separates that server from its agent bindings:

```python
async with open_mcp_host() as host:
    async with host.bind(catalog=catalog, execute=dispatcher) as connection:
        ...
```

`corral.backend.mcp.open_mcp_host()` owns the listener and async server task.
`host.bind()` starts the listener if needed and registers a catalog and authorized
action dispatcher, with no dependency on agent classes. The runtime supplies
`session.execute`, preserving
the ordinary authored commit and environment tool execution path. Each binding
has an unguessable route on the shared host and port; requests resolve directly
to their binding. The transport creates no separate tool implementations.

Transport selection also runs for `run_delegate()` and `spawn_subagent()`.
Delegates sharing a session receive context-local connection details, and their
caller's connection is restored when they return or raise. Subagents and forked
sessions borrow the same host while retaining their own catalogs, identities,
branches and dispatchers. Standalone `run_agent_session()` creates a host when
none is supplied, and nested agents borrow it. Host and connection details are
ephemeral; recovery provisions fresh resources against the saved execution
history, and state snapshots contain only data.

The server runs on the execution's event loop and preserves the application's
signal handlers. Requests inherit the context captured when their binding was
registered, including delegate iteration limits. Synchronous tools continue to
run off the event loop through the session's execution path.

When an invocation exits, the runtime revokes its route and drains accepted
requests before recording agent completion. Other bindings remain usable.
Before-task hooks run before binding registration, so a rejecting hook grants
no tool connection and does not start a listener.
At execution teardown, agents and their descendants finish or are cancelled,
their requests are drained, and the server is stopped and awaited before the
final execution commit. Accepting `submit_answer` does not stop the server: the
HTTP response and the SDK's final processing must finish first. Cancellation
awaits cleanup; a synchronous tool already running in a Python thread must
finish before session resources can be released, with no cleanup timeout that
silently abandons it.

Completion waits for outstanding subagents and propagates unobserved child
failures. A failure already delivered through `wait_for_subagent()` is not raised
again by cleanup, allowing the caller to catch it and complete with a fallback.

Custom adapters previously opening MCP themselves should declare
`tool_transport = "mcp"` and consume `session.tool_connection.mcp_url`.
`ToolConnection` and `ToolResponse` live in `corral.core.tool`.
`AgentSession.close()` is removed; transport cleanup belongs to the runtime.

## Multi-agent context isolation

`ExecutionState` partitions conversations, actions, tool invocations, usage,
and algorithm state by agent run. It has no flat global message list.
`AgentContextResolver` exposes only an agent's own conversation and actions,
tool or subagent results it requested, task context, handoff, and explicitly
imported trace details.

`AgentSession.spawn_subagent()` appends `agent.spawned` at the current branch
tip, registers the child, and runs it with its own `ActorRef` and filtered
context. In the parent's conversation, spawning is rendered exactly like a
tool call and child completion automatically produces the corresponding
tool-style result summary. Spawning is not branching. The child's private
conversation and state remain isolated: a parent may inspect descendants, but
inspection does not change model context. Selected trace details can be added
by `import_subagent_context()`, which appends `context.imported` with source
commit hashes for provenance. Children and siblings cannot inspect a parent's
or each other's private trace.

Model-driven inspection is an opt-in session capability. An agent declaring
`AgentSessionCapabilities(inspect_subagents=True)` receives the general
`inspect_subagent` tool in addition to its environment tools. Calling it follows
the ordinary action/tool-completion path and returns a bounded child context,
optionally including recent commits. Agents without the capability never see
the schema, and the context resolver still enforces ancestor-only access. AI
Scientist opts in by default.

`run_delegate()` remains available for composite scaffolds that deliberately
share one identity and context. Explicit experimental histories use
`fork_branch()`.

## Recovery and terminal behavior

An `agent.turn_recorded` event durably records actions before execution.
`tool.started` records the stable invocation identity; `tool.completed` stores
the observation and all environment/workspace/runtime effects atomically. A
retry materializes the current head, finds pending actions, and resumes them
without asking the model to decide again. Repeating a completed action request
returns its existing idempotent commit.

`submit_answer` is the only canonical answer path and only a root agent run may
submit. The runtime records `submission.accepted`, the agent outcome, and then
`execution.completed`. Ordinary actions cannot run after submission, and no
commit may follow `execution.completed`.

## Observability

Observers receive persisted commits after the transaction succeeds. Langfuse
exports the typed event delta plus commit, author, branch, causal, timing, and
usage metadata; it never receives a complete `ExecutionState`. Event payloads
are redacted before export, idempotent commit hashes are emitted once, and an
observer failure cannot roll back persistence.

## Task and benchmark execution

`execute_task` in `run.py` launches one task. `CorralRunner` schedules trials
with asyncio, applies retries and concurrency limits, and waits for dependency
outputs. Scheduling runs in the invoking process, while authored commits keep
task state persisted independently.

## Why evaluation is separate

Execution produces a final `ExecutionState` projection and optional task output.
A scorer evaluates that projection independently and associates its result with
the final commit hash. Benchmark correctness never mutates execution state and
never controls whether downstream task dependencies receive an output.
