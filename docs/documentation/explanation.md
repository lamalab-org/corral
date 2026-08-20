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

## Why Temporal owns orchestration

Temporal handles task attempts, retries, cancellation, task-DAG readiness,
bounded parallelism, and benchmark Continue-As-New. Workflow history contains
only serializable specifications and commit-backed projection references. Live
agent objects, environment resources, stores, clients, and secrets remain
worker-side.

## Why evaluation is separate

Execution produces a final `ExecutionState` projection and optional task output.
A scorer evaluates that projection independently and associates its result with
the final commit hash. Benchmark correctness never mutates execution state and
never controls whether downstream task dependencies receive an output.
