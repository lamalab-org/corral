# Architecture

Corral treats an append-only chain of complete immutable States as the durable
source of truth. The execution head identifies the canonical checkpoint, much
like a Git branch ref identifies its current commit.

```text
initial State -> persist task start
      |
      v
persist configured State
      |
      v
agent proposes Action -> persist before-tool State
      |
      v
environment executes -> persist after-tool State
      |
     ...
      |
      v
completed State -> persist task end
```

Every checkpoint is a full value containing messages, namespaced agent state,
environment data, workspace manifest, usage, runtime data, and dependency
outputs. Each child stores its parent's content hash. Intermediate model-only
edits may be squashed into the next before-tool snapshot, but neither side of a
tool transition is omitted.

The StateStore also keeps an append-only execution head. Canonical task
checkpoints advance it atomically; speculative AI Scientist branches are saved
as sibling histories without replacing it. If the head contains an Action with
no tool observation, a retry executes that exact Action ID and then commits its
observation.

Every agent owns a complete reasoning loop through
`run_session(AgentSession) -> AgentOutcome`. All tool calls execute against the
same State representation and produce the same durable before/after
checkpoints.

Composition does not introduce a second execution API. Plan-then-execute and
reflect-then-act scaffolds invoke their child through
`AgentSession.run_delegate`, so child hooks, typed outcomes, and terminal-tool
validation match a top-level run while the outer scaffold folds combined usage
once. The delegate sees only the task interaction budget left after the outer
planning or reflection calls. Speculative tree search uses
`AgentSession.fork_branch` and
`AgentSession.adopt_branch`; branches retain isolated workspaces and do not move
the canonical head until the selected State is adopted.

Agent instances contain configuration, not attempt state. Compound scaffolds
store algorithm-specific progress under the namespaced
`State.runtime.metadata["agent_state"]` mapping through `AgentSession`.
AI Scientist records its search tree, journal, counters, graph, and branch-tool
statistics there. Reflexion reads the evaluated prior attempt as an explicit
Corral State and writes its bounded verbal memory into the current State. This
makes their shared instances reentrant and keeps search/reflection provenance
out of worker-local Python-object memory.

## Why Temporal owns orchestration

Temporal handles task attempts, retries, cancellation, task-DAG readiness,
bounded parallelism, and benchmark Continue-As-New. Workflow history contains
only serializable specifications and State references. Live agent objects,
environment resources, stores, clients, and secrets remain worker-side.

The same `TaskWorkflow` is used for a benchmark attempt and for standalone
execution. Its task execution is one `run_task` Activity, followed by an
optional `evaluate_task` Activity.

## Why submission is a tool

`submit_answer` is the only terminal answer path. It is represented as an
Action and recorded in the canonical interaction history like every other
tool call. The tool is added to every `AgentSession` catalog, including the MCP
catalog used by native harness agents. A completed outcome without a preceding
successful `submit_answer` call is a protocol failure; the runtime never turns
plain text into a submission afterward.

## Why evaluation is separate

Execution produces a final State and optional task output. A `Scorer`
evaluates that State independently and stores an `EvaluationResult` alongside
reporting data. Benchmark correctness never mutates execution State and never
controls whether downstream task dependencies receive an output.

## Task dependencies

A downstream task becomes ready when each required upstream task has produced
a valid output. Dependency outputs are explicit State data. Task ordering and
unreachable tracking belong to the Temporal benchmark workflow rather than an
in-process runner scheduler.
