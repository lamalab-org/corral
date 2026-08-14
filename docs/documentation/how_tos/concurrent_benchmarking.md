# How to Run Benchmarks Concurrently (`abench`)

**Goal**: Run many trials in parallel instead of one at a time, under limits you control.

**When to use this**: You have more than a handful of task/trial pairs (e.g. `pass@k` over many tasks) and each trial spends most of its time waiting on an LLM or a tool. Serial `bench()` runs them one after another; the concurrent path overlaps them and can cut wall-clock time dramatically.

`Corral` exposes concurrency through two twin entry points on `CorralRunner`:

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `runner.bench(max_concurrency=...)` | You're in **synchronous** code | Internally drives an event loop via `anyio.run`, so it must **not** be called from inside a running event loop. |
| `await runner.abench(...)` | You're **already inside an event loop** | The async counterpart; pair it with an `AsyncCorralRouter`. |

At `max_concurrency=1` both keep the byte-for-byte serial behaviour, so the serial `bench()` from [Tutorial 1](../tutorials/tut_1_first_benchmark.md) is just the degenerate case of this same API.

## 1. Provide an agent

Action-based agents retain immutable configuration only, so one instance can be
shared safely by concurrent trials:

```python
from corral.agents import ReActAgent

runner = CorralRunner(interface, agent=ReActAgent(model="openai/gpt-4o"))
```

Use an `agent_factory` when the configuration itself should vary by trial. The
factory is a callable `(TrialContext) -> BaseAgent`:

```python
from corral import CorralRunner, TrialContext
from corral.agents import BaseAgent, ReActAgent


def make_agent(context: TrialContext) -> BaseAgent:
    # `context` carries task_id, trial_index, session_id, and benchmark_run_id.
    return ReActAgent(model="openai/gpt-4o")


runner = CorralRunner(interface, agent_factory=make_agent)
```

You may also pass a shared `agent=` alongside the factory. It is used for run
metadata and report labeling while the factory supplies each trial's policy.

## 2. Run it — synchronous

```python
result = runner.bench(
    task_ids=["task_1", "task_2", "task_3"],
    trials_per_task=3,
    k_values=[1, 2, 3],
    max_concurrency=5,  # up to 5 trials in flight across the whole run
    run_name="react_gpt4o",
)

print(f"Average Score: {result.average_score():.2f}")
print(f"Pass@1:        {result.pass_at_k(1):.2f}")
result.generate_report("results.json")
```

Independent tasks overlap directly; a single collector records every trial and checkpoints deterministically, so resuming an interrupted run skips already-recorded trials.

## 3. Run it — asynchronous

Inside an event loop, use `abench` directly with an `AsyncCorralRouter`:

```python
import anyio
from corral import AsyncCorralRouter, CorralRunner, TrialContext
from corral.agents import BaseAgent, ClaudeCodeAgent


async def main() -> None:
    interface = AsyncCorralRouter("http://localhost:8000")

    def make_agent(_context: TrialContext) -> BaseAgent:
        return ClaudeCodeAgent(model="claude-sonnet-4-5")

    runner = CorralRunner(interface, agent_factory=make_agent)

    result = await runner.abench(
        task_ids=await interface.get_available_tasks(),
        trials_per_task=3,
        k_values=[1, 2, 3],
        max_concurrency=5,
        run_name="claude_code_spectra",
    )
    print(f"Overall score: {result.total_score:.3f}")


anyio.run(main)
```

## 4. Overlap repeated trials of the *same* task

By default, repeated trials of one task stay **serialised** (they'd otherwise fight over the same mutable server-side environment). To overlap them too, raise `max_concurrency_per_task`:

```python
runner.bench(
    task_ids=["task_1"],
    trials_per_task=4,
    max_concurrency=4,
    max_concurrency_per_task=2,  # two trials of task_1 at once
)
```

Values above `1` require the environment server to support **trial runtimes** (isolated per-trial environments); the runner then routes each trial through its own runtime so repeated trials no longer share state. If the server doesn't support it, the run fails fast with a clear error.

**Envs that keep process-global state are isolated (or serialised) automatically.** Trial runtimes isolate per-trial *Python state* (a fresh `CorralState` and workspace), but they can't isolate **process-global** state — module globals or a stateful native library shared by the whole interpreter. Each env declares its safety through `Environment.concurrency`:

- `"thread"` *(default)* — no such state; trials overlap freely in-process.
- `"process"` — keeps process-global state (e.g. wetlab's reaktoro system lives in module globals). If the environment server was launched with a **process-worker pool** (its `run_server` was given a `build_envs` builder), each such trial runs in its **own worker process**, so independent trials overlap up to the pool size. Without a pool, they fall back to running **one at a time across the whole benchmark**.
- `"serial"` — the explicit escape hatch: never overlaps anything, even with a worker pool.

You don't set this per run — it ships with the env. It's a correctness guardrail: a stateful env stays correct under a concurrent run, and gets the speed-up once its server runs a worker pool.

> Chained tasks are an exception: a dependency chain pins all of an episode's nodes to one worker (to share their dependency outputs), so `"process"` nodes in a chain stay serialised even with a pool.

## 5. Fine-grained limits with `ConcurrencyConfig`

The two scalars above are shorthands. For the full set of levers, pass a [`ConcurrencyConfig`](../reference.md):

```python
from corral import ConcurrencyConfig, CorralRunner

config = ConcurrencyConfig(
    global_trials=16,  # == max_concurrency
    per_task=4,  # == max_concurrency_per_task
    per_model={"claude-opus": 8, "gpt-5.6": 16},  # per-model rate-limit cap
    tool_jobs_per_trial=6,  # background jobs within one trial
    configure_apps=2,  # simultaneous app-configure steps
)

# Set it once on the runner...
runner = CorralRunner(interface, agent_factory=make_agent, concurrency=config)

# ...or pass it per call (this wins over the runner's and over the scalars):
runner.bench(task_ids=[...], trials_per_task=4, concurrency=config)
```

| Field | What it bounds |
|-------|----------------|
| `global_trials` | Max trials executing simultaneously across the whole benchmark. |
| `per_task` | Max simultaneous trials of the **same** task (needs trial-runtime support above `1`). |
| `per_model` | Max simultaneous trials per **agent model** — your provider/rate-limit lever. A model absent from the map is bounded only by `global_trials`. |
| `tool_jobs_per_trial` | Max background jobs running at once **within a single trial runtime**. Defaults to the server's historical value. |
| `configure_apps` | Max trials configuring their additional apps/services at once — the fixed-port / expensive-setup lever. |

**Precedence**: a `concurrency=` passed to `bench`/`abench` beats one set on the `CorralRunner`, which beats the scalar `max_concurrency` / `max_concurrency_per_task` arguments. `per_model` and `configure_apps` are enforced only on the concurrent path (they're meaningless when one trial runs at a time).

## 6. Chained tasks

A dependency chain (see [Task Chaining](task_chaining.md)) runs concurrently as an *episode DAG*: each trial round is an isolated episode whose sibling tasks run in parallel, dependents start once their parents succeed, broken chains propagate as *unreachable*, and rounds overlap up to the global limit — all from the same `bench(max_concurrency=...)` / `abench` call.

**Done**: Your benchmark now runs trials in parallel, bounded by exactly the limits you set, with the same results, checkpoints, and reports as the serial path.
