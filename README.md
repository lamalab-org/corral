# Corral: Scientific Agent Benchmark

<p align="center">
    <a href="https://github.com/lamalab-org/corral/actions/workflows/tests.yaml">
        <img alt="Tests" src="https://github.com/lamalab-org/corral/actions/workflows/tests.yaml/badge.svg" />
    </a>
    <a href="https://pypi.org/project/corral">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/corral" />
    </a>
    <a href="https://github.com/lamalab-org/corral/blob/main/LICENSE.md">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/corral" />
    </a>
    <a href='https://lamalab-org.github.io/corral/'>
        <img src="https://github.com/lamalab-org/corral/actions/workflows/docs.yaml/badge.svg" alt='Documentation Status' />
    </a>
    <a href="https://github.com/lamalab-org/corral/blob/main/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/corral_logo_final.png">
  <img alt="Corral logo" src="docs/_static/definitive.png" width='300px'>
</picture>
</p>

A comprehensive benchmarking framework for evaluating AI agents on science tasks. The system provides standardized environments, tools, and evaluation metrics to test agent performance across diverse materials science challenges.

## 🚀 Quick Start: Run One Task from One Environment

The shortest useful Corral run does not need `CorralRunner`, trials, scoring,
or report generation. `corral run` sends one independent task through a direct
Temporal task Workflow and starts its Corral worker in the same process.

You need Python 3.11 or newer for SampleMath, [`uv`](https://docs.astral.sh/uv/),
a model API key such as `OPENAI_API_KEY`, and the Temporal CLI. Clone the
repository and install SampleMath together with the framework:

```bash
git clone https://github.com/lamalab-org/corral.git
cd corral/tasks/samplemath
uv sync
```

Start a local Temporal service in one terminal:

```bash
temporal server start-dev
```

In another terminal, run one task with one agent and model:

```bash
uv run corral run \
  --agent tool-calling \
  --environment samplemath \
  --task task1 \
  --model openai/gpt-5.6
```

This path always uses `evaluate=False`: it prints the final status, submitted
answer, and commit hash, and writes the authored commit ledger to
`.corral/run-commits.sqlite3`. It does not create trials, invoke a scorer, or
generate a benchmark report. Because it executes exactly one task, it rejects a
task with upstream dependencies and points to `corral bench` instead.

Keep the command running while its in-process worker executes the task. Temporal
retains the durable Workflow history; re-running with the same `--execution-id`,
task queue, and commit database attaches to that execution with a new in-process
worker.

The equivalent Python API is below. Save this as `quickstart.py` in the current
`tasks/samplemath` directory:

```python
import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from temporalio.client import Client

from corral import (
    CorralActivities,
    RuntimeRegistry,
    TaskWorkflowInput,
    TemporalTaskExecutor,
    create_worker,
    execute_task,
    load_environment_group,
)
from corral.agents import ToolCallingAgent
from corral.persistence import SQLiteCommitStore

AGENT_ID = "tool-calling"
TASK_ID = "task1"
TASK_QUEUE = "corral-quickstart"
MODEL = "openai/gpt-5.6"


async def main():
    load_dotenv()
    environment = load_environment_group("samplemath")[TASK_ID]
    store = SQLiteCommitStore(".corral/quickstart-commits.sqlite3")
    registry = RuntimeRegistry(
        agents={AGENT_ID: ToolCallingAgent(model=MODEL)},
        environments={TASK_ID: environment},
    )

    try:
        client = await Client.connect("localhost:7233")
        async with create_worker(
            client,
            task_queue=TASK_QUEUE,
            activities=CorralActivities(store, registry),
        ):
            state = await execute_task(
                executor=TemporalTaskExecutor(
                    client,
                    state_store=store,
                    task_queue=TASK_QUEUE,
                ),
                task=TaskWorkflowInput(
                    execution_id=f"samplemath-task1-{uuid4().hex}",
                    task_id=TASK_ID,
                    environment_id=TASK_ID,
                    agent_id=AGENT_ID,
                    model=MODEL,
                    max_iterations=10,
                    evaluate=False,
                ),
            )
    finally:
        registry.close()
        store.close()

    print(f"status: {state.runtime.status}")
    print(f"answer: {state.submission}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it with the current SampleMath virtual environment:

```bash
uv run python quickstart.py
```

`evaluate=False` is explicit here: the result is the agent's final immutable
`ExecutionState` projection, and the example prints only its status and submitted answer. No scorer,
aggregate metric, or benchmark report runs. `task1` is independent; use
`corral bench` for a task such as `task4` whose inputs come from earlier tasks.

## 📊 Running Benchmarks

### Run any agent with the `corral` CLI

The installed `corral` command can run every public concrete agent against a
registered environment. Start a local Temporal service first:

```bash
temporal server start-dev
```

Each task environment is a separate Python project because it has its own
dependencies. Change into that project once, then `uv` automatically uses its
environment and no `--project` option is needed:

```bash
cd tasks/samplemath
uv sync
uv run corral bench \
  --agent react \
  --environment samplemath \
  --model openai/gpt-4o \
  --task task4 \
  --trials 3 \
  --report .corral/samplemath-react-report.json
```

`corral bench` calls the same `CorralRunner` and
`TemporalBenchmarkExecutor` used by Python callers. It hosts the Temporal worker,
worker-side registry, and Activities in the CLI process for the duration of the
benchmark.

If that environment is already activated, `uv run` is optional and the command
is simply `corral bench ...`. The `--project` form is only needed when invoking
`uv` from elsewhere in the monorepo.

The available agent names are `ai-scientist`, `claude-code`, `codex`,
`llm-planner`, `openhands`, `react`, `reflexion`, `terminus`, and
`tool-calling`. Class names and underscore spellings such as `ReActAgent` and
`tool_calling` are accepted as aliases. Claude Code, Codex, and OpenHands need
their corresponding optional package extra and report an installation hint if
it is missing.

Use `--agent-kwargs` for constructor-specific settings. Where the agent supports
them, explicit `--model`, `--api-endpoint`, and `--temperature` options take
precedence over the same keys in that JSON object:

```bash
uv run corral bench \
  --agent codex --environment samplemath --task task1 \
  --model gpt-5.4 \
  --agent-kwargs '{"reasoning_effort": "high"}'
```

`ReflexionAgent` wraps `ToolCallingAgent` by default. Select another actor with
`{"actor": "react", "actor_kwargs": {...}}`. For `AIScientistAgent`, a JSON
`config` object is validated as `AIScientistConfig`; add
`"config_profile": "sakana"` to use `SakanaAIScientistConfig`.

### Compatibility `ToolCallingAgent` script

[`run_scripts/run_tool_calling.py`](run_scripts/run_tool_calling.py) delegates to
the same local benchmark execution function as `corral bench`; it does not maintain
a separate runner implementation. The runner infers `BenchmarkTaskMetadata`
and includes the selected task's dependencies automatically. Its existing
options, defaults, and invocation remain supported:

From the repository root:

```bash
uv sync --project tasks/samplemath
uv run --project tasks/samplemath python run_scripts/run_tool_calling.py \
  --environment samplemath --list-tasks

uv run --project tasks/samplemath python run_scripts/run_tool_calling.py \
  --environment samplemath --task task4 --model openai/gpt-4o \
  --report .corral/samplemath-report.json
```

The built-in presets are `afm`, `catalyst`, `corral_md`, `ml`,
`resistor_network`, `retrosynthesis`, `samplemath`, `spectra_elucidation`, and
`wetlab`. These are fixed choices for `--environment`. Use `--env-kwargs` for
environment-specific configuration, including the common `level`, `subtasks`,
`task_config`, and `work_dir` keys. Environment dependencies and credentials
still need to be configured as described in each task package's README.

Omit `--task` to run every task returned by the environment. Pass `--task`
multiple times to select specific tasks; their required dependencies are
included automatically.

Activity and heartbeat timeouts are disabled by default. Set either one in
seconds when a deployment needs a deadline or liveness detection, for example
`--activity-timeout 1800 --heartbeat-timeout 30`; both options also accept
`none` explicitly. Activities emit heartbeats every five seconds while long
operations are running.

For example, select SampleMath's subtask set with one environment argument:

```bash
python run_scripts/run_tool_calling.py --environment samplemath \
  --env-kwargs '{"subtasks": true}'
```

### Scored, Multi-Trial Benchmarks

`CorralRunner` is the higher-level convenience layer for repeated trials,
evaluation, metric calculation, and reports. Use it after the direct execution
path above when those benchmark features are actually needed. Given the
`client`, `store`, and all-task worker from the task-group example:

```python
from corral import CorralRunner, TemporalBenchmarkExecutor

environments = load_environment_group("samplemath")
runner = CorralRunner(
    TemporalBenchmarkExecutor(client, task_queue=TASK_QUEUE),
    environments=environments,
    agent_id=AGENT_ID,
    model=MODEL,
    max_iterations=10,
    state_store=store,
)
result = await runner.run(
    "samplemath-run-2",
    task_ids=["task1", "task2"],
    trials_per_task=3,
    k_values=[1, 2, 3],
    max_parallel=4,
    max_parallel_per_task=2,
)
```

Temporal owns concurrency, task-level retries, task-DAG readiness, and durable
progress. Tool verbosity is fixed to Corral's default (`brief`) on this path.

## 🏗️ Available Environments

The framework includes several pre-built environments:

| Environment | Description |
|-------------|-------------|
| `samplemath` | Basic mathematical operations |
| `spectra_elucidation` | Spectroscopy/NMR spectra elucidation tasks |
| `corral_md` | LAMMPS molecular dynamics simulation setup |
| `catalyst` | Catalysis research and material design tasks |
| `afm` | Atomic force microscopy image analysis |
| `ml` | Machine learning model training and evaluation |

## 🤖 Available Agents

The framework includes several built-in agent types:

### AIScientistAgent

Uses progressive tree search to formulate hypotheses, run experiments, refine
results, and verify conclusions. It is based on Sakana AI's original
[AI Scientist v2 implementation](https://github.com/SakanaAI/AI-Scientist-v2).

```python
from corral.agents import AIScientistAgent

agent = AIScientistAgent(
    model="gpt-4o",
    evaluator_model="gpt-4o",
)
```

### ReActAgent

Uses the ReAct (Reasoning and Acting) framework for step-by-step problem solving.

```python
from corral.agents import ReActAgent

agent = ReActAgent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022" or any other model litellm supports
    temperature=0.1,
)
```

### ToolCallingAgent

Uses native function calling from LLM providers to solve tasks by leveraging built-in tool/function calling capabilities.

```python
from corral.agents import ToolCallingAgent

agent = ToolCallingAgent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022" or any other model LiteLLM supports
    temperature=0.0,
)
```

### LLMPlanner

Uses hierarchical planning with high-level planning and low-level execution delegation to other agents.

```python
from corral.agents import LLMPlanner

agent = LLMPlanner(model="gpt-4o", temperature=0.1)
```

### ReflexionAgent

Implements the Reflexion architecture ([paper](https://arxiv.org/abs/2303.11366)) which adds self-reflection and learning from mistakes.

```python
from corral.agents import ReActAgent, ReflexionAgent, ToolCallingAgent

# Create base agent (the "Actor")
base_agent = ToolCallingAgent(model="gpt-4o", temperature=0.1)

# Wrap with Reflexion capabilities
reflexion_agent = ReflexionAgent(
    actor=base_agent,
    reflection_model="gpt-4o",  # Model for generating reflections
    reflection_temperature=0.0,  # Deterministic reflections
)
```

## 💾 Commit Ledger

Corral persists one small, immutable, typed commit for every durable event in a
SQLite ledger. `parent_hash` links each ordinary commit to the current branch
head, while `based_on_hash` records the projection the author actually saw.
`ExecutionState` is rebuilt from the ledger plus occasional replay snapshots;
agents receive an authorized `AgentContext`, not the complete execution trace.

Agent and tool authors are bound capabilities. Subagents share the same linear
branch and appear in the parent conversation as ordinary tool calls with
automatic result summaries. Their full conversations and state remain private
unless the parent chooses to inspect and import selected details. Parallel tool
completions are stored in real completion order and presented to the model in
declared action order. Shared environment and workspace effects carry revision
preconditions, so overlapping writes conflict instead of silently rebasing.
Explicit experiments use `SQLiteCommitStore.create_branch()`; spawning a
subagent never creates a branch.

Subagent inspection is opt-in per agent. Agents declaring
`AgentSessionCapabilities(inspect_subagents=True)` receive an
`inspect_subagent` tool that returns a bounded authorized child context through
the normal tool-call ledger path. AI Scientist enables it; other built-in
agents do not expose it.

## 🔧 Contributing

### Adding a New Environment

1. **Create environment directory**

   ```bash
   mkdir -p tasks/my_new_env/my_new_env
   cd tasks/my_new_env
   ```

2. **Create pyproject.toml**

   ```toml
   [project]
   name = "my_new_env"
   version = "0.1.0"
   dependencies = [
       "corral",
       # Add your specific dependencies
   ]
   ```

3. **Create tools**

   ```python
   # tasks/my_new_env/my_new_env/tools.py
   from corral.core.tool import tool


   @tool
   def my_custom_tool(input_param: str) -> str:
       """Description of what the tool does.

       Args:
           input_param: Description of the parameter

       Returns:
           Description of the return value
       """
       # Your tool implementation
       return f"Processed: {input_param}"
   ```

   Note that the docstring has to be formatted correctly for the tool to be registered properly. This means it has to include a description of the parameters and return values as in the example above.

4. **Define the task and environment**

   ```python
   # tasks/my_new_env/my_new_env/env.py
   from corral.core.environment import Environment, Toolset
   from corral.core.task import TaskDefinition

   from .tools import my_custom_tool

   task = TaskDefinition(
       name="task_1",
       description="Solve this problem: Problem 1",
       tools=["my_custom_tool"],
       scoring_fn=lambda answer: float(answer == "Answer 1"),
       submission_format={"answer": "string"},
       resolve_answer=False,
   )
   environment = Environment(
       "task_1",
       task,
       toolset=Toolset(pool={"my_custom_tool": my_custom_tool}),
   )
   ```

### Adding a New Agent

1. **Create agent file**

   ```python
   # src/corral/agents/my_agent.py
   from corral.agents import AgentOutcome
   from corral.core import submit_answer_action


   class MyAgent:
       model = "gpt-4o"

       async def run_session(self, session):
           result = await session.execute(submit_answer_action("Your final answer"))
           if not result.success:
               return AgentOutcome(
                   status="protocol_failure",
                   error=f"submit_answer failed: {result.error}",
               )
           return AgentOutcome(status="completed", answer="Your final answer")
   ```

2. **Add to agent registry**

   ```python
   # src/corral/agents/__init__.py
   from .my_agent import MyAgent

   __all__ = ["MyAgent", ...]
   ```

3. **Register your agent on the Temporal worker**

   ```python
   from corral.agents.my_agent import MyAgent
   from corral import RuntimeRegistry

   registry = RuntimeRegistry(
       agents={"my-agent": MyAgent()},
       environments={"my-environment": environment},
   )
   ```

### Development Setup

1. **Install development dependencies**

   ```bash
   uv pip install -e .
   ```

2. **Install pre-commit hooks with commitizen commits**

   ```bash
   pre-commit install --hook-type commit-msg --hook-type pre-push
   ```

## 📋 Advanced Usage

### Tool Creation

#### Standard Tools

```python
from corral.core.tool import tool


@tool
def calculate_molecular_weight(formula: str) -> float:
    """Calculate molecular weight from chemical formula.

    Args:
        formula: Chemical formula (e.g., 'H2O', 'CH4')

    Returns:
        Molecular weight in g/mol
    """
    # Implementation here
    pass
```

### Evaluation Metrics

The framework provides comprehensive evaluation metrics:

```python
result = await runner.run(
    "metrics-run-1",
    trials_per_task=10,
    k_values=[1, 3, 5],
)

metrics = result.calculate_metrics()
print(metrics["average_score"])
print(metrics["pass_at_1"])

# Per-task analysis
for task_id, task_result in result.task_results.items():
    print(task_id, [trial.score for trial in task_result.trials])
```

## 🤝 Community

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/lamalab-org/corral/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/lamalab-org/corral/discussions)
- **Contributing**: See our [Contributing Guide](CONTRIBUTING.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Citation

If you use Corral in your research, please consider citing:

```bibtex
@article{ríos-garcía2026ai,
  title   = {AI scientists produce results without reasoning scientifically},
  author  = {Martiño Ríos-García and Nawaf Alampara and Chandan Gupta and Indrajeet Mandal and Sajid Mannan and Ali Asghar Aghajani and N. M. Anoop Krishnan and Kevin Maik Jablonka},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2604.18805}
}
```
