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

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` (recommended) or `pip` for package management
- A Temporal service and Corral worker with the benchmark's agents and
  environments registered by durable ID

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/lamalab-org/corral.git
   cd corral
   ```

2. **Install the framework**

   ```bash
   uv pip install -e .
   ```

### Quick Start

Agents and environments are registered on Temporal workers. The client-side
runner contains only their IDs and benchmark scheduling metadata:

```python
from temporalio.client import Client

from corral import BenchmarkTaskMetadata, CorralRunner, TemporalBenchmarkExecutor
from corral.persistence import JSONLStateStore


async def run_benchmark():
    client = await Client.connect("localhost:7233")
    store = JSONLStateStore(".corral/states.jsonl")
    executor = TemporalBenchmarkExecutor(client, task_queue="corral")
    runner = CorralRunner(
        executor,
        {
            "math_1": BenchmarkTaskMetadata(
                agent_id="react-gpt4o",
                environment_id="samplemath",
                model="gpt-4o",
                max_iterations=10,
            )
        },
        state_store=store,
    )

    return await runner.run("samplemath-run-1")
```

## 📊 Running Benchmarks

### Selecting Tasks and Trials

```python
result = await runner.run(
    "samplemath-run-2",
    task_ids=["math_1", "math_2"],
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

Uses progressive tree search over Corral tool experiments. It formulates
hypotheses, runs preliminary and discriminating investigations, optionally tunes
experimental parameters, verifies conclusions, and synthesizes a submit-ready
answer from a global evidence journal. Each tree node is one bounded scientific
experiment: an experiment worker chooses one tool action, observes its result,
and then chooses the next action without branching the tree between tool calls.

```python
from corral.agents import AIScientistAgent, AIScientistConfig

agent = AIScientistAgent(
    model="gpt-4o",
    evaluator_model="gpt-4o",
    config=AIScientistConfig(
        max_tool_calls=24,
        max_actions_per_node=3,
        max_children_per_node=3,
        tree_exploration_weight=0.1,
        max_llm_tokens=200_000,
    ),
)
```

The default configuration now relies on its per-stage search budgets (18
ordinary nodes in total) instead of a three-node global cap. Search nodes and
stage-boundary validation nodes have independent optional caps via
`max_search_nodes` and `max_validation_nodes`. A failed preliminary stage is a
hard gate, so later research never
builds on a non-working baseline.

Successful internal checkpoints remain expandable until their child cap is
reached, allowing several alternative refinements instead of only one chain.
Each main stage has its own search scope and is seeded explicitly by the
previous stage's listwise-selected winner. Tuning and research continue until a
new checkpoint beats that seed or their budget is exhausted; the manager can
create a new evidence-dependent substage only after the critic confirms that
the current agenda's observable criteria have been met. Tuning experiments all
derive from the Stage-1 winner and verification experiments all derive from the
Stage-3 winner, except continuations/debugs that repair one experiment. Winning
checkpoints are independently replicated and aggregated at stage boundaries,
while Stage 4 remains focused on ablations and optional counterfactual tests.

When tools expose a declared task-internal scalar, the manager parses its value
and provenance from actual observations and uses it as the objective ranking
anchor; it never reads the benchmark score. Non-continuation children use the
`auto` trial-state strategy by default: clone-capable environments inherit a
checkpoint, while other environments start clean. Set
`trial_state_inheritance="replay"` only when a stateful environment requires
inheritance but cannot clone trials. If a local trial produces plot/image artifacts,
the evaluator sends them to a multimodal-capable model gateway and records its
visual feedback in the research journal. Boundary repetitions can be disabled
for deterministic or especially costly tasks with
`stage_boundary_replications=0`.

For comparison with AI Scientist v2 rather than the cheaper Corral defaults,
use `SakanaAIScientistConfig`. It supplies the 20/12/12/18 stage budgets,
0.5 debug probability, debug depth 3, and four experiment workers. Stage 1/3
fill each worker batch by selecting parents independently, prefer distinct root
trees before reuse, and use listwise LLM selection with deterministic metric
fallback. The profile also uses fixed-baseline tuning/ablation, runs Stage 3/4
to their full budgets, validates the best result when a non-initial stage
exhausts its budget, allows up to 16 children and 16 adaptive actions per node,
debugs failed leaves as a chain instead of repeatedly branching from an
already-expanded failure, always attempts Stage 2 (including procedural or
experimental tuning), and treats any tool-successful working Stage-1 node as
sufficient to advance without an extra critic-validity gate.

Sakana-profile boundary replication replays the winner's exact realized action
sequence in clean trials rather than asking an LLM to reconstruct it. A
schema-declared `seed` or `random_state` argument is changed to seeds 0, 1, and
2; tools without either field receive independent byte-for-byte logical
replays. Aggregation records deterministic run/success counts and, when all
replications expose the declared scalar objective, its values, mean, sample
standard deviation, standard error, and seeds before the critic interprets the
evidence. The profile does not include AI Scientist's manuscript, citation, or
review pipeline.

Agent constructors contain model/scaffold configuration only. The interaction
budget is configured once per task through
`BenchmarkTaskMetadata(max_iterations=...)` and is supplied to every agent by
its `AgentSession`; agent-level `max_iterations`/`max_turns` arguments are not
supported.

Compound agents use the same session protocol as direct agents. LLMPlanner and
Reflexion run their executor/actor with `AgentSession.run_delegate(...)`, which
applies the normal hook and `submit_answer` lifecycle without double-counting
usage; planning/reflection calls are deducted before the delegate receives the
remaining task interaction budget. AI Scientist creates isolated speculative histories with
`fork_branch(...)` and promotes its selected history with `adopt_branch(...)`;
its tree-search behavior remains distinct while every physical action still
crosses an `AgentSession` boundary.

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

## 💾 Checkpoint System

Corral persists a complete immutable State immediately before and after every
tool call. Each snapshot contains the agent history, namespaced agent state,
environment, workspace manifest, usage, and runtime data, linked to its parent
by a content hash. An append-only execution head identifies the canonical
branch, while AI Scientist may retain speculative sibling branches. If a worker
stops after the before-tool checkpoint, Temporal resumes the exact pending
Action ID instead of asking the model to decide again. The old runner checkpoint
directory and post-processing LLM call are gone.

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
