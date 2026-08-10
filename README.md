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

3. **Install specific environment dependencies**

   ```bash
   # create task environments
   cd tasks/samplemath && uv venv && uv pip install -e .  # create an env for running sample math
   # ... repeat for other tasks as needed
   ```

### Quick Start

1. **Start a task environment server**

   ```bash
   cd tasks/samplemath/samplemath
   python env.py  # Starts server on http://localhost:8000
   ```

2. **Run benchmark in another terminal**

   ```python
   from corral import CorralRunner, CorralRouter
   from corral.agents import ReActAgent
   from corral.report import CorralWandbLogger

   # Setup interface
   interface = CorralRouter("http://localhost:8000")
   # Setup the WandB logger
   wandblogger = CorralWandbLogger(
       project="corral",
       group="experiment_group",
       name="run_name",
   )
   # Setup the agent
   agent = ReActAgent(model="gpt-4o", max_iterations=10, temperature=0.1)

   # Run benchmark
   runner = CorralRunner(interface, agent, logger=wandblogger)
   result = runner.bench()

   print(f"Overall score: {result.total_score:.2f}")
   ```

## 📊 Running Benchmarks

### Single Task Execution

```python
from corral import CorralRunner, CorralRouter
from corral.agents import ReActAgent

interface = CorralRouter("http://localhost:8000")
agent = ReActAgent(model="gpt-4o")
runner = CorralRunner(interface, agent)

# Run specific task
result = runner.bench(task_ids=["math_1"])
```

### Multiple Tasks

```python
# Run specific tasks
result = runner.bench(task_ids=["math_1", "math_2", "math_3"])

# Run all available tasks
result = runner.bench()  # Uses all tasks in the environment
```

### Multiple Trials with Different Parameters

```python
# Run multiple trials per task
result = runner.bench(
    task_ids=["math_1", "math_2"],
    trials_per_task=3,
    k_values=[1, 2, 3],  # Evaluate with different k values for pass@k metrics
    tool_verbosity="MINIMAL",  # Options: FULL, MINIMAL, NONE
)

# Evaluate with different k values for pass@k metrics
result = runner.bench(trials_per_task=5, k_values=[1, 2, 3, 4, 5])
```

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
`max_search_nodes` and `max_validation_nodes`; legacy `max_nodes` limits search
nodes only. A failed preliminary stage is a hard gate, so later research never
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

### ReActAgent

Uses the ReAct (Reasoning and Acting) framework for step-by-step problem solving.

```python
from corral.agents import ReActAgent

agent = ReActAgent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022" or any other model litellm supports
    temperature=0.1,
    max_iterations=10,
)
```

### ToolCallingAgent

Uses native function calling from LLM providers to solve tasks by leveraging built-in tool/function calling capabilities.

```python
from corral.agents import ToolCallingAgent

agent = ToolCallingAgent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022" or any other model LiteLLM supports
    temperature=0.0,
    max_iterations=10,
)
```

### LLMPlanner

Uses hierarchical planning with high-level planning and low-level execution delegation to other agents.

```python
from corral.agents import LLMPlanner

agent = LLMPlanner(model="gpt-4o", temperature=0.1, max_iterations=5)
```

### ReflexionAgent

Implements the Reflexion architecture ([paper](https://arxiv.org/abs/2303.11366)) which adds self-reflection and learning from mistakes.

```python
from corral.agents import ReActAgent, ReflexionAgent, ToolCallingAgent

# Create base agent (the "Actor")
base_agent = ToolCallingAgent(model="gpt-4o", max_iterations=10, temperature=0.1)

# Wrap with Reflexion capabilities
reflexion_agent = ReflexionAgent(
    actor=base_agent,
    reflection_model="gpt-4o",  # Model for generating reflections
    reflection_temperature=0.0,  # Deterministic reflections
)

# Use like any other agent
runner = CorralRunner(interface, reflexion_agent)
result = runner.bench(task_ids=["task_1"], trials_per_task=5)
```

## 💾 Checkpoint System

The framework automatically saves checkpoints during benchmark runs.

Checkpoints are automatically searched and loaded when resuming interrupted runs.

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
   from corral.backend.tool import tool


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

4. **Implement environment class**

   ```python
   # tasks/my_new_env/my_new_env/env.py
   from corral.backend import Environment
   from corral.backend.server import create_benchmark_server


   class MyEnvironment(Environment):
       def __init__(self, task_id: str, problem: str, answer: str):
           self.problem = problem
           self.correct_answer = answer
           super().__init__(task_id)

           # Add your tools
           self.add_tool(my_custom_tool)

       def get_task_prompt(self) -> str:
           return f"Solve this problem: {self.problem}"

       def score(self) -> float:
           if self.state.submitted_answer is None:
               return 0.0
           return 1.0 if self.state.submitted_answer == self.correct_answer else 0.0


   # Define your tasks
   environments = {
       "task_1": MyEnvironment("task_1", "Problem 1", "Answer 1"),
       "task_2": MyEnvironment("task_2", "Problem 2", "Answer 2"),
   }

   # Create server
   if __name__ == "__main__":
       app = create_benchmark_server(environments)
       import uvicorn

       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

### Adding a New Agent

1. **Create agent file**

   ```python
   # src/corral/agents/my_agent.py
   from corral.agents import BaseAgent
   from corral import CorralRunner


   class MyAgent(BaseAgent):
       def __init__(self, model: str, **kwargs):
           super().__init__(model, **kwargs)
           # Add your agent-specific initialization

       def run(self, interface: CorralRouter, task_id: str) -> str:
           # Get task information
           guide = interface.get_task_guide(task_id)

           # Your agent logic here
           # Use interface.execute_tool() to call tools

           return "Your final answer"
   ```

2. **Add to agent registry**

   ```python
   # src/corral/agents/__init__.py
   from .my_agent import MyAgent

   __all__ = ["MyAgent", ...]
   ```

3. **Test your agent**

   ```python
   from corral.agents.my_agent import MyAgent
   from corral import CorralRunner, CorralRouter

   agent = MyAgent(model="gpt-4o")
   interface = CorralRouter("http://localhost:8000")
   runner = CorralRunner(interface, agent)

   result = runner.bench()
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
from corral.backend.tool import tool


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

#### [Modal](https://modal.com) Tools (Cloud Execution)

Modal allows you to run computationally intensive tasks in the cloud. See [Modal docs](https://modal.com/docs) for setup.

```python
from corral.utils.modal import modal_tool, MODAL_TOOL_REGISTRY
from modal import App, Image

app = App("my-corral-tools")


@modal_tool(app=app, image=Image.debian_slim().pip_install("rdkit"), memory=1024)
def complex_calculation(data: str) -> str:
    """Run computationally intensive task in the cloud."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(data)
    return f"Molecule has {mol.GetNumAtoms()} atoms"


# Access the tool
tool_instance = MODAL_TOOL_REGISTRY["complex_calculation"]
```

For Corral-specific usage, see the [Modal App Documentation](tasks/corral_md/modal_app/README.md).

### MCP (Model Context Protocol) Integration

Corral tools can be easily converted to MCP format for use with MCP-compatible clients like Claude Desktop:

```python
from corral.backend.tool import tool
from corral.router.verbosity import ToolVerbosity


@tool
def my_scientific_tool(param: str) -> str:
    """Scientific tool description.

    Args:
        param: Parameter description

    Returns:
        Result description
    """
    return f"Result: {param}"


# Convert to MCP format
mcp_definition = my_scientific_tool.to_mcp()

# With specific verbosity level
mcp_brief = my_scientific_tool.to_mcp(verbosity=ToolVerbosity.BRIEF)
```

Create a custom MCP server:

```python
from mcp.server import Server
from mcp.types import Tool as MCPTool
import importlib
import inspect
from corral.backend.tool import Tool

# Load tools from a module
module = importlib.import_module("my_domain.tools")
tools = {name: obj for name, obj in inspect.getmembers(module) if isinstance(obj, Tool)}

# Create MCP server
server = Server("my-corral-tools")


@server.list_tools()
async def list_tools():
    return [MCPTool(**tool.to_mcp()) for tool in tools.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    tool = tools[name]
    # Execute and return results
    ...
```

For more details on creating tools and MCP integration, see the [Tools Documentation](docs/TOOLS_README.md).

### Environment Configuration

For environments requiring file I/O:

```bash
export CORRAL_FS_PROTOCOL=local
export BASE_IO_PATH=/path/to/work/directory
```

### Evaluation Metrics

The framework provides comprehensive evaluation metrics:

```python
result = runner.bench(trials_per_task=10, k_values=[1, 3, 5])

# Access detailed results
print(f"Total score: {result.total_score}")
print(f"Pass@1: {result.pass_at_k[1]}")
print(f"Pass@3: {result.pass_at_k[3]}")
print(f"Average trials: {result.average_trials}")

# Per-task analysis
for task_id, task_result in result.task_results.items():
    print(f"Task {task_id}: {task_result.success_rate:.2f} success rate")
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
