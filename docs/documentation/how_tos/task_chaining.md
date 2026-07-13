# How to Implement Task Chaining

**Goal**: Create a workflow where one task's output feeds into the next task.

**When to use this**: Your problem requires multiple steps where each builds on previous results.

Chained tasks are normal tasks: each dependency is declared at the input-field
level on the task itself (`input_map`), and at runtime the linked environments
share a single run store (`task_runs`) through their `CorralState`. The
dependency graph is always derived from the task definitions — no separate
group object holds runtime results.

## 1. Define dependent tasks

```python
from corral.backend.task import InputRef, TaskDefinition

# Task 1: Retrieve data
retrieve_task = TaskDefinition(
    name="retrieve",
    description="Fetch molecule structure from database",
    tools=["database_query"],
    scoring_fn=lambda x: 1.0 if x else 0.0,
    submission_format={"structure": "string"},
    initial_input={"molecule_id": "mol_123"},
)

# Task 2: Analyze (uses output from Task 1)
analyze_task = TaskDefinition(
    name="analyze",
    description="Analyze the molecule structure",
    tools=["structure_analyzer"],
    scoring_fn=lambda x: 1.0 if "properties" in x else 0.0,
    submission_format={"analysis": "dict"},
    input_map={
        # The task receives an input named "structure", taken from the
        # "answer" key of the "retrieve" task's output
        "structure": InputRef("retrieve", key="answer"),
    },
)

tasks = {"retrieve": retrieve_task, "analyze": analyze_task}
```

`InputRef.key` supports dotted paths into structured outputs, e.g.
`InputRef("retrieve", key="answer.smiles")`.

## 2. Create the linked environments

A single entry point takes the flat task collection and does everything —
validates the graph, derives the grouping (each connected component of the
dependency graph shares one run store) and returns server-ready environments:

```python
from corral.backend.env import build_environments

environments = build_environments(
    tasks,
    base_work_dir="workdir",
    name="workflow",  # optional label for tracing / docs
    available_tools=my_tools,  # dict[str, Tool], resolved by each task's `tools`
)
```

There is no separate authoring path for a single task: it is just the
degenerate case of a one-node component, so `build_environments({"solo": task})`
works too.

## 3. Customize behaviour without subclassing

Custom prompts, per-trial setup and scoring are injected through the
(immutable) definition, not by subclassing the environment. Dependency outputs
are always read through the shared `CorralState`:

```python
from corral.backend.env import Environment
from corral.backend.task import InputRef, TaskDefinition


def analyze_prompt(env: Environment) -> str:
    task = env.current_task
    prompt = task.description
    for name, ref in task.input_map.items():
        if env.state.is_completed(ref.task_id):
            prompt += f"\n{name}: {env.state.get_output(ref.task_id, ref.key)}"
    return prompt


analyze_task = TaskDefinition(
    name="analyze",
    description="Analyze the molecule structure",
    tools=["structure_analyzer"],
    scoring_fn=score_analysis,
    submission_format={"analysis": "dict"},
    input_map={"structure": InputRef("retrieve", key="answer")},
    prompt_fn=analyze_prompt,  # custom prompt
    # setup_fn=...,               # per-trial setup / hidden tool args
    resolve_answer=False,  # don't path-resolve a non-file answer
)
```

For genuinely stateful integrations (hardware, pickled inventories) you can
still subclass `Environment` and pass the subclass to `build_environments`:

```python
class MyEnvironment(Environment):
    def configure_additional_apps(self):
        ...  # set up an instrument before each trial
        return "configured"


environments = build_environments(
    tasks,
    base_work_dir="workdir",
    env_cls=MyEnvironment,
)
```

## 4. Run the chained workflow

```python
# The runner executes in dependency order; grouping is automatic.
runner.bench(task_ids=list(environments.keys()), trials_per_task=1)
```

**Done**: Tasks now execute sequentially with data passing between them.
