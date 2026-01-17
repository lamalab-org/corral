# How to Implement Task Chaining

**Goal**: Create a workflow where one task's output feeds into the next task.

**When to use this**: Your problem requires multiple steps where each builds on previous results.

## 1. Define dependent tasks

```python
from corral.backend.task import TaskDefinition, TaskGroup

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
    input_from_tasks=["retrieve"],  # Dependency
)

task_group = TaskGroup(
    group_id="workflow", tasks={"retrieve": retrieve_task, "analyze": analyze_task}
)
```

## 2.  Create environments that use task results

```python
class ChainedEnvironment(Environment):
    def __init__(self, task_id: str, task_group: TaskGroup):
        super().__init__(task_id)
        self.task_group = task_group
        self.current_task = task_group.tasks[task_id]

        # Add tools
        for tool_name in self.current_task.tools:
            self.add_tool(get_tool(tool_name))

    def get_task_prompt(self) -> str:
        # Get input from previous tasks
        task_input = self.task_group.get_task_input(self.task_id)

        prompt = self.current_task.description
        if task_input:
            prompt += f"\n\nPrevious results:\n{task_input}"

        return prompt

    def score(self) -> float:
        score = self.current_task.scoring_fn(self.state.submitted_answer)

        # Store for next task
        self.task_group.store_result(
            self.task_id, {"answer": self.state.submitted_answer}, score
        )

        return score
```

## 3. Run the chained workflow

```python
# Create environments for each task
environments = {
    task_id: ChainedEnvironment(task_id, task_group)
    for task_id in task_group.tasks.keys()
}

# The runner will execute in dependency order
runner.bench(task_ids=list(environments.keys()), trials_per_task=1)
```

**Done**: Tasks now execute sequentially with data passing between them.
