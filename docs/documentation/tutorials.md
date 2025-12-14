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
