# Tutorial 2: Building a Multi-Tool Environment

**What you'll learn**: Create an environment with multiple tools and see how agents use them together.

**Prerequisites**: Complete Tutorial 1

## Step 1: Create tools for a calculator

Create `calculator_env.py`:

```python
from corral.backend.env import Environment
from corral.backend.tool import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b
```

Notice we created three separate tools. Each tool does one thing clearly.

## Step 2: Create the environment

Add to `calculator_env.py`:

```python
class CalculatorEnvironment(Environment):
    def __init__(self, task_id: str, problem: str, answer: float):
        super().__init__(task_id)
        self.problem = problem
        self.answer = answer

        # Add all three tools
        self.add_tool(add)
        self.add_tool(multiply)
        self.add_tool(subtract)

    def get_task_prompt(self) -> str:
        return f"Calculate: {self.problem}\nSubmit your answer as a number."

    def score(self) -> float:
        if self.state.submitted_answer is None:
            return 0.0
        try:
            result = float(self.state.submitted_answer)
            return 1.0 if abs(result - self.answer) < 0.001 else 0.0
        except (ValueError, TypeError):
            return 0.0
```

The environment now has three tools available. The agent will need to choose which ones to use.

## Step 3: Create complex tasks

Add to `calculator_env.py`:

```python
from corral.backend.server import create_benchmark_server
import uvicorn

environments = {
    "simple_add": CalculatorEnvironment("simple_add", "10 + 5", 15),
    "two_step": CalculatorEnvironment("two_step", "(10 + 5) * 2", 30),
    "three_step": CalculatorEnvironment("three_step", "10 * 3 - 5", 25),
}

if __name__ == "__main__":
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Notice the tasks increase in complexity. The three-step task requires the agent to use multiple tools.

## Step 4: Run with verbose output

Start the server:

```bash
python calculator_env.py
```

In another terminal, create `run_verbose.py`:

```python
from corral.run import CorralRunner
from corral.router import CorralRouter
from corral.agents import ReActAgent

interface = CorralRouter(base_url="http://localhost:8000")
agent = ReActAgent(model="openai/gpt-4o", max_iterations=10)
runner = CorralRunner(interface, agent)

result = runner.bench(
    task_ids=["three_step"], trials_per_task=1, verbose=True  # Enable verbose output
)

print(f"\nScore: {result.average_score():.2f}")

# Check tool usage
for trial in result.all_results:
    stats = trial.tool_statistics
    print(f"Tools used: {stats['tools_used']}")
    print(f"Total tool calls: {stats['total_calls']}")
```

Run it:

```bash
python run_verbose.py
```

Watch the agent's reasoning process. You'll see it call `multiply` first, then `subtract`. The agent conversation is saved in JSON files in your working directory.

## What you accomplished

- ✅ Created multiple related tools
- ✅ Built tasks requiring multi-step reasoning
- ✅ Enabled verbose output to see agent thinking
- ✅ Examined tool usage statistics

---
