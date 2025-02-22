# Material Agent Benchmark

The system consists of three main components:

- Environment Service (`corral`)- Hosts tasks and tools
- Benchmark Interface (`MatAgentBenchmark`)- Communicates with the `corral` service, runs evaluations
- `Agent` - Solves tasks using available tools

## 1. Create environment and add tools example:

example of creating a tool and running the environment server

```bash
cd tasks/samplemath/samplemath
python -m env # start corral service
```

Creating a simple environment with a math task and a calculator tool

- Create a new environment class that inherits from `Environment` (this comes with tool calling and benchmark, tool instructions)
- Implement the required methods
- Add tools to the environment
- Add scoring logic

```python
class MathEnvironment(Environment):
    def __init__(self, task_id: str, question: str, answer: float):
        self.question = question
        self.correct_answer = answer
        super().__init__(task_id)

        # Add multiple tools
        self.add_tool(calculator)
        self.add_tool(UnitConverterTool())

    def get_task_prompt(self) -> str:
        return f"Solve this math problem: {self.question}"

    def score(self) -> float:
        """Score based on submitted answer"""
        if self.state.submitted_answer is None:
            return 0.0
        try:
            submitted_result = float(self.state.submitted_answer)
            return 1.0 if abs(submitted_result - self.correct_answer) < 0.001 else 0.0
        except ValueError:
            return 0.0


environments = {
    "math_1": MathEnvironment("math_1", "What is 23 + 45?", 68),
    "math_2": MathEnvironment("math_2", "What is 12 * 8?", 96),
    "math_3": MathEnvironment("math_3", "What is 99 * 63 * 999 * 111?", 691614693),
}
```

### Create tools with `tool` decorator

```python
from corral.utils import tool

@tool
def percentage_calculator(value: float, percentage: float = 100.0) -> float:
    """Calculate percentage of a value.

    Args:
        value: The base value
        percentage: The percentage to calculate (defaults to 100.0)

    Returns:
        float: The calculated result
    """
    return (value * percentage) / 100.0

```

### Create tool that would run in [Modal](https://modal.com/) environment

```python
from corral.utils import modal_tool, MODAL_TOOL_REGISTRY
@modal_tool(app=app, image=Image.debian_slim().pip_install("numerizer"), memory=512)
def number_convert(text: str, return_float: bool = False) -> str:
    """
    Convert number words to numeric representation.

    Args:
        text: Text containing number words (e.g. 'forty two', 'one million')
        return_float: Whether to return float for decimal values (choices: [True, False])

    Returns:
        String containing the numeric representation
    """
    from numerizer import numerize

    result = numerize(text)
    return str(float(result)) if return_float and "." in result else result


# TODO: decorator could not return the tool instance
number_converter = MODAL_TOOL_REGISTRY["number_convert"]
```python

## 2. Create agent example:

see an example of a simple agent.
```python
```bash
cd agents/baseline
```

Creating an agent
- Create a new agent class that implements the `Agent` protocol
- TODO: Toolcalling parsing final answer submission parsing from message to the server
(see the example on how this is being done now.)

```python
class Agent(Protocol):
    """Protocol defining what an agent must implement"""

    def solve_task(self, interface: BenchmarkInterface, task_id: str) -> str:
        """Solve a task and return the answer"""
        ...
        guide = interface.get_task_guide(task_id) # get instruction on task, tools available and their respective tool calling syntax
        system_prompt = BASELINESYSTEMPROMPT.format(guide=guide) # set these instructions to the system prompt
        interface.execute_tool(task_id, tool_request["tool_name"], tool_request["arguments"]) # how to communicate tool calls


```

## 3. Benchmark Interface

Setup benchmark interface
- Start `corral` service and get the base url. See step 1. (This would `start` corral server, with 3 tasks and 2 tools)

```python
from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

interface = BenchmarkInterface(base_url) # defaults to "http://localhost:8000"
agent = ClaudeAgent(api_key=os.getenv("ANTHROPIC_API_KEY")) # or any other agent
runner = MatAgentBenchmark(interface, agent)

result = runner.bench()
```

## Development 

This project uses pre-commit hooks to maintain code quality. The hooks run automatically on each commit to ensure consistent code formatting and catch common issues early.

### Setup

1. Install Python dependencies: `pip install pre-commit`
2. Install the pre-commit hooks: `pre-commit install`

The checks will run automatically when you commit changes. However, you can also run them manually:

- Run on all files: `pre-commit run --all-files`
- Run on specific files: `pre-commit run --files path/to/file1.py path/to/file2.py`

For best experience, install `ruff` in your editor to format code on save and to show linting errors.