# Technical descriptions of Corral's components and APIs.

## REST API Endpoints

### Task Management

**GET `/tasks`**
Returns: `list[str]` - Available task IDs

**GET `/tasks/{task_id}/prompt`**
Returns: `{"prompt": str}` - Task prompt without tools

**GET `/tasks/{task_id}/guide?verbosity={level}`**
Parameters: `verbosity` (str, optional)
Returns: `{"prompt": str}` - Complete task guide

**GET `/tasks/{task_id}/tools?verbosity={level}`**
Parameters: `verbosity` (str, optional)
Returns: `{"tools": list[dict]}` - Structured tool definitions

**GET `/tasks/{task_id}/tools/guide?verbosity={level}`**
Parameters: `verbosity` (str, optional)
Returns: `{"prompt": str}` - Tools guide only

### Task Execution

**POST `/tasks/{task_id}/tools/execute`**
Body: `{"tool_name": str, "arguments": dict}`
Returns: `{"result": ToolCall}` - Tool execution result

**POST `/tasks/{task_id}/submit`**
Body: `{"answer": str}`
Returns: `TrialCompletionResponse` - Score and trial data

**POST `/tasks/{task_id}/surrender`**
Returns: `TrialCompletionResponse` - Surrender confirmation

**POST `/tasks/{task_id}/configure`**
Returns: `{"status": str}` - Configuration status

### State Queries

**GET `/tasks/{task_id}/status`**
Returns: Current task state dictionary

**GET `/tasks/{task_id}/trials/{trial_id}`**
Returns: `{"trial_state": dict}` - Specific trial state

**GET `/tasks/{task_id}/last_score`**
Returns: `{"score": float, "trial_id": str}` - Most recent score

**GET `/dependency_chain`**
Returns: `{"dependency_chain": bool}` - Chain support flag

---


### Class: `CorralRouter`

Interface for agent-environment communication.

**Constructor**:

```python
interface = CorralRouter(
    base_url="http://localhost:8000",
    default_verbosity="brief",
)
```

**Parameters**:

- `base_url` (str): URL of the Corral server

- `default_verbosity` (str | None): Default tool verbosity level

**Methods**:

#### `get_available_tasks() -> list[str]`
Returns list of available task IDs.

**Example**:
```python
interface = CorralRouter(base_url="http://localhost:8000")
tasks = interface.get_available_tasks()
# Returns: ["task_1", "task_2", "task_3"]
```

#### `get_task_prompt(task_id: str) -> str | list[dict]`
Returns task prompt without tool descriptions.

**Parameters**:
- `task_id` (str): Task identifier

**Returns**: Task description string or list of message dicts

#### `get_task_guide(task_id: str, verbosity: str | None = None) -> str`
Returns complete task guide including tools.

**Parameters**:
- `task_id` (str): Task identifier

- `verbosity` (str | None): Tool description verbosity level

**Returns**: Complete task guide with tool descriptions

#### `get_tools_guide(task_id: str, verbosity: str | None = None) -> str`
Returns tools guide only.

**Parameters**:
- `task_id` (str): Task identifier

- `verbosity` (str | None): Tool description verbosity level

**Returns**: Tool descriptions

#### `get_available_tools_for_task(task_id: str, verbosity: str | None = None) -> dict`
Returns structured tool information.

**Parameters**:
- `task_id` (str): Task identifier

- `verbosity` (str | None): Tool description verbosity level

**Returns**: Dictionary with tool definitions

#### `execute_tool(task_id: str, tool_name: str, arguments: dict) -> ToolResponse`
Executes a tool in the environment.

**Parameters**:
- `task_id` (str): Task identifier

- `tool_name` (str): Name of tool to execute

- `arguments` (dict): Tool arguments

**Returns**: `ToolResponse` object with success, result, and error fields

**Example**:
```python
response = interface.execute_tool(
    "task_1", "calculator", {"operation": "+", "a": 10, "b": 5}
)
print(response.success)  # True
print(response.result)  # 15
```

#### `submit_answer(task_id: str, answer: str) -> TaskTrialResult`
Submits final answer for scoring.

**Parameters**:
- `task_id` (str): Task identifier

- `answer` (str): Final answer

**Returns**: `TaskTrialResult` with score and state

#### `surrender_task(task_id: str) -> TaskTrialResult`
Surrenders the current task without submitting answer.

**Parameters**:
- `task_id` (str): Task identifier

**Returns**: `TaskTrialResult` with surrendered=True and score=0.0

#### `get_task_status(task_id: str) -> dict`
Returns current task state.

**Parameters**:
- `task_id` (str): Task identifier

**Returns**: Dictionary with current state including score, trial_id, etc.

#### `supports_dependency_chain() -> bool`
Checks if environment supports task chaining.

**Returns**: Boolean indicating chain support

---

## Environment Class

### Abstract Class: `Environment`

Base class for all Corral environments.

**Constructor**:
```python
Environment(
    task_id,
    base_work_dir="tmp",
    fs_manager=None,
)
```

**Parameters**:
- `task_id` (str): Unique task identifier

- `base_work_dir` (str): Base directory for trial workspaces

- `fs_manager` (FSManager | None): Optional file system manager

**Attributes**:
- `task_id` (str): Task identifier

- `tools` (dict[str, Tool]): Available tools

- `state` (TaskState): Current trial state

- `trial_counter` (int): Trial number counter

- `current_work_dir` (str): Current trial workspace

**Abstract Methods** (must implement):

#### `get_task_prompt() -> str | list[dict]`
Returns task description for agent.

#### `score() -> float`
Evaluates agent's solution and returns score (0.0 to 1.0).

**Concrete Methods**:

#### `add_tool(tool: Tool) -> None`
Registers a tool with the environment.

**Parameters**:
- `tool` (Tool): Tool to add

#### `call_tool(tool_name: str, arguments: dict) -> ToolCall`
Executes a tool and records the call.

**Parameters**:
- `tool_name` (str): Tool name

- `arguments` (dict): Tool arguments

**Returns**: `ToolCall` object with result and status

#### `submit_answer(answer: str) -> float`
Submits answer and computes score.

**Parameters**:
- `answer` (str): Final answer

**Returns**: Score value

#### `surrender() -> float`
Marks task as surrendered.

**Returns**: Current score (usually 0.0)

#### `reset_state() -> str`
Resets environment for new trial.

**Returns**: New trial ID

#### `get_current_work_dir() -> str`
Returns current trial workspace path.

#### `configure_additional_apps() -> str`
Optional: Configure external services.

**Returns**: Configuration status message

---

## BaseAgent Class

### Abstract Class: `BaseAgent`

Base class for all Corral agents.

**Constructor**:
```python
BaseAgent(
    model="openai/gpt-4o",
    max_iterations=10,
    api_endpoint=None,
    system_prompt=None,
    user_prompt=None,
    extractor_prompt=None,
    surrender_prompt=None,
    temperature=0.7,
    hooks=None,
    **kwargs,
)
```

**Parameters**:
- `model` (str): LLM model identifier

- `max_iterations` (int): Maximum reasoning iterations

- `api_endpoint` (str | None): Optional API endpoint

- `system_prompt` (str | None): System prompt template

- `user_prompt` (str | None): User prompt template

- `extractor_prompt` (str | None): Answer extraction prompt

- `surrender_prompt` (str | None): Surrender instructions

- `temperature` (float): LLM sampling temperature

- `hooks` (AgentHooks | None): Lifecycle hooks

- `**kwargs`: Additional LLM parameters

**Attributes**:
- `model` (str): Model name

- `messages` (list): Conversation history

- `token_usage` (dict): Token usage statistics

- `hooks` (AgentHooks): Registered hooks

**Abstract Methods**:

#### `run(interface, task_id, history=None, task_prompt=None, examples=None, **kwargs) -> str`
Main agent reasoning loop. Must be implemented by subclasses.

**Parameters**:
- `interface` (CorralRouter): Router interface
- `task_id` (str): Task to solve
- `history` (list | None): Conversation history
- `task_prompt` (str | None): Override task prompt
- `examples` (list | None): Few-shot examples
- `**kwargs`: Additional parameters

**Returns**: Final answer string

**Concrete Methods**:

#### `get_llm_response(tools: list[dict] | None = None) -> Any`
Calls LLM with current messages.

**Parameters**:
- `tools` (list[dict] | None): Optional function definitions

**Returns**: LLM response object

#### `get_total_token_usage() -> dict[str, int]`
Returns cumulative token usage.

**Returns**: Dictionary with prompt_tokens, completion_tokens, total_tokens

#### `reset_token_usage() -> None`
Resets token counters.

---

## Built-in Agent Types

### `ReActAgent`

Implements Reasoning and Acting framework.

**Additional Parameters**:
- All BaseAgent parameters

**Response Format**:
```
<thought>reasoning here</thought>
<action>tool_name</action>
<action_input>{"arg": "value"}</action_input>
```

### `ToolCallingAgent`

Uses native LLM function calling.

**Additional Parameters**:
- All BaseAgent parameters

**Note**: Automatically converts Corral tools to OpenAI function format.

### `LLMPlanner`

Hierarchical planning agent.

**Additional Parameters**:
- `planner_model` (str): Model for high-level planning

- `executor_model` (str): Model for low-level execution

### `ReflectionAgent`

Self-reflective agent.

**Additional Parameters**:
- `max_reflections` (int): Maximum reflection cycles

---

## ToolVerbosity Enum

Tool description verbosity levels.

```python
class ToolVerbosity(Enum):
    MINIMAL = "minimal"  # Name + basic description
    BRIEF = "brief"  # + [BRIEF] sections
    DETAILED = "detailed"  # + [DETAILED] sections
    PROCEDURAL = "procedural"  # + [PROCEDURAL] sections
    CONTEXTUAL = "contextual"  # + [CONTEXTUAL] sections
    WORKFLOW = "workflow"  # + [WORKFLOW_INTEGRATION]
    SYNTACTICAL = "syntactical"  # + [SYNTACTICAL] sections
    COMPREHENSIVE = "comprehensive"  # + [RAISES], [LIMITATIONS], [EXAMPLES]
    FULL = "full"  # Complete docstring
```

---

## HookPoint Enum

Agent lifecycle hook points.

```python
class HookPoint(Enum):
    BEFORE_TASK = "before_task"  # Before task starts
    AFTER_TASK = "after_task"  # After task completes
    BEFORE_ITERATION = "before_iteration"  # Before each LLM call
    AFTER_ITERATION = "after_iteration"  # After tools execute
```

---

## TaskDefinition Dataclass

Defines a task within a TaskGroup.

**Fields**:
- `name` (str): Task name

- `description` (str): Task description

- `tools` (list[str]): Required tool names

- `scoring_fn` (Callable): Scoring function

- `submission_format` (dict[str, str]): Expected answer format

- `scoring_inputs` (dict): Additional scoring parameters

- `input_from_tasks` (list[str]): Task dependencies

- `initial_input` (dict): Initial inputs

**Methods**:
- `has_dependencies() -> bool`: Returns True if task has dependencies

---

## TaskGroup Dataclass

Container for related tasks.

**Fields**:
- `group_id` (str): Group identifier

- `tasks` (dict[str, TaskDefinition]): Task definitions

- `results` (dict[str, Any]): Stored results

- `scores` (dict[str, float]): Stored scores

- `chained_tasks` (bool): Whether tasks are chained

**Methods**:
- `get_task_input(task_id: str) -> dict`: Get inputs for task

- `store_result(task_id: str, result: dict, score: float) -> None`: Store result

- `get_ordered_tasks() -> list[str]`: Get tasks in dependency order

- `check_dependencies_satisfied(task_id: str) -> bool`: Check if dependencies met

---

## BenchmarkResult Dataclass

Results from benchmark execution.

**Fields**:
- `task_results` (dict[str, TaskTrialResults]): Results per task

- `k` (list[int]): k values for pass@k metrics

- `total_duration` (float | None): Total benchmark duration

- `verbosity` (str | None): Tool verbosity used

**Properties**:
- `all_task_ids` (list[str]): All task IDs

- `total_tasks` (int): Number of tasks

- `all_results` (list[TaskTrialResult]): Flat list of all trials

**Methods**:
- `average_score() -> float`: Mean score across all trials

- `task_average_score(task_id: str) -> float`: Mean score for task

- `pass_at_k(k: int) -> float`: Pass@k metric

- `best_pass_k() -> tuple[int, float]`: Best pass@k value

- `overall_average_duration() -> float | None`: Mean trial duration

- `total_token_usage() -> dict[str, int]`: Total tokens used

- `total_tool_execution_duration() -> float`: Total time in tools

---

## TaskTrialResult Dataclass

Result of a single trial.

**Fields**:
- `task_id` (str): Task identifier

- `trial_id` (str): Trial identifier

- `score` (float): Trial score

- `state` (dict): Final state

- `tool_statistics` (dict): Tool usage stats

- `duration` (float | None): Trial duration in seconds

- `token_usage` (dict | None): Token usage

- `error_message` (str | None): Error if failed

- `surrendered` (bool): Whether surrendered

**Properties**:
- `success` (bool): True if score > 0, no error, not surrendered

---
