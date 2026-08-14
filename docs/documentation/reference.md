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

Defines a task (immutable). A task that depends on other tasks is a normal task whose `input_map` references other tasks' outputs.

**Fields**:
- `name` (str): Task name

- `description` (str): Task description

- `tools` (list[str]): Required tool names

- `scoring_fn` (Callable): Scoring function

- `submission_format` (dict[str, str]): Expected answer format

- `scoring_inputs` (dict): Additional scoring parameters

- `initial_input` (dict): Static input values available to the task

- `input_map` (dict[str, InputRef]): Named inputs taken from other tasks' outputs

**Methods**:
- `dependencies() -> set[str]`: Ids of the tasks referenced by `input_map`

---

## InputRef Dataclass

Frozen reference to the output of another task.

**Fields**:
- `task_id` (str): The upstream task

- `key` (str): Output key to read, default `"answer"`; supports dotted paths into structured outputs (e.g. `"answer.smiles"`)

---

## Task helper functions

- `build_dependency_graph(tasks: Mapping[str, TaskDefinition]) -> dict[str, list[str]]`: Derive the dependency graph from task definitions

- `topological_order(tasks: Mapping[str, TaskDefinition]) -> list[str]`: Return task ids in dependency order; raises `ValueError` on cycles

- `validate_task_graph(tasks: Mapping[str, TaskDefinition]) -> None`: Raise `ValueError` on unknown dependencies or cycles

---

## CorralState Dataclass

Single source of truth for one environment's runtime state (`corral.backend.state`). Every `Environment` owns one instance as `env.state`; environments, tools, scoring, and the server read/write through it.

**Fields** (selection):
- `task_id`, `task_prompt`, `trial_id`, `trial_counter`: Identity of the current trial

- `task_group_id`: Linked-task identity (optional)

- `status`, `started_at`, `ended_at`: Trial lifecycle

- `messages` (list[dict]), `tool_calls` (list[ToolCall]): Interaction trace

- `submitted_answer`, `score`, `feedback`, `surrendered`, `is_attempted`: Submission/scoring

- `task_runs` (dict[str, TaskRunState]): Per-task runtime outcomes (output, score, feedback); shared (same dict object) between linked environments

- `workspace`, `artifacts`, `hidden_args`: Runtime resources

- `trials` (dict[str, dict]): Archived snapshots of completed trials

**Methods**:
- `start_new_trial(task_prompt, workspace=None) -> str`: Archive the finished trial and start a clean one

- `submit(answer)` / `surrender()` / `set_score(score, feedback=None)`: Submission lifecycle

- `record_message(role, content, **metadata)` / `record_tool_call(call)`: Trace recording

- `store_task_output(task_id, answer, score=None, feedback=None) -> None`: Record a task's output for dependent tasks

- `get_output(task_id, key="answer") -> Any`: Read one field of a task's output; `key` may be a dotted path

- `is_completed(task_id: str) -> bool`: Whether a task has stored output

- `dependencies_satisfied(task: TaskDefinition) -> bool`: Check if all of a task's dependencies are completed

- `resolve_inputs(task: TaskDefinition) -> dict`: Combine `initial_input` with resolved `input_map` values; raises if a dependency has not completed

- `tool_statistics() -> dict`: Aggregated tool-call statistics

- `snapshot(include_trials=True) -> dict`: Serializable snapshot (excludes `hidden_args`)

---

## CorralRunner Class

Executes agents against an environment and aggregates results.

**Constructor**:

```python
runner = CorralRunner(
    interface,  # CorralRouter or AsyncCorralRouter
    agent=None,  # a single shared BaseAgent (serial runs)
    agent_factory=None,  # (TrialContext) -> BaseAgent (required for concurrency)
    concurrency=None,  # default ConcurrencyConfig for this runner
    checkpoint_dir="./benchmark_checkpoints",
    enable_surrender=False,
    metrics=None,
)
```

Exactly one agent source is required. A shared `agent` accumulates per-run state, so it is fine for serial `bench()` but **cannot** back concurrent trials; pass an `agent_factory` for those. Both may be supplied — the factory then mints each trial's agent and the shared agent is used only for run metadata / report labeling.

### `bench(...) -> BenchmarkResult`

Run the benchmark synchronously. Key parameters:

- `task_ids` (list[str] | None): Tasks to run; `None` runs every available task.
- `trials_per_task` (int): Trials per task (for `pass@k`).
- `k_values` (int | list[int] | None): k values for pass@k metrics.
- `run_name` (str | None): Name used in the report filename.
- `max_concurrency` (int, default `1`): Max trials in flight across the whole run. `1` is the byte-for-byte serial path; `> 1` runs concurrently via `abench` and **requires** an `agent_factory`.
- `max_concurrency_per_task` (int, default `1`): Max simultaneous trials of the *same* task. Above `1` requires the server to support trial runtimes.
- `agent_factory` (AgentFactory | None): Per-call override of the runner's factory.
- `concurrency` (ConcurrencyConfig | None): Full config; when given it supersedes the two scalars above.

When the effective concurrency exceeds `1`, `bench` drives an event loop via `anyio.run`, so it must **not** be called from inside a running event loop.

### `abench(...) -> BenchmarkResult`

Async counterpart of `bench` with the same arguments. Use it directly when you are already inside an event loop (pair it with an `AsyncCorralRouter`).

---

## ConcurrencyConfig Dataclass

Frozen value object bounding how many trials — and how much per-trial work — run at once. Pass it to `CorralRunner(concurrency=...)` or to `bench`/`abench(concurrency=...)`.

**Fields**:

- `global_trials` (int, default `1`): Max trials executing simultaneously across the whole benchmark (the `max_concurrency` scalar).

- `per_task` (int, default `1`): Max simultaneous trials of the same task (the `max_concurrency_per_task` scalar). Values above `1` require server-side trial-runtime support.

- `per_model` (Mapping[str, int] | None): Per-model cap on simultaneous trials, keyed by agent model name (e.g. `{"claude-opus": 8}`). A model absent from the map is bounded only by `global_trials`. Each cap must be `>= 1`.

- `tool_jobs_per_trial` (int, default `4`): Max background jobs running at once within a single trial runtime; forwarded to each runtime's `JobManager`.

- `configure_apps` (int | None): Max trials configuring their additional apps/services simultaneously. Must be `>= 1` when set.

`per_model` and `configure_apps` are enforced only on the concurrent path (`abench`, and `bench` when effective concurrency exceeds `1`). All numeric fields are validated (`>= 1`) at construction.

---

## TrialContext Dataclass

Immutable description of the trial an `AgentFactory` is building for. Passed to the factory so it can bind an agent to one specific trial.

**Fields**:

- `task_id` (str): Task this trial belongs to.

- `trial_index` (int): Zero-based index of this trial within the task.

- `session_id` (str): Benchmark session id (checkpoint namespace).

- `benchmark_run_id` (str): Identifier for the whole benchmark invocation.

`AgentFactory` is the callable protocol `(TrialContext) -> BaseAgent`.

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
