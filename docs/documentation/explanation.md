
These explanations help you understand the "why" behind `Corral`'s design.

## Why Microservice Architecture?

`Corral` separates the environment (server) and agent (runner) into distinct services. This design choice stems from several important considerations.

### Isolation and Safety

By running agents and environments in separate processes, we achieve:

**Resource Isolation**: If an agent consumes excessive memory or CPU, it doesn't affect the environment or other agents. Each component can be monitored and controlled independently.

**Security Boundaries**: Environments may interact with external systems, databases, or instruments. Separating them from agent code creates a security boundary - agents cannot directly access environment internals.

**Failure Independence**: If an agent crashes or hangs, the environment remains available. The runner can restart the agent without losing environment state.

### Scalability

The microservice design enables:

**Distributed Execution**: Agents can run on different machines from environments. This is crucial when environments require specialized hardware (GPUs, lab instruments) while agents need only compute.

**Parallel Benchmarking**: Multiple agents can evaluate against the same environment server simultaneously. This accelerates research comparing different agent architectures.

**Language Flexibility**: While `Corral` is Python-based, the REST API allows implementing agents in any language. Researchers can leverage the ecosystem best suited for their agent design.

### Reproducibility

REST API communication provides:

**Observable Interactions**: Every agent-environment interaction is an HTTP request/response, which can be logged, replayed, and audited. This creates a complete record of agent behavior.

**Versioned Protocols**: The API can be versioned, allowing environments and agents to evolve independently while maintaining compatibility.

**Deterministic Replay**: By recording API interactions, we can replay agent executions exactly, crucial for debugging and scientific reproducibility.

### Development Workflow

Separation enables:

**Independent Development**: Environment creators and agent researchers work independently. Changes to agent logic don't require environment modifications and vice versa.

**Easier Testing**: Environments can be tested with simple HTTP clients before agents exist. Agents can be developed against mock environments.

**Clear Contracts**: The API defines exactly what environments must provide and what agents can expect, reducing ambiguity.

### The Trade-off

The microservice approach adds complexity - you must run two processes, manage network communication, and handle distributed failure modes. `Corral` accepts this trade-off because the benefits (isolation, scalability, reproducibility) are fundamental to reliable agent research.

For simple cases, you might prefer a monolithic design where agent and environment run in the same process. `Corral` prioritizes the needs of serious research over convenience for simple cases.

---

## Understanding the Verbosity System

The verbosity system in `Corral` allows controlling how much detail tool descriptions provide. This addresses a fundamental question in agent research: *How much does documentation context affect agent performance?*

### The Research Question

When we give LLM agents access to tools, we provide descriptions of what each tool does. But how should these descriptions be written?

- Should we provide just a brief summary?
- Should we include detailed usage instructions?
- Do examples help or add noise?
- Does explaining when to use a tool improve decision-making?

These questions matter because token limits and context management are real constraints. If minimal documentation works just as well as comprehensive documentation, we should use minimal documentation to save tokens and reduce latency.

### The Solution: Structured Verbosity

Rather than forcing one documentation style, `Corral` lets you tag different *types* of information in tool docstrings:

```
[BRIEF] - What the tool does (1-2 sentences)
[DETAILED] - How the tool works (technical details)
[PROCEDURAL] - When to use the tool (decision guidance)
[WORKFLOW_INTEGRATION] - How it fits with other tools
[SYNTACTICAL] - Syntax details and formats
[EXAMPLES] - Usage examples
```

Then, at benchmark time, you select a verbosity level that determines which tags are included.

### Why This Matters

**Ablation Studies**: You can run the same agent on the same tasks with different verbosity levels and measure the impact. This reveals whether your agent benefits from detailed documentation or not.

**Model Comparison**: Different models may respond differently to documentation. A highly capable model might infer usage from brief descriptions, while a smaller model might need explicit guidance.

**Token Optimization**: Once you know the minimal effective verbosity for your use case, you can reduce token usage without sacrificing performance.

**Real-World Relevance**: In production systems, documentation quality varies. Testing agents across verbosity levels reveals robustness to documentation variation.

### Design Principles

The system follows several principles:

**Composable**: Higher verbosity levels include everything from lower levels plus additional tags. This ensures fair comparison - you're always adding information, not changing it.

**Semantic**: Each tag type represents a different *kind* of information (what vs. when vs. how), not just "more words."

**Opt-in**: Tools work fine without any verbosity tags. The `[BRIEF]` tag is optional - if absent, the regular docstring serves as brief documentation.


---

## The Philosophy Behind Hooks

Hooks in `Corral` allow injecting custom code at specific points in agent execution.

### The Core Problem

When building agents, you often need to:

- Log what's happening for debugging
- Modify behavior in specific situations
- Inject test conditions or failures
- Track custom metrics
- Implement intervention studies

You could hard-code these features into each agent, but this creates several problems:

**Tight Coupling**: Agent logic becomes mixed with logging, metrics, and special-case handling.

**Limited Reusability**: You can't easily apply the same logging to different agents.

**Fork Proliferation**: Every research variation requires copying and modifying agent code.

### The Hook Solution

Hooks provide *extension points* where you can inject behavior without modifying agent code:

```python
def my_hook(context: HookContext) -> None:
    # Your custom logic
    print(f"Iteration {context.iteration}")


hooks = AgentHooks()
hooks.register(HookPoint.BEFORE_ITERATION, my_hook)

agent = ReActAgent(model="gpt-4", hooks=hooks)
```

The agent doesn't know about your hook. It just exposes hook points and executes registered callbacks.

### Why This Design

**Separation of Concerns**: Agent logic stays focused on reasoning. Logging, metrics, and interventions live in hooks.

**Composability**: Multiple hooks can be registered at the same point. You can combine logging hooks, metric hooks, and intervention hooks.

**Reusability**: Write a hook once, use it with any agent that supports the hook point.

**Research Flexibility**: Run the same agent with different hooks to test interventions without modifying agent code.

### Hook Point Selection

`Corral` provides four hook points, chosen to cover common needs without overwhelming users:

**BEFORE_TASK**: For setup, intervention injection, initialization
**AFTER_TASK**: For cleanup, final metrics, result processing
**BEFORE_ITERATION**: For logging, iteration-specific setup
**AFTER_ITERATION**: For analyzing tool calls, checking progress

This is deliberately minimal. More hook points would provide more flexibility but increase complexity. These four cover most research needs while keeping the system understandable.

### The HookContext Design

Hooks receive a `HookContext` object that's *mutable*. This is important:

```python
def early_stopping_hook(context: HookContext) -> None:
    if too_many_errors(context):
        context.should_continue = False  # Stops agent
```

The hook can *modify* the context, affecting agent behavior. This is more powerful than read-only observation.

The context includes:
- State (messages, iteration number, task_id)
- Control flags (should_continue, skip_current_step)
- Storage (metadata dict for custom data)

This gives hooks both observability and control.

### Interventions as Hooks

Intervention studies (where you inject thoughts or actions into agents) are implemented as hooks:

```python
intervention_hook = create_intervention_hook(
    intervention_map={"task_1": "helpful hint"}, execute_tools=True
)
hooks.register(HookPoint.BEFORE_TASK, intervention_hook)
```

This demonstrates hook composability - interventions are just another type of hook, compatible with logging hooks and metric hooks.


### Design Trade-offs

Hooks add complexity - there's more API surface, more concepts to learn. For simple use cases, they're overkill.

`Corral` accepts this because the target use case is *research*, where you'll run many variations and need flexibility. The hook system pays for itself when you need to:

- Run the same agent with and without interventions
- Compare different logging strategies
- Test custom stopping conditions
- Inject test failures

If you're just running one agent on one task once, hooks are unnecessary. But research involves many variations, and hooks enable that efficiently.

---

## Task Chaining vs. Independent Tasks

`Corral` supports two task execution modes: independent tasks and chained tasks. Understanding when to use each requires understanding what they represent.

### Independent Tasks

In independent mode, each task is self-contained. An agent's solution to task_1 doesn't affect task_2.

**When This Makes Sense**:

- Tasks are different instances of the same problem type
- You want to measure success rate across a task suite
- Tasks share no state or context

**Example**: A benchmark with tasks like:
- "Calculate 25 * 4"
- "Write a function to reverse a string"
- "Explain why the sky is blue"

These are unrelated. Success on one doesn't help with others.

**Execution Model**: The runner completes all trials of task_1, then all trials of task_2. Tasks can even run in parallel (though `Corral` currently runs sequentially).

### Chained Tasks

In chained mode, tasks form a workflow where later tasks depend on earlier task outputs.

**When This Makes Sense**:

- Tasks represent steps in a multi-stage problem
- Later tasks need data produced by earlier tasks
- You're modeling real-world workflows that have stages
- You want to measure end-to-end workflow completion

**Example**: A molecular design workflow:
- Task 1: "Retrieve structure for material X"
- Task 2: "Calculate band gap using the structure from task 1"
- Task 3: "Suggest modifications based on band gap from task 2"

Each task needs the previous task's output.

**Execution Model**: For each trial, the runner completes task_1, then task_2 (using task_1's output), then task_3 (using task_2's output). Trials are atomic - if task_2 fails, you still run a new trial from task_1.

### Why Both Modes?

Different research questions need different models:

**Independent Tasks** answer: "What fraction of diverse problems can the agent solve?"

**Chained Tasks** answer: "Can the agent complete complex multi-step workflows?"

These are fundamentally different questions.

### Implementation Differences

**Independent Tasks**:
```python
environments = {
    "task_1": Environment1(...),
    "task_2": Environment2(...),
}
```

Each environment is independent. No shared state.

**Chained Tasks**:
```python
task_group = TaskGroup(
    tasks={
        "task_1": TaskDefinition(...),
        "task_2": TaskDefinition(..., input_from_tasks=["task_1"]),
    }
)

environments = {
    task_id: ChainedEnvironment(task_id, task_group) for task_id in task_group.tasks
}
```

Environments share a `TaskGroup` that coordinates state passing.

### Scoring Implications

**Independent**: Each task's score is independent. Average score = mean across all task trials.

**Chained**: A workflow succeeds only if all tasks succeed. You might score:
- Individual task success
- Partial workflow completion (got through 2 of 3 tasks)
- End-to-end success (all tasks solved)

`Corral` tracks both individual task scores and full workflow completion.

### Design Philosophy

`Corral` doesn't force one model. It provides primitives for both because they serve different research needs.

If you're evaluating general agent capabilities across diverse tasks, use independent tasks.

If you're evaluating agent performance on complex, structured problems requiring multiple steps, use chained tasks.

You can even mix them - some tasks independent, some chained - by creating separate `TaskGroup` instances.

The key insight: *task independence vs. chaining is a property of the research question, not the implementation*. `Corral`'s architecture lets you model both.

---

## Scoring Function Design Principles

Scoring functions in `Corral` evaluate agent solutions, typically returning values between 0.0 (failure) and 1.0 (success). Designing good scoring functions requires careful thought.

### Binary vs. Continuous Scores

**Binary Scoring** (0.0 or 1.0):
```python
def score(self):
    return 1.0 if self.state.submitted_answer == self.correct_answer else 0.0
```

**When to use**:
- Task has a clear right/wrong answer
- Partial credit doesn't make sense
- You want simple pass/fail metrics

**Continuous Scoring** (range 0.0 to 1.0):
```python
def score(self):
    error = abs(float(self.state.submitted_answer) - self.true_value)
    if error < 0.1:
        return 1.0
    elif error < 0.5:
        return 0.8
    elif error < 1.0:
        return 0.5
    return 0.0
```

**When to use**:
- Answers have varying quality
- Near-misses should get partial credit
- You want fine-grained performance measurement

### Determinism

Scoring functions should be deterministic - same input always produces same score.

**Bad**:
```python
def score(self):
    # Uses current time - not deterministic!
    if datetime.now().hour < 12:
        return 1.0
    return 0.0
```

**Good**:
```python
def score(self):
    # Uses only submitted answer and fixed ground truth
    return compare(self.state.submitted_answer, self.ground_truth)
```

**Why**: Research requires reproducibility. If scores change between runs, you can't compare results or debug issues.

### Robustness to Format Variation

Agents might format answers differently:

- "42" vs. "42.0" vs. "The answer is 42"
- "Yes" vs. "yes" vs. "YES" vs. "True"

Robust scoring handles variation:

```python
def score(self):
    # Extract number from answer text
    import re

    match = re.search(r"\d+\.?\d*", self.state.submitted_answer)
    if match:
        answer = float(match.group())
        return 1.0 if abs(answer - self.correct_answer) < 0.001 else 0.0
    return 0.0
```

**Trade-off**: More lenient parsing means agents get credit despite poor formatting. Stricter parsing encourages agents to format correctly but may penalize correct answers in wrong format.

### Using External Ground Truth

Sometimes you need external systems for scoring:

```python
def score(self):
    # Use external library to validate structure
    from pymatgen import Structure

    try:
        structure = Structure.from_str(self.state.submitted_answer, fmt="cif")
        return structure_quality_score(structure, self.reference)
    except:
        return 0.0
```

**Considerations**:
- External dependencies must be reproducible (fixed versions)
- External calls should be fast (scoring happens per trial)
- Failures should be handled gracefully (bad format -> 0.0, not crash)

### Partial Credit for Process

Should agents get credit for using the right approach even if the final answer is wrong?

**Process-based scoring**:
```python
def score(self):
    score = 0.0

    # Check if correct tools were used
    tools_used = set(call.tool_name for call in self.state.tool_calls)
    if "database_query" in tools_used:
        score += 0.3  # Credit for retrieving data

    if "structure_analyzer" in tools_used:
        score += 0.3  # Credit for analysis

    # Check final answer
    if self.state.submitted_answer == self.correct_answer:
        score += 0.4  # Credit for correct answer

    return score
```

**Trade-offs**:
- Encourages agents to follow intended workflow
- Allows measuring "partial completion"
- But may reward inefficient approaches
- Requires defining what counts as "correct process"

`Corral` doesn't prescribe a philosophy - you design scoring to match your research question.

### Scoring in Chained Tasks

In task chains, you have two scoring opportunities:

**Individual Task Scores**:
```python
def score(self):
    # Did this task succeed?
    return 1.0 if valid(self.state.submitted_answer) else 0.0
```

**Workflow-Level Scores**:
```python
# After all tasks complete
workflow_score = task1_score * 0.3 + task2_score * 0.3 + task3_score * 0.4
```

You can weight tasks by importance or require all tasks to succeed.

### The Meta-Principle

Scoring functions encode what you value. Ask:

- Do I care about exact answers or approximate answers?
- Should process matter or only outcomes?
- Are some errors worse than others?
- Should agents get credit for trying?

Your scoring function answers these questions. Design it to match your research goals, not generic notions of "correctness."

---

## When to Use Surrender

The surrender mechanism lets agents quit tasks they cannot solve. This might seem like giving up, but in research and production, knowing when to quit is valuable.

### The Problem

Without surrender, agents may:

- Waste time and tokens on impossible tasks
- Hit iteration limits without useful output
- Get stuck in loops trying approaches that cannot work
- Consume resources that could go to solvable tasks

Consider an agent given a task: "Find the molecular structure of unobtanium." If the database doesn't contain unobtanium, no amount of searching will succeed. An agent that recognizes this and surrenders is more efficient than one that tries all 10 iterations.

### When Surrender Makes Sense

**Impossible Tasks**: If the task cannot be solved with available tools, surrender is correct.

**Missing Prerequisites**: If data or tools the task requires are unavailable, surrender is appropriate.

**Resource Constraints**: If solving the task would exceed reasonable resource bounds, surrender saves resources.

**Ambiguous Tasks**: If the task description is unclear or contradictory, surrender with explanation is better than guessing.


### Design Philosophy

`Corral` makes surrender explicit and trackable:

- Agents must deliberately return "GIVE UP"
- Results mark `surrendered=True`
- Metrics distinguish surrenders from failures

This lets you analyze:
- Surrender rates per task
- Whether surrenders are justified
- Resource savings from surrender
- Agent calibration (surrenders on hard tasks, persists on easy ones)

Surrender is a signal worth studying, not a failure to hide.

---
