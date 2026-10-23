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
  <img alt="Corral logo" src="docs/_static/corral_logo_final.png" width='300px'>
</picture>
</p>

A comprehensive benchmarking framework for evaluating AI agents on science tasks. The system provides standardized environments, tools, and evaluation metrics to test agent performance across diverse materials science challenges.

## 🚀 Quick Start: Run One Task from One Environment

The shortest useful Corral run does not need `CorralRunner`, trials, scoring,
or report generation. `corral run` executes one independent task using
the `execute_task` function in `run.py`.

You need Python 3.11 or newer for SampleMath, [`uv`](https://docs.astral.sh/uv/),
and a model API key such as `OPENAI_API_KEY`. Clone the
repository and install SampleMath together with the framework:

```bash
git clone https://github.com/lamalab-org/corral.git
cd corral/tasks/samplemath
uv sync
```

Run one task with one agent and model:

```bash
uv run corral run \
  --agent tool-calling \
  --environment samplemath \
  --task task1 \
  --model openai/gpt-5.6
```

This path skips evaluation: it prints the final status, submitted
answer, and commit hash, and writes the authored commit ledger to
`.corral/run-commits.sqlite3`. It does not create trials, invoke a scorer, or
generate a benchmark report. Because it executes exactly one task, it rejects a
task with upstream dependencies and points to `corral bench` instead.

Re-running with the same `--execution-id` and commit database restores the
latest persisted task state.

The equivalent Python API is below. Save this as `quickstart.py` in the current
`tasks/samplemath` directory:

```python
import asyncio
from uuid import uuid4

from dotenv import load_dotenv

from corral import (
    RuntimeRegistry,
    RunTaskInput,
    execute_task,
    load_environment_group,
)
from corral.agents import ToolCallingAgent
from corral.persistence import SQLiteCommitStore

AGENT_ID = "tool-calling"
TASK_ID = "task1"
MODEL = "openai/gpt-5.6"


async def main():
    load_dotenv()
    environment = load_environment_group("samplemath")[TASK_ID]
    store = SQLiteCommitStore(".corral/quickstart-commits.sqlite3")
    registry = RuntimeRegistry(
        agents={AGENT_ID: ToolCallingAgent(model=MODEL)},
        environments={TASK_ID: environment},
    )

    try:
        state = await execute_task(
            state_store=store,
            registry=registry,
            task=RunTaskInput(
                execution_id=f"samplemath-task1-{uuid4().hex}",
                task_id=TASK_ID,
                environment_id=TASK_ID,
                agent_id=AGENT_ID,
                model=MODEL,
                max_iterations=10,
            ),
        )
    finally:
        registry.close()
        await store.aclose()

    print(f"status: {state.status}")
    print(f"answer: {state.submission}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it with the current SampleMath virtual environment:

```bash
uv run python quickstart.py
```

The result is a `StateRef` pointing to the persisted final state, and the
example prints its status and submitted answer. No scorer,
aggregate metric, or benchmark report runs. `task1` is independent; use
`corral bench` for a task such as `task4` whose inputs come from earlier tasks.

## 📊 Running Benchmarks

### Run any agent with the `corral` CLI

The installed `corral` command can run every public concrete agent against a
registered environment.

Each task environment is a separate Python project because it has its own
dependencies. Change into that project once, then `uv` automatically uses its
environment and no `--project` option is needed:

```bash
cd tasks/samplemath
uv sync
uv run corral bench \
  --agent react \
  --environment samplemath \
  --model openai/gpt-4o \
  --task task4 \
  --trials 3
```

`corral bench` calls the same `CorralRunner` used by Python callers. The runner
schedules trials in the CLI process using `asyncio`.

Benchmarks run one Docker container per `(run, task, trial)` by default. Corral
builds `corral-benchmark:latest` from `docker/benchmark.Dockerfile` when that
image is missing, resolves it to an immutable image ID before scheduling any
trial, and applies these defaults: 2 CPUs, 4 GiB memory, 256 PIDs, a read-only
root filesystem, and Docker's `bridge` network so the container has outbound
network access. Pass `--sandbox-network none` for fully network-isolated trials.
Use `--sandbox local` only for local debugging; `corral run` remains
local and never contacts Docker.

By default, each invocation creates one descriptive, self-contained directory
below `.corral/runs/`. Its name records the UTC start time, agent, model,
environment, requested tasks, k/trial count, and sandbox mode. The exact task
and trial checkpoint layout is:

```text
.corral/runs/<descriptive-run-name>/
├── report.json
├── run-metadata.json
└── task-<task-id>/
    └── k-<1-based-trial>/
        ├── metadata.json
        ├── commits.sqlite3
        ├── artifacts/
        ├── workspace-snapshots/
        │   ├── <revision>.json
        │   └── latest.json
        ├── state-snapshots/
        │   ├── <sequence>-<commit>.json
        │   ├── latest.json
        │   └── final.json
        └── recovery/
```

Workspace snapshot manifests reference the content-addressed files in
`artifacts/`, so the combination is a complete, restorable workspace snapshot.
State projections are exported at the ledger interval and again at terminal
completion; `commits.sqlite3` remains the authoritative, replayable history.
Docker runs also place `request.json`, `result.json`, and `sandbox.json` in the
trial directory.

The final `report.json` includes the resolved agent, model, environment, task
mapping, benchmark settings, concurrency, retry policy, sandbox configuration,
and output paths. Credential-like values in runtime option dictionaries are
redacted. `--output-dir` (also accepted as `--state-dir`) moves the runs root,
and `--report` overrides the default report location. If a task launch
fails, its retry discards the old container and volume, creates a clean
workspace, and restores the last committed workspace revision. Completed
actions are not repeated; filesystem changes from an action interrupted before
its completion commit are deliberately excluded. `--keep-sandboxes on-failure`
or `always` retains Docker resources for inspection without making them the
recovery source of truth.

The main isolation controls all have CLI defaults and can be overridden with
`--sandbox-image`, `--sandbox-cpus`, `--sandbox-memory`,
`--sandbox-pids-limit`, `--sandbox-network`, `--keep-sandboxes`, and repeated
`--sandbox-env NAME` arguments. Only allowlisted host variables are forwarded;
the built-in list covers common model-provider credentials. Environment stacks
that need dependencies beyond the base image should provide their own image
with `--sandbox-image`, optionally paired with an image-specific registry via
`--sandbox-registry-module module:create_registry`.

If that environment is already activated, `uv run` is optional and the command
is simply `corral bench ...`. The `--project` form is only needed when invoking
`uv` from elsewhere in the monorepo.

The available agent names are `ai-scientist`, `claude-code`, `codex`, `epistemic`,
`llm-planner`, `openhands`, `react`, `reflexion`, `terminus`, and
`tool-calling`. Class names and underscore spellings such as `ReActAgent` and
`tool_calling` are accepted as aliases. Claude Code, Codex, and OpenHands need
their corresponding optional package extra and report an installation hint if
it is missing.

Use `--agent-kwargs` for constructor-specific settings. Where the agent supports
them, explicit `--model`, `--api-endpoint`, and `--temperature` options take
precedence over the same keys in that JSON object:

```bash
uv run corral bench \
  --agent codex --environment samplemath --task task1 \
  --model gpt-5.4 \
  --agent-kwargs '{"reasoning_effort": "high"}'
```

`ReflexionAgent` wraps `ToolCallingAgent` by default. Select another actor with
`{"actor": "react", "actor_kwargs": {...}}`. For `AIScientistAgent`, a JSON
`config` object is validated as `AIScientistConfig`; add
`"config_profile": "sakana"` to use `SakanaAIScientistConfig`.

### Compatibility `ToolCallingAgent` script

[`run_scripts/run_tool_calling.py`](run_scripts/run_tool_calling.py) delegates to
the same local benchmark execution function as `corral bench`; it does not maintain
a separate runner implementation. The runner infers `BenchmarkTaskMetadata`
and includes the selected task's dependencies automatically. Its existing
options, defaults, and invocation remain supported:

From the repository root:

```bash
uv sync --project tasks/samplemath
uv run --project tasks/samplemath python run_scripts/run_tool_calling.py \
  --environment samplemath --list-tasks

uv run --project tasks/samplemath python run_scripts/run_tool_calling.py \
  --environment samplemath --task task4 --model openai/gpt-4o \
  --report .corral/samplemath-report.json
```

The built-in presets are `afm`, `corral_md`, `resistor_network`,
`retrosynthesis`, `samplemath`, `spectra_elucidation`, and `wetlab`. These are
fixed choices for `--environment`. Use `--env-kwargs` for
environment-specific configuration, including the common `level`, `subtasks`,
`task_config`, and `work_dir` keys. Environment dependencies and credentials
still need to be configured as described in each task package's README.

Omit `--task` to run every task returned by the environment. Pass `--task`
multiple times to select specific tasks; their required dependencies are
included automatically.

The runner applies no time limit to task execution or evaluation. Failed attempts
are retried up to `--max-attempts` times (default: 3).

For example, select SampleMath's subtask set with one environment argument:

```bash
python run_scripts/run_tool_calling.py --environment samplemath \
  --env-kwargs '{"subtasks": true}'
```

#### Langfuse observability

`run_tool_calling.py` loads the repository's `.env` before starting a run.
Configure Langfuse there with the keys and URL for your project:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
CORRAL_LANGFUSE_ENABLED=true
```

Install the optional SDK in the environment running the script. From the
repository root, use `uv sync --extra langfuse`, then retain the extra when
running with uv:

```bash
uv run --extra langfuse python run_scripts/run_tool_calling.py \
  --environment samplemath --task task1 --run-id samplemath-traced
```

When using a task package's environment instead, add `--with 'langfuse>=4,<5'`
to its `uv run --project tasks/<environment>` command. Each task produces a
trace named `corral.task.<task-id>`, with model turns, token usage, and tool
results, grouped into a Langfuse session by run ID. Traces are flushed before
the runner exits. Docker runs forward the Langfuse settings and build a
separate image with the SDK installed; custom images need the `langfuse`
extra too. Set `CORRAL_LANGFUSE_ENABLED=false` to keep only local logging.

### Scored, Multi-Trial Benchmarks

`CorralRunner` is the higher-level convenience layer for repeated trials,
evaluation, metric calculation, and reports. Use it after the direct execution
path above when those benchmark features are actually needed. Given the
`store` and model settings from the example above (keep the store open
until the benchmark finishes):

```python
from corral import CorralRunner

environments = load_environment_group("samplemath")
registry = RuntimeRegistry(
    agents={AGENT_ID: ToolCallingAgent(model=MODEL)},
    environments=environments,
)
runner = CorralRunner(
    registry,
    environments=environments,
    agent_id=AGENT_ID,
    model=MODEL,
    max_iterations=10,
    state_store=store,
)
result = await runner.run(
    "samplemath-run-2",
    task_ids=["task1", "task2"],
    trials_per_task=3,
    k_values=[1, 2, 3],
    max_parallel=4,
    max_parallel_per_task=2,
    max_parallel_evaluations=2,
    max_parallel_total=6,
)
```

`CorralRunner` handles concurrency, task retries, and dependency readiness.
Task execution and evaluation are split into two separate processes in corral.

## 🏗️ Available Environments

The framework includes several pre-built environments:

| Environment | Description |
|-------------|-------------|
| `afm` | Hardware-in-the-loop atomic force microscopy tasks |
| `corral_md` | LAMMPS molecular dynamics simulation setup and analysis |
| `resistor_network` | Hidden resistor-topology inference from measurements |
| `retrosynthesis` | Reaction-template-based synthesis planning |
| `samplemath` | Lightweight mathematical operations |
| `spectra_elucidation` | Spectroscopy-based organic structure elucidation |
| `wetlab` | Simulated qualitative inorganic ion analysis |

## 🤖 Available Agents

The framework includes several built-in agent types:

### AIScientistAgent

Uses progressive tree search to formulate hypotheses, run experiments, refine
results, and verify conclusions. It is based on Sakana AI's original
[AI Scientist v2 implementation](https://github.com/SakanaAI/AI-Scientist-v2).

```python
from corral.agents import AIScientistAgent

agent = AIScientistAgent(
    model="gpt-4o",
    evaluator_model="gpt-4o",
)
```

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

### EpistemicAgent

Extends `ToolCallingAgent` with a Decisions assessment after each public actor
turn. The initial turn receives no epistemic report and makes no
annotation API calls: the agent states initial hypotheses and justifies its first
experiments without judging nonexistent prior experiments or reports.
It classifies the seven productive motifs and eleven
antipatterns from the final graph aggregation in `reasoning_reports/analyze.py`.
The standalone registry reuses `corral.epistemic_analysis.patterns`; no graph
extraction is required. Subsequent turns ask the agent to interpret previous experiments,
check the assessment against the observations, and justify its next experiments.
Before `submit_answer`, the prompt requires a visible submission justification
connecting the answer to specific observations, considering relevant alternatives
and remaining uncertainty, and explaining why the evidence suffices. This text
accompanies the tool call; its `answer` argument keeps the task's required format.
Completion still requires `submit_answer`. Each attempted answer receives a fresh
assessment that includes its proposed answer and justification, even on the first
turn. The attempt is saved as a `submission_proposal` before the assessment. If
any selected unconstructive pattern is supported by the explanation check,
submission is held and the actor receives feedback to validate, revise, or
challenge the assessment using concrete evidence. It must call `submit_answer`
again; each attempt is reassessed. A clear assessment permits native submission.
The existing actor iteration limit bounds this loop; exhaustion does not
silently accept a held answer. An allowed surrender bypasses answer review.
Its own `epistemic/user_prompt` replaces the base tool-calling prompt. It requests
a brief scientific explanation of observations, evidence, and the purpose of
each tool call, including when using several tools in one response. It also
permits text-only explanations, which count toward the iteration limit.
The system prompt, continuation reminder, and submission tool description are
overridden for this agent; the base `ToolCallingAgent` keeps its original prompts
and reminder. These are prompt instructions, not a runtime guarantee that the
model will provide an explanation. Text-only turns are assessed before the
next actor turn; earlier epistemic reports remain excluded from the annotator.
Only public assistant text, tool calls, and observations enter the conversation
and epistemic trajectory. Provider thinking and reasoning summaries are never
promoted to visible text. Empty assistant turns (including legacy saved turns)
are omitted from model and annotator input; tool-only replies remain valid.
An empty response is accounted for without adding an assistant message. The
loop adds one continuation reminder and retries without reassessing unchanged
evidence, stopping with `protocol_failure` after three consecutive empty
responses. These requests still count toward the iteration limit and usage.
For actor `content_filter` or `refusal` stop reasons, `EpistemicAgent` retries
the unchanged request once by default, if the iteration budget allows it.
It keeps the same messages, tools, model settings, and epistemic report.
The refused attempt contributes usage but no transcript message or tool
execution; the retry does not trigger another assessment. Every attempt counts
toward the iteration limit. The retry allowance resets after a non-refused
response. If refusals persist, the trial ends with `agent_failure`, retaining
the final public refusal text and usage from all attempts.
Set `actor_refusal_max_retries=0` to disable this behavior or choose another
non-negative retry limit. This setting belongs only to `EpistemicAgent`;
the base `ToolCallingAgent` still stops immediately on a refusal.
Classifier retries remain controlled separately by `epistemic_max_retries`.

```python
from corral.agents import EpistemicAgent

agent = EpistemicAgent(
    model="openai/gpt-4o",  # Actor: any model supported by ToolCallingAgent.
    actor_refusal_max_retries=1,  # Retries after the initial refused actor call.
    epistemic_model="gpt-6-luna",  # Separate Decisions classifier.
    epistemic_explanation_model="gpt-6-luna",  # Defaults to epistemic_model.
    epistemic_explanation_concurrency=8,  # Per assessment, one request per candidate.
    epistemic_timeout_s=120,
    epistemic_max_retries=2,
    epistemic_pattern_thresholds="positive-mass-20261020-v1",  # Default policy.
)
```

```bash
uv run corral bench \
  --agent epistemic --environment samplemath --task task1 \
  --model openai/gpt-4o \
  --agent-kwargs '{"epistemic_model":"gpt-6-luna"}'
```

Set `OPENAI_API_KEY` for the classifier. Its connection can be configured with
`epistemic_api_key` and `epistemic_base_url`, independently of the actor's
`api_endpoint`. The [Decisions API](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)
currently supports `gpt-6-luna`. The annotator uses the SDK's native Decisions
helper when available, or its generic `POST /decisions` transport on older SDKs.

Offline graph annotation (`corral analyze`, with either `--annotator llm` or
`--annotator jev`) also labels the 11 behavioral checks defined in
[`additional_checks.md`](additional_checks.md). A third pass runs after node and
edge extraction and makes one request per check per trace (11 requests per trace).
Each request contains only that check's definition and the entire normalized
trace, including full tool outputs and messages after submission; `--window`,
`--overlap`, and `--tool-output-max-chars` only control the graph passes.
The package includes a copy of the definitions for installed runs; update both
copies together (a regression test checks that they match).

Saved graphs contain `trace_checks` and `trace_checks_status` alongside `nodes`
and `edges`. Each check is `present`, `absent`, or `uncertain`;
`insufficient_validation` can also be `not_applicable` when no validation tools
were available. LLM results include explanations and exact source quotes;
Decisions results retain the choice, confidence, and status probabilities.
Results appear in the comparison viewer and analysis reports. Older saved graphs
remain readable, with these checks unassessed. A context overflow or invalid
response leaves the graph incomplete and preserves its nodes and edges, with
the failure recorded in quality notes. This pass never splits or clips the trace.

Use the standalone epistemic annotator independently with chronological message dictionaries:

```python
import asyncio
import json
from pathlib import Path
from corral.epistemic_analysis import DecisionsAnnotator


async def annotate_saved_conversation():
    messages = json.loads(Path("messages.json").read_text())
    report = await DecisionsAnnotator().annotate(messages, task="Describe the task")
    Path("epistemic-report.json").write_text(json.dumps(report.to_dict(), indent=2))
    print(report.format_for_agent())


asyncio.run(annotate_saved_conversation())
```

`annotate` accepts a sequence of provider-style messages, including assistant
`tool_calls` and tool outputs. Pass only the observed prefix for online use.
An optional caller-owned `AsyncOpenAI` client can be passed as `client` to reuse
connections. An agent accepts a configured annotator through `annotator`.
An explicitly supplied annotator owns its configuration; the agent does not
replace its threshold policy.

By default, `EpistemicAgent` selects candidate patterns when
`P(active) + P(resolved) >= threshold`, using the fixed per-pattern values in
[`decision_thresholds.py`](src/corral/epistemic_analysis/decision_thresholds.py).
The `positive-mass-20261020-v1` policy contains the offline thresholds fitted on
328 completed trials pooled across agents, maximizing mean macro F1 against
the Decisions and Claude graph annotations. Those fits used 16-message windows
with 4-message overlap; online application assesses each full observed prefix.
There is no online refitting or carry-over of earlier positive decisions.
These are exploratory agreement settings, not calibrated correctness probabilities.
Four fitted thresholds are zero, so those patterns always become candidates
unless the classifier refuses; the evidence check below still applies.
This probability rule selects historical occurrence candidates for inspection;
it does not determine whether a finding can block submission.

Set `epistemic_pattern_thresholds=None` (JSON `null` in `--agent-kwargs`) to use
the original `active`-choice selection, or pass a mapping of all 18 motif names
to finite thresholds in `[0, 1]`. The standalone `DecisionsAnnotator` retains
the original selection by default; opt in with
`DecisionsAnnotator(pattern_thresholds="positive-mass-20261020-v1")`.

All 18 motifs are classified in one request as `absent`, `emerging`, `active`,
`resolved`, or `uncertain`, with probabilities. Each question includes a full
graph description specifying the events, relationships, ordering, and missing
links, along with the original notation and examples/counterexamples. Shared
glossaries define the nodes and relationships. The classifier reassesses the
trajectory independently without receiving previous reports. The original
classifier status and probabilities are preserved, even when the probability
threshold selects a pattern whose chosen status was `absent` or `emerging`.
The agent's reports show only selected **unconstructive patterns** (`antipattern`)
whose classifier status is **active** and whose separate explanation identifies
a current, unresolved scientific defect with message evidence. An explanation
cannot reactivate a finding classified as absent, emerging, resolved, or uncertain.
The saved policy identifies this feedback gate as `current-unresolved-v2`,
separately from the unchanged historical probability thresholds.
`fixed_belief_trace` retains its original meaning (no hypothesis revision in the
observed prefix), but is descriptive: it remains in the saved assessment and
never holds submission. Productive patterns are likewise omitted from actor
feedback and never hold a submission. The explanation check supports a complete
productive sequence even when its classifier status is `resolved`; successful
completion does not erase that occurrence.

The definitions distinguish an unresolved defect from a historical occurrence.
Support is assessed for the particular claim: evidence for an overall answer
does not automatically support an asserted intensity, exact match, or exclusion
of a named rival. Valid calculations, evidence-based interpretations, and
correctly qualified approximate comparisons still count as support.
`premature_commitment` can occur when an untested claim is first treated as
settled; an earlier challenge or refusal to revise is not required.
`one_sided_confirmation` requires selective commitment that neglects a specific
relevant alternative or discriminating check. Claiming that a preferred answer's
matching result excludes a named rival can establish selectivity without a
separate refusal; an existing discriminating observation can justify the
exclusion. A hypothesis surviving a test does not need a contradictory result.
`uninformative_test` requires a missing outcome
still relevant to the inference; reported execution failures are observed
outcomes, and superseded or abandoned tests do not create permanent obligations.
Corrected, retracted, or adequately qualified claims cease to be active when the
later reasoning actually removes the defect. A disclaimer alone does not repair
a conclusion that still relies on the unsupported claim, and correcting one
claim does not repair a different remaining defect. The complete criteria
and positive/negative examples are in
[`motifs.py`](src/corral/epistemic_analysis/motifs.py).
Confidence scores, probability distributions, thresholds, and
generic graph definitions are not included in the agent-facing feedback.
Refusals remain explicitly marked as uncertain in the saved assessment.
These are temporal motif assessments, not deterministic graph counts
or judgments of scientific correctness. Newly arrived results are not treated as
ignored before the agent has an opportunity to respond. This waiting rule
applies to omitted responses, not to an unsupported assertion or closure that
is already explicit in the latest message.

As soon as the Decisions response arrives and passes validation, the annotator
verifies selected candidates in parallel, with one Responses API request per
pattern and up to eight requests in flight per assessment. Set
`explanation_concurrency` on the standalone annotator or
`epistemic_explanation_concurrency` on the agent to a positive integer to change
the limit. Concurrent trials each have their own limit. All explanations must
finish and pass validation before the actor continues; a failed request cancels
the remaining explanations. Requests use structured output and the same filtered
trajectory. The explanation model receives the complete candidate graph definition
and must identify the actual observations and relationships supporting each
pattern in prose, together with a boolean support judgment. Message IDs and numbered
citations are not requested or required for feedback or submission review. It can reject unsupported
detections; these remain in the saved assessment but are omitted from agent feedback.
No explanation call is made when there are no selected candidates. The explanation
model defaults to the Decisions model and can be configured with
`explanation_model` on the standalone annotator or `epistemic_explanation_model`
on the agent. All calls use the annotator's API connection settings. Parallel
requests each include the full trajectory, increasing request volume and repeated
input tokens compared with a single combined explanation request.
If no finding passes the current-defect gate, the feedback says
"No present unconstructive epistemic patterns detected."

The actor retains all earlier reports in chronological order and receives a new
report after new public activity. A resumed session with prior agent activity
receives feedback immediately; task-only context does not trigger a judgment.
Reports and `epistemic_submission_review` continuation notices are excluded from the annotator's input,
including when resuming a session, from both classification and explanation calls.
Complete reports, probabilities, explanations, support judgments, raw responses, and
usage remain in named `epistemic_report` session messages under
`metadata.epistemic_report`. Each assessment also records `selection_probability`,
`selection_threshold`, and `selected`; the report's `selection_policy` records
the rule, all thresholds, and calibration provenance. This metadata is never sent to the actor, including
on resume; it sees only the formatted report text.
The legacy `evidence_message_ids` field remains readable in historical reports
and is empty in new assessments.
New reports store every explanation response in `raw_explanation_responses`,
ordered by motif rather than completion time. The legacy singular
`raw_explanation_response` is also populated when there is exactly one explanation
request; it is `None` for multiple requests. Token and call totals include every
explanation response exactly once.
The actor, classifier, and explanation calls contribute to token
and model-call accounting; `max_iterations` still limits actor turns. Each request
contains the full prefix without silent truncation, so classification cost grows
with the conversation. After bounded SDK retries, API or malformed-response
errors end the session with `agent_failure` rather than silently skipping feedback.
A failed submission assessment preserves the proposed answer without accepting it.

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

## 💾 Commit Ledger

Corral persists one small, immutable, typed commit for every durable event in a
SQLite ledger. `parent_hash` links each ordinary commit to the current branch
head, while `based_on_hash` records the projection the author actually saw.
`ExecutionState` is rebuilt from the ledger plus occasional replay snapshots;
agents receive an authorized `AgentContext`, not the complete execution trace.

Agent and tool authors are bound capabilities. Subagents share the same linear
branch and appear in the parent conversation as ordinary tool calls with
automatic result summaries. Their full conversations and state remain private
unless the parent chooses to inspect and import selected details. Parallel tool
completions are stored in real completion order and presented to the model in
declared action order. Shared environment and workspace effects carry revision
preconditions, so overlapping writes conflict instead of silently rebasing.
Explicit experiments use `SQLiteCommitStore.create_branch()`; spawning a
subagent never creates a branch.

Subagent inspection is opt-in per agent. Agents declaring
`AgentSessionCapabilities(inspect_subagents=True)` receive an
`inspect_subagent` tool that returns a bounded authorized child context through
the normal tool-call ledger path. AI Scientist enables it; other built-in
agents do not expose it.

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

3. **Register your agent for execution**

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
- **Community standards**: See our [Code of Conduct](CODE_OF_CONDUCT.md)

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
