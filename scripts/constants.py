"""Shared constants for corral benchmark scripts.

All named constants used across :mod:`push_reports_to_hf`, :mod:`push_qa_to_hf`,
:mod:`push_trace_to_hf`, and :mod:`combine_md_reports` live here so that the
individual scripts stay lean and the canonical values are defined in one place.
"""

# push_reports_to_hf
HF_REPO_REPORTS = "jablonkagroup/corral_runs_reports"

# push_qa_to_hf
HF_REPO_QA_REPORTS = "jablonkagroup/corral-QAs-reports"
HF_REPO_QA_TOPIC = "jablonkagroup/corral-QAs-topic_reports"

# push_trace_to_hf
HF_REPO_TRACE = "jablonkagroup/corral-oss-trace-logprobs"

# push_intervention_reports_to_hf / push_intervention_traces_to_hf
HF_REPO_INTERVENTION_REPORTS = "jablonkagroup/corral-intervention-reports"
HF_REPO_INTERVENTION_TRACES = "jablonkagroup/corral-intervention-traces"

# ---------------------------------------------------------------------------
# Report file-walking
# ---------------------------------------------------------------------------

# Prevents accidentally ingesting agent logs or cache files that share the
# .json extension.
SKIP_DIR_PREFIXES: tuple[str, ...] = (
    "agent_logs",
    "checkpoints",
    "__pycache__",
    ".ipynb_checkpoints",
    "logprobs",
    "metrics",
)

# Multiple name variants per model exist across old and new report dirs; map
# all to a single underscore-safe key.
MODEL_CANONICAL: dict[str, str] = {
    "claude_sonnet_45": "claude_4_5",
    "claude_45_sonnet": "claude_4_5",
    "claude-sonnet-4-5": "claude_4_5",
    "claude_45": "claude_4_5",
    "claude": "claude_4_5",
    "gpt-4o": "gpt_4o",
    "gpt_4o": "gpt_4o",
    "gpt4o": "gpt_4o",
    "gpt_oss_120b": "gpt_oss_120b",
    "gpt_oss_120": "gpt_oss_120b",
    "gpt-oss-120b": "gpt_oss_120b",
}

# Display names use dashes (human-readable form) while canonical keys use
# underscores (safe for column names and config identifiers).
MODEL_DISPLAY: dict[str, str] = {
    "claude_4_5": "claude-4.5",
    "gpt_4o": "gpt-4o",
    "gpt_oss_120b": "gpt-oss-120b",
}

# V2 top-level dir names may differ from the v1 filename fragments; a separate
# map avoids false-positive substring matches.
MODEL_DIR_MAP: dict[str, str] = {
    "claude_sonnet_45": "claude_4_5",
    "gpt-4o": "gpt_4o",
    "gpt_oss_120b": "gpt_oss_120b",
}

# Both top-level dir names and sub-directory names (e.g. 'melting' inside
# 'md') must resolve to the same canonical env key for correct grouping.
ENV_MAP: dict[str, str] = {
    "afm": "afm",
    "catalyst": "catalyst",
    "md": "md",
    "melting": "md",
    "quenching": "md",
    "surface_energy": "md",
    "ml": "ml",
    "retrosynthesis": "retro",
    "retro": "retro",
    "resistor": "resistor",
    "resistor_networks": "resistor",
    "resistor_network": "resistor",
    "spectra": "spectra",
    "spectra_elucidation": "spectra",
    "wetlab": "wetlab",
}

# Reports older than gpt-oss-120b used snake_case; newer reports already use
# Title Case — map both so all records share the same column names.
METRICS_KEY_MAP: dict[str, str] = {
    "average_score": "Average Score",
    "overall_success_rate": "Overall Success Rate",
    **{f"pass@{i}": f"Pass@{i}" for i in range(1, 6)},
    **{f"pass^{i}": f"Pass^{i}" for i in range(1, 6)},
    "total_tasks": "Total Tasks",
    "tool_verbosity": "Tool Verbosity",
    "surrendered_trials": "Total Surrendered Trials",
    "total_tool_execution_duration": "Total Tool Execution Duration",
    "total_benchmark_duration": "Total Benchmark Duration",
}

# Task-level aggregates carry a 'Task ' prefix to avoid column-name collisions
# with identically named top-level metrics.
TASK_KEY_MAP: dict[str, str] = {
    "success_rate": "Task Success Rate",
    "average_score": "Task Average Score",
    "total_token_usage": "Task Total Token Usage",
    **{f"pass@{i}": f"Task Pass@{i}" for i in range(1, 6)},
    **{f"pass^{i}": f"Task Pass^{i}" for i in range(1, 6)},
}

# Keys that receive special handling (token-usage & tool-calls).
METRICS_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "total_token_usage",
        "Total Token Usage",
        "total_tool_calls",
        "successful_tool_calls",
        "failed_tool_calls",
        "Total Tool Calls",
    }
)

# Full dataset schema
ALL_COLUMNS: list[str] = [
    "model",
    "agent_type",
    "environment",
    "level",
    "category",
    "Average Score",
    "Overall Success Rate",
    *[f"Pass@{i}" for i in range(1, 6)],
    *[f"Pass^{i}" for i in range(1, 6)],
    "Total Tasks",
    "Tool Verbosity",
    "Total Tool Calls",
    "Total Surrendered Trials",
    "Total Tool Execution Duration",
    "Total Benchmark Duration",
    "Overall Average Duration",
    "Overall Total Duration",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "Task Results",
]

MD_ENVS: list[str] = ["melting", "quenching", "surface_energy"]

AGENT_TYPES: list[str] = ["react", "tool_calling"]

AGENT_CLASS_NAMES: dict[str, str] = {
    "react": "ReActAgent",
    "tool_calling": "ToolCallingAgent",
}

VERBOSITIES: list[str] = ["brief", "comprehensive", "workflow"]

KNOWN_QA_MODELS: set[str] = {"claude", "gpt", "gpt_oss"}

# Every canonical key produced by MODEL_CANONICAL must have an entry in
# MODEL_DISPLAY; without it _build_record silently stores the raw underscore
# key in the 'model' column, breaking dataset grouping and UI display.
_missing_display = set(MODEL_CANONICAL.values()) - set(MODEL_DISPLAY)
assert not _missing_display, f"MODEL_CANONICAL values not found in MODEL_DISPLAY — add entries for: {_missing_display}"

# Same requirement for the v2 top-level directory map.
_missing_v2_display = set(MODEL_DIR_MAP.values()) - set(MODEL_DISPLAY)
assert not _missing_v2_display, f"MODEL_DIR_MAP values not found in MODEL_DISPLAY — add entries for: {_missing_v2_display}"

# Every Title-Case name that METRICS_KEY_MAP produces must be declared in
# ALL_COLUMNS; an undeclared column causes unpredictable type inference.
_unmapped_metrics = set(METRICS_KEY_MAP.values()) - set(ALL_COLUMNS)
assert (
    not _unmapped_metrics
), f"METRICS_KEY_MAP targets missing from ALL_COLUMNS — add: {_unmapped_metrics}"

# AGENT_CLASS_NAMES must cover all AGENT_TYPES.
assert set(AGENT_TYPES) == set(AGENT_CLASS_NAMES.keys()), (
    f"AGENT_CLASS_NAMES keys {set(AGENT_CLASS_NAMES.keys())} must match "
    f"AGENT_TYPES {set(AGENT_TYPES)}"
)

# rstrip('s') is only safe for the two known section names.
assert (
    "tasks".rstrip("s") == "task"
), "rstrip('s') no longer produces expected singular form for 'tasks'."
assert (
    "subtasks".rstrip("s") == "subtask"
), "rstrip('s') no longer produces expected singular form for 'subtasks'."

del _missing_display, _missing_v2_display, _unmapped_metrics
