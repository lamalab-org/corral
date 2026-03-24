"""Build and publish trace datasets to the Hugging Face Hub.

This script reads Corral summary reports and per-trial trace JSON files,
matches them by agent and verbosity, validates the expected directory layout,
assembles a tabular dataset, and uploads grouped subsets to the Hub.
"""

import ast
import json
import re
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import fire
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from loguru import logger
from report_constants import (
    AGENTS,
    ALLOWED_CATEGORIES,
    ALLOWED_MODELS,
    LEVEL_PREFIX,
    VERBOSITY,
)


def load_json(path):
    """Load a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """

    path = Path(path)
    with path.open() as f:
        return json.load(f)


def extract_observations(messages):
    """Extract tool observations from a trace message list.

    The first user message is skipped because it contains the original task
    prompt rather than a tool observation.

    Args:
        messages: Sequence of trace messages.

    Returns:
        A list of parsed observation payloads.
    """

    observations = []

    if not messages:
        return observations

    first_user_skipped = False

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if not first_user_skipped:
                first_user_skipped = True
                continue

            if "Observation:" not in content:
                continue

            payload = content.split("Observation:", 1)[1].strip()

        elif role == "tool":
            payload = content.strip()

        else:
            continue

        try:
            observations.append(json.loads(payload))
            continue
        except json.JSONDecodeError:
            pass

        try:
            observations.append(ast.literal_eval(payload))
        except Exception:
            logger.warning(f"[WARNING] Failed to parse observation: {payload}")
            logger.info(payload)

    return observations


def check_tool_calls(trial_metadata, trace_json):
    """Compare reported tool calls with parsed observations from a trace.

    Args:
        trial_metadata: Trial metadata from the summary JSON.
        trace_json: Full trace payload.
    """

    tool_calls_report = trial_metadata.get("tool_calls")

    messages = trace_json.get("messages")

    observations = extract_observations(messages)

    if observations != tool_calls_report:
        logger.warning("observation does not match tool_calls_report")
        logger.info(f" number of observations : {len(observations)}")
        logger.info(f"number of tool_calls_report : {len(tool_calls_report)}")


def normalize(text: str):
    """Normalize a label for fuzzy agent and verbosity matching.

    Args:
        text: Raw file or directory name.

    Returns:
        Lowercased text with separators normalized to spaces.
    """

    return re.sub(r"[-_]+", " ", text.lower())


def extract_pair(name: str):
    """Extract agent and verbosity labels from a name.

    Args:
        name: File or directory name to inspect.

    Returns:
        A tuple of agent and verbosity when both can be resolved, otherwise
        None.
    """

    name = normalize(name)

    agent = None
    verbosity = None

    for key, val in AGENTS.items():
        if key in name:
            agent = val
            break

    for v in VERBOSITY:
        if v in name:
            verbosity = v
            break

    if agent and verbosity:
        return (agent, verbosity)

    return None


def extract_timestamp(path: Path, task_name: str):
    """Extract the UTC timestamp encoded in a trace filename.

    Args:
        path: Trace file path.
        task_name: Task prefix used in the filename.

    Returns:
        Parsed UTC timestamp.
    """

    suffix = path.stem[len(task_name) + 1 :]

    return datetime.strptime(suffix, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def collect_traces_for_task(trace_dir: Path, task_name: str):
    """Collect and order trace files for a task.

    Args:
        trace_dir: Directory containing trace JSON files.
        task_name: Task name prefix to match.

    Returns:
        Sorted trace paths for the task.
    """

    traces = [
        f for f in trace_dir.rglob("*.json") if f.stem.startswith(f"{task_name}_")
    ]

    traces.sort(key=lambda p: extract_timestamp(p, task_name))

    return traces


def build_trials_from_pair(config, pair):
    """Build dataset rows for one summary/trace pairing.

    Args:
        config: Dataset configuration metadata.
        pair: Matched summary and trace locations for one agent/verbosity pair.

    Returns:
        A tuple with the dataset rows, missing trace count, and extra trace
        count.
    """

    dataset = []

    if not pair["json_files"] or not pair["directories"]:
        return dataset, 0, 0

    summary_path = Path(pair["json_files"][0])
    trace_dir = Path(pair["directories"][0])

    summary = load_json(summary_path)
    task_results = summary.get("task_results", {})

    missing_files = 0
    extra_traces = 0

    for task_name in task_results:
        trials = task_results.get(task_name).get("trials")
        trials = sorted(trials, key=lambda x: int(x["trial_id"]))

        traces = collect_traces_for_task(trace_dir, task_name)

        if len(trials) > len(traces):
            logger.warning(
                f"[WARNING] {task_name}: "
                f"{len(trials)} trials but only {len(traces)} traces."
            )
            missing_files += len(trials) - len(traces)

        elif len(trials) < len(traces):
            logger.warning(
                f"[WARNING] {task_name}: "
                f"{len(trials)} trials but {len(traces)} traces found. "
                "Extra traces will be ignored."
            )
            extra_traces += len(traces) - len(trials)

        elif len(trials) == len(traces):
            for trial_meta, trace_path in zip_longest(trials, traces):
                trace_json = load_json(trace_path)

                messages = trace_json.get("messages")
                model_version = trace_json.get("model")
                timestamp = trace_json.get("timestamp")

                trial_id = (
                    trial_meta.get("trial_id") if trial_meta is not None else None
                )

                entry = {
                    **config,
                    "agent": pair["agent"],
                    "verbosity": pair["verbosity"],
                    "task_name": task_name,
                    "trial_id": trial_id,
                    **(
                        {k: v for k, v in trial_meta.items() if k != "tool_calls"}
                        if trial_meta
                        else {}
                    ),
                    "messages": messages,
                    "trace_file": str(trace_path) if trace_path else None,
                    "model_version": model_version,
                    "timestamp": timestamp,
                }

                dataset.append(entry)

    return dataset, missing_files, extra_traces


def match_agent_verbosity(category_path: Path):
    """Group files and directories by detected agent and verbosity.

    Args:
        category_path: Category directory containing reports and traces.

    Returns:
        Grouped metadata for each detected agent/verbosity pair.
    """

    grouped = {}

    for item in category_path.iterdir():
        if item.is_dir() and item.name.lower().startswith("logprobs"):
            continue

        pair = extract_pair(item.name)
        if pair is None:
            logger.warning(f"No report could be found for {item.name}. Ignoring.")
            continue

        if pair not in grouped:
            grouped[pair] = {
                "agent": pair[0],
                "verbosity": pair[1],
                "directories": [],
                "json_files": [],
            }

        if item.is_dir():
            grouped[pair]["directories"].append(str(item))

        elif item.is_file() and item.suffix == ".json":
            grouped[pair]["json_files"].append(str(item))

    return list(grouped.values())


def validate_path(path: str):
    """Validate the expected report directory structure.

    Args:
        path: Input path expected to contain model, environment, level, and
            category components.

    Returns:
        Tuple of model, level, category, and environment.

    Raises:
        ValueError: If the path does not match the expected structure or uses
            unsupported values.
    """

    p = Path(path)

    if len(p.parts) < 2:
        raise ValueError(
            "Path must follow structure: {model}/{environment} or "
            "{model}/{environment}/{level}/{category}"
        )

    if len(p.parts) >= 4 and p.parts[-2].startswith(LEVEL_PREFIX):
        model = p.parts[-4]
        level = p.parts[-2]
        category = p.parts[-1]
        environment = p.parts[-3]
    else:
        model = p.parts[-2]
        environment = p.parts[-1]
        level = None
        category = None

    normalized_model = model.replace("-", "_")

    if normalized_model not in ALLOWED_MODELS:
        raise ValueError(f"Invalid model '{model}'. Allowed models: {ALLOWED_MODELS}")

    if level is not None and not level.startswith(LEVEL_PREFIX):
        raise ValueError(f"Invalid level '{level}'. Must start with '{LEVEL_PREFIX}'")

    if category is not None and category not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Allowed: {ALLOWED_CATEGORIES}"
        )

    return normalized_model, level, category, environment


def push_trials_to_hub(trials_df: pd.DataFrame, dataset_name: str):
    """Upload grouped trace subsets to the Hugging Face Hub.

    Args:
        trials_df: Dataframe containing all trial rows.
        dataset_name: Destination dataset repository name.
    """

    api = HfApi()

    api.create_repo(
        repo_id=dataset_name,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    group_cols = ["model", "env", "level", "category", "agent", "verbosity"]

    for subset_keys, subset_df in trials_df.groupby(group_cols):
        model, env, level, category, agent, verbosity = subset_keys

        subset_name = f"{model}-{env}-{level}-{category}-{agent}-{verbosity}-traces"

        logger.info(f"\nUploading subset: {subset_name}")

        dataset = Dataset.from_pandas(subset_df.reset_index(drop=True))

        dataset.push_to_hub(
            dataset_name,
            config_name=subset_name,
            private=False,
        )


def build_dataset_configs(path: str):
    """Build trace dataset rows from a report directory and upload them.

    Args:
        path: Root path for one model/environment report tree.

    Raises:
        ValueError: If the assembled dataframe contains missing values.
    """

    p = Path(path)

    model, level, category, env = validate_path(path)

    configs = []
    level_dirs = [d for d in p.rglob("*") if d.is_dir() and d.name.startswith("level_")]

    if not level_dirs:
        level_dirs = [p]

    for level_path in level_dirs:
        level = level_path.name if level_path.name.startswith("level_") else "level_1"

        for category_dir in level_path.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            config = {
                "model": model,
                "env": env,
                "level": level,
                "category": category,
                "path": str(category_dir),
            }

            configs.append(config)

    all_trials = []
    missing_files = 0
    extra_traces = 0

    for config in configs:
        category_path = Path(config["path"])

        pairs = match_agent_verbosity(category_path)

        logger.info(f"path : {category_path}")

        for pair in pairs:
            trials, missing, extra = build_trials_from_pair(config, pair)

            all_trials.extend(trials)
            missing_files += missing
            extra_traces += extra

    if missing_files > 0:
        logger.error(f"Can't push!!! Number of missing files : {missing_files}")
        return

    if extra_traces > 0:
        logger.error(f"Can't push!!! Number of extra traces : {extra_traces}")
        return

    trials_df = pd.DataFrame(all_trials)

    # "error" is sparse by design (only present on failed trials).
    # Other optional fields may be None when an agent crashes before
    # submitting, which is valid data.  Fill them with typed sentinels so the
    # strict NaN check below only fires for truly unexpected missing columns.
    OPTIONAL_STR_COLS = ["error", "submitted_answer"]
    OPTIONAL_INT_COLS = ["total_calls", "successful_calls", "failed_calls"]
    OPTIONAL_LIST_COLS = ["tools_used", "error_types"]

    for col in OPTIONAL_STR_COLS:
        if col in trials_df.columns:
            trials_df[col] = trials_df[col].fillna("")

    for col in OPTIONAL_INT_COLS:
        if col in trials_df.columns:
            trials_df[col] = trials_df[col].fillna(0).astype(int)

    for col in OPTIONAL_LIST_COLS:
        if col in trials_df.columns:
            trials_df[col] = trials_df[col].apply(
                lambda v: v if isinstance(v, list) else []
            )

    if trials_df.isna().any().any():
        nan_counts = trials_df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]

        raise ValueError(f"NaNs detected in trials dataframe:\n{nan_cols}")

    DATASET_NAME = "jablonkagroup/corral-traces"
    push_trials_to_hub(trials_df, DATASET_NAME)


if __name__ == "__main__":
    fire.Fire(build_dataset_configs)
