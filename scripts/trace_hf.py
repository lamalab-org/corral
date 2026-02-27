import argparse
import ast
import json
import re
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from loguru import logger

AGENTS = {
    "react": "ReActAgent",
    "reactagent": "ReActAgent",
    "tool calling": "ToolCallingAgent",
    "toolcalling": "ToolCallingAgent",
    "toolcallingagent": "ToolCallingAgent",
}

VERBOSITY = {"brief", "comprehensive", "workflow"}


def load_json(path):
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def extract_observations(messages):
    observations = []

    if not messages:
        return observations

    first_user_skipped = False

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        # -----------------------------
        # USER ROLE
        # -----------------------------
        if role == "user":
            # ignore first user message
            if not first_user_skipped:
                first_user_skipped = True
                continue

            if "Observation:" not in content:
                continue

            payload = content.split("Observation:", 1)[1].strip()

        # -----------------------------
        # TOOL ROLE
        # -----------------------------
        elif role == "tool":
            payload = content.strip()

        else:
            continue

        # -----------------------------
        # Try parsing
        # -----------------------------
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
    tool_calls_report = trial_metadata.get("tool_calls")

    messages = trace_json.get("messages")

    observations = extract_observations(messages)

    if observations != tool_calls_report:
        logger.warning("observation does not match tool_calls_report")
        logger.info(f" number of observations : {len(observations)}")
        logger.info(f"number of tool_calls_report : {len(tool_calls_report)}")


def normalize(text: str):
    # keep word boundaries
    return re.sub(r"[-_]+", " ", text.lower())


def extract_pair(name: str):
    name = normalize(name)

    agent = None
    verbosity = None

    # agent detection
    for key, val in AGENTS.items():
        if key in name:
            agent = val
            break

    # verbosity detection
    for v in VERBOSITY:
        if v in name:
            verbosity = v
            break

    if agent and verbosity:
        return (agent, verbosity)

    return None


def extract_timestamp(path: Path, task_name: str):
    # remove "{task_name}_"
    suffix = path.stem[len(task_name) + 1 :]

    # suffix = "20250803_224644"
    return datetime.strptime(suffix, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def collect_traces_for_task(trace_dir: Path, task_name: str):
    traces = [
        f for f in trace_dir.rglob("*.json") if f.stem.startswith(f"{task_name}_")
    ]

    # sort by timestamp encoded in filename
    traces.sort(key=lambda p: extract_timestamp(p, task_name))

    return traces


def build_trials_from_pair(config, pair):
    dataset = []

    if not pair["json_files"] or not pair["directories"]:
        return dataset

    summary_path = Path(pair["json_files"][0])
    trace_dir = Path(pair["directories"][0])

    summary = load_json(summary_path)
    task_results = summary.get("task_results", {})

    missing_files = 0

    for task_name in task_results:
        trials = task_results.get(task_name).get("trials")
        trials = sorted(trials, key=lambda x: int(x["trial_id"]))

        traces = collect_traces_for_task(trace_dir, task_name)

        missing_files += len(trials) - len(traces)

        if len(trials) > len(traces):
            logger.warning(
                f"[WARNING] {task_name}: "
                f"{len(trials)} trials but only {len(traces)} traces."
            )
            # for trial_id, trial_meta in enumerate(trials):
            #     trace_path = None
            #     messages = None
            #     model_version = None
            #     timestamp = None

        elif len(trials) < len(traces):
            logger.warning(
                f"[WARNING] {task_name}: "
                f"{len(trials)} trials but {len(traces)} traces found. "
                "Extra traces will be ignored."
            )

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

    return dataset, missing_files


# ---------------------------------
# Core matcher
# ---------------------------------
def match_agent_verbosity(category_path: Path):
    grouped = {}

    for item in category_path.iterdir():
        # ✅ Ignore logprobs directories
        if item.is_dir() and item.name.lower().startswith("logprobs"):
            continue

        pair = extract_pair(item.name)
        if pair is None:
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


def main():
    parser = argparse.ArgumentParser(description="Build dataset configs")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root reports directory path",
    )

    args = parser.parse_args()
    path = args.path

    p = Path(path)

    env = p.parts[-1]
    model = p.parts[-2]

    if model == "claude":
        model = "claude_sonnet_45"

    if model == "gpt-oss-120b":
        model = "gpt_oss_120b"

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

    for config in configs:
        category_path = Path(config["path"])

        pairs = match_agent_verbosity(category_path)

        for pair in pairs:
            trials, missing = build_trials_from_pair(config, pair)

            all_trials.extend(trials)
            missing_files += missing

    if missing_files > 0:
        logger.error(f"Can't push!!! Number of missing files : {missing_files}")

    else:
        DATASET_NAME = "jablonkagroup/corral-traces"
        api = HfApi()

        api.create_repo(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        trials_df = pd.DataFrame(all_trials)

        group_cols = ["model", "env", "level", "category"]

        for subset_keys, subset_df in trials_df.groupby(group_cols):
            model, env, level, category = subset_keys

            subset_name = f"{model}_{env}_{level}_{category}"

            logger.info(f"\nUploading subset: {subset_name}")

            dataset = Dataset.from_pandas(subset_df.reset_index(drop=True))

            dataset.push_to_hub(
                DATASET_NAME,
                config_name=subset_name,
                private=False,
            )


if __name__ == "__main__":
    main()
