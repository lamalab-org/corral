import argparse
import json
import re
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

METRIC_KEYS = [
    "success_rate",
    "average_score",
    "pass@1",
    "pass@2",
    "pass@3",
    "pass@4",
    "pass@5",
    "pass^1",
    "pass^2",
    "pass^3",
    "pass^4",
    "pass^5",
]


def load_json(path):
    path = Path(path)
    with path.open() as f:
        return json.load(f)


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


def get_scores(config, report_path):
    task_dataset = []
    run_dataset = []

    report = load_json(report_path)
    task_results = report.get("task_results", {})

    pair = extract_pair(report_path)

    if pair is None:
        logger.warning(f"extract_pair returned None for report: {report_path}")
        agent = None
        verbosity = None

    else:
        agent, verbosity = pair

    overall_metrics = report.get("metrics", {})

    run_scores = {k: overall_metrics.get(k) for k in overall_metrics}

    run_dataset.append(
        {
            **config,
            "agent": agent,
            "verbosity": verbosity,
            **run_scores,
        }
    )

    for task_name, task_json in task_results.items():
        scores = {k: task_json.get(k) for k in METRIC_KEYS if k in task_json}

        entry = {
            **config,
            "agent": agent,
            "verbosity": verbosity,
            "task_name": task_name,
            **scores,
        }

        task_dataset.append(entry)

    return task_dataset, run_dataset


def get_jsons(category_path: Path):
    return [
        str(item)
        for item in category_path.iterdir()
        if item.is_file() and item.suffix == ".json"
    ]


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

    if model == "gpt-4o":
        model = "gpt_4o"

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

    all_reports = []
    overall_reports = []

    for config in configs:
        category_path = Path(config["path"])

        report_paths = get_jsons(category_path)

        for report_path in report_paths:
            report, overall_report = get_scores(config, report_path)
            all_reports.extend(report)
            overall_reports.extend(overall_report)

    DATASET_NAME = "jablonkagroup/corral-reports"
    api = HfApi()

    api.create_repo(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    trials_df = pd.DataFrame(all_reports)

    group_cols = ["model", "env", "level", "category", "agent", "verbosity"]

    for subset_keys, subset_df in trials_df.groupby(group_cols):
        model, env, level, category, agent, verbosity = subset_keys

        subset_name = (
            f"{model}-{env}-{level}-{category}-{agent}-{verbosity}-task_reports"
        )

        logger.info(f"\nUploading subset: {subset_name}")

        dataset = Dataset.from_pandas(subset_df.reset_index(drop=True))

        dataset.push_to_hub(
            DATASET_NAME,
            config_name=subset_name,
            private=False,
        )

    overall_df = pd.DataFrame(overall_reports)

    group_cols = ["model", "env", "level", "category", "agent", "verbosity"]

    for subset_keys, subset_df in overall_df.groupby(group_cols):
        model, env, level, category, agent, verbosity = subset_keys

        subset_name = (
            f"{model}-{env}-{level}-{category}-{agent}-{verbosity}-overall_reports"
        )

        logger.info(f"\nUploading subset: {subset_name}")

        dataset = Dataset.from_pandas(subset_df.reset_index(drop=True))

        dataset.push_to_hub(
            DATASET_NAME,
            config_name=subset_name,
            private=False,
        )


if __name__ == "__main__":
    main()
