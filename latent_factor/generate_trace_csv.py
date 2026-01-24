#!/usr/bin/env python3
"""
Script to generate overall_trace.csv from agent run reports.

CSV columns:
- model: model name (e.g., claude-sonnet-4.5, gpt-4o)
- environment: task environment (e.g., afm, catalyst, spectra)
- task: task name + trial id
- level: difficulty level (e.g., level_1, level_2)
- scaffold: agent type (React or Tool_calling)
- verbosity: verbosity level (brief, comprehensive, workflow)
- success: 0 or 1
- reasoning_score: optional (left empty for now)
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Model directory configurations (from task_subtask.py)
MODEL_CONFIGS = {
    "Claude-4.5": {
        "dir": "claude_sonnet_45",
        "display_name": "claude-sonnet-4.5",
        "prefix_map": {
            "afm": "claude-sonnet-4-5",
            "catalyst": "claude",
            "md": "claude_45",
            "ml": "claude",
            "resistor": "claude",
            "retrosynthesis": "claude_45_sonnet",
            "spectra": "claude_45_sonnet",
            "wetlab": "claude_sonnet_45",
        },
    },
    "GPT-4o": {
        "dir": "gpt-4o",
        "display_name": "gpt-4o",
        "prefix_map": {
            "afm": "gpt-4o",
            "catalyst": "gpt4o",
            "md": "gpt_4o",
            "ml": "gpt4o",
            "resistor": "gpt4o",
            "retrosynthesis": "gpt_4o",
            "spectra": "gpt_4o",
            "wetlab": "gpt_4o",
        },
    },
    "GPT-OSS-120B": {
        "dir": "gpt-oss-120b",
        "display_name": "gpt-oss-120b",
        "prefix_map": {
            "spectra": "gpt_oss_120",
        },
    },
}

# Environments to scan
ENVIRONMENTS = [
    "afm",
    "catalyst",
    "md",
    "ml",
    "resistor",
    "retrosynthesis",
    "spectra",
    "wetlab",
]

# Verbosity levels
VERBOSITIES = ["brief", "comprehensive", "workflow"]

# Agent types
AGENTS = ["react", "tool_calling"]

# MD subtypes
MD_SUBTYPES = ["melting", "quenching", "surface_energy"]


def get_json_path(
    base_path: Path,
    model_name: str,
    env: str,
    agent: str,
    level: str,
    verbosity: str,
    subtype: Optional[str] = None,
) -> Path:
    """Construct the path to the JSON file based on environment and naming convention."""
    model_config = MODEL_CONFIGS[model_name]
    model_dir = model_config["dir"]
    prefix = model_config["prefix_map"].get(env)

    if prefix is None:
        return None

    tasks_path = base_path / model_dir / env / level / "tasks"

    if env == "catalyst":
        agent_name = agent if agent == "react" else "toolcalling"
        filename = f"{prefix}-{agent_name}-catalyst-{verbosity}_verbosity_single.json"

    elif env == "retrosynthesis":
        lvl_num = level.split("_")[1]
        filename = f"{prefix}-{agent}-retro_lvl{lvl_num}_env-{verbosity}_verbosity.json"

    elif env == "afm":
        task_num = level.split("_")[1]
        agent_name = "React" if agent == "react" else "Tool_calling"
        # Try primary naming convention
        filename = f"{prefix}-{agent_name}-task_{task_num}-{verbosity}_verbosity.json"
        # Check if file exists, if not try alternate naming (underscore instead of hyphen)
        primary_path = tasks_path / filename
        if not primary_path.exists() and agent == "tool_calling":
            # Some files use underscore between prefix and Tool_calling
            filename = (
                f"{prefix}_Tool_calling-task_{task_num}-{verbosity}_verbosity.json"
            )

    elif env == "ml":
        agent_name = agent if agent == "react" else "toolcalling"
        filename = f"{prefix}-{agent_name}-ml-{verbosity}_verbosity_single.json"

    elif env == "resistor":
        agent_name = agent if agent == "react" else "toolcalling"
        filename = (
            f"{prefix}-{agent_name}-resistor_network-{verbosity}_verbosity_single.json"
        )

    elif env == "md":
        if subtype is None:
            return None
        filename = (
            f"{prefix}-{agent}-{subtype}-{verbosity}-task-{level}-6_Nov_optimised.json"
        )

    elif env == "spectra":
        lvl_num = level.split("_")[1]
        agent_name = agent if agent == "react" else "tool_calling"
        filename = (
            f"{prefix}-{agent_name}-spectra_lvl{lvl_num}_env-{verbosity}_verbosity.json"
        )

    elif env == "wetlab":
        lvl_num = level.split("_")[1]
        agent_name = "ReAct" if agent == "react" else "Tool_Calling"
        filename = f"{prefix}-{agent_name}-WetLab_Level_{lvl_num}-{verbosity}.json"

    else:
        return None

    return tasks_path / filename


def parse_json_file(json_path: Path) -> List[Dict[str, Any]]:
    """Parse a JSON report file and extract trial data."""
    results = []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading {json_path}: {e}")
        return results

    task_results = data.get("task_results", {})

    for task_name, task_data in task_results.items():
        trials = task_data.get("trials", [])

        for trial in trials:
            trial_id = trial.get("trial_id", "unknown")
            success = trial.get("success", False)

            results.append(
                {
                    "task_name": task_name,
                    "trial_id": trial_id,
                    "success": 1 if success else 0,
                }
            )

    return results


def discover_levels(base_path: Path, model_dir: str, env: str) -> List[str]:
    """Discover available levels for a given model and environment."""
    env_path = base_path / model_dir / env
    if not env_path.exists():
        return []

    levels = []
    for item in env_path.iterdir():
        if item.is_dir() and item.name.startswith("level_"):
            levels.append(item.name)

    return sorted(levels)


def process_environment(
    base_path: Path,
    model_name: str,
    env: str,
    level: str,
    agent: str,
    verbosity: str,
    subtype: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process a single environment configuration and return rows."""
    rows = []

    json_path = get_json_path(
        base_path, model_name, env, agent, level, verbosity, subtype
    )

    if json_path is None:
        return rows

    if not json_path.exists():
        logger.warning(f"File not found: {json_path}")
        return rows

    logger.info(f"Processing: {json_path}")

    trials = parse_json_file(json_path)

    model_display = MODEL_CONFIGS[model_name]["display_name"]
    scaffold = "React" if agent == "react" else "Tool_calling"

    # For md environment, include subtype in environment name
    env_name = f"{env}_{subtype}" if subtype else env

    for trial in trials:
        row = {
            "model": model_display,
            "environment": env_name,
            "task": trial["task_name"],  # Remove the _trial_{trial_id} part
            "level": level,
            "scaffold": scaffold,
            "verbosity": verbosity,
            "success": trial["success"],
            "reasoning_score": "",
        }
        rows.append(row)

    return rows


def main():
    # Base directory for reports
    base_path = Path(__file__).parent

    all_rows = []
    files_found = 0
    files_missing = 0

    # Iterate through all model configurations
    for model_name, model_config in MODEL_CONFIGS.items():
        model_dir = model_config["dir"]
        model_path = base_path / model_dir

        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            continue

        logger.info(f"Processing model: {model_name}")

        # Iterate through environments
        for env in ENVIRONMENTS:
            # Check if this model has a prefix for this environment
            if env not in model_config["prefix_map"]:
                continue

            # Discover levels
            levels = discover_levels(base_path, model_dir, env)

            if not levels:
                continue

            logger.info(f"  Environment: {env}, Levels: {levels}")

            for level in levels:
                for agent in AGENTS:
                    for verbosity in VERBOSITIES:
                        # Special handling for md environment (has subtypes)
                        if env == "md":
                            for subtype in MD_SUBTYPES:
                                rows = process_environment(
                                    base_path,
                                    model_name,
                                    env,
                                    level,
                                    agent,
                                    verbosity,
                                    subtype,
                                )
                                if rows:
                                    files_found += 1
                                    all_rows.extend(rows)
                                else:
                                    files_missing += 1
                        else:
                            rows = process_environment(
                                base_path, model_name, env, level, agent, verbosity
                            )
                            if rows:
                                files_found += 1
                                all_rows.extend(rows)
                            else:
                                files_missing += 1

    # Write CSV
    output_path = base_path / "overall_trace_v2.csv"

    fieldnames = [
        "model",
        "environment",
        "task",
        "level",
        "scaffold",
        "verbosity",
        "success",
        "reasoning_score",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Generated: {output_path}")
    logger.info(f"Total rows: {len(all_rows)}")
    logger.info(f"Files found: {files_found}")
    logger.info(f"Files missing: {files_missing}")

    # logger.info summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary by model:")
    logger.info("=" * 60)
    for model_config in MODEL_CONFIGS.values():
        model_name = model_config["display_name"]
        count = sum(1 for r in all_rows if r["model"] == model_name)
        if count > 0:
            successes = sum(
                1 for r in all_rows if r["model"] == model_name and r["success"] == 1
            )
            logger.info(
                f"  {model_name}: {count} trials, {successes} successes ({100 * successes / count:.1f}%)"
            )
        else:
            logger.info(f"  {model_name}: 0 trials")

    logger.info("\n" + "=" * 60)
    logger.info("Summary by environment:")
    logger.info("=" * 60)
    env_names = sorted(set(r["environment"] for r in all_rows))
    for env in env_names:
        count = sum(1 for r in all_rows if r["environment"] == env)
        if count > 0:
            successes = sum(
                1 for r in all_rows if r["environment"] == env and r["success"] == 1
            )
            logger.info(
                f"  {env}: {count} trials, {successes} successes ({100 * successes / count:.1f}%)"
            )

    logger.info("\n" + "=" * 60)
    logger.info("Summary by scaffold:")
    logger.info("=" * 60)
    for scaffold in ["React", "Tool_calling"]:
        count = sum(1 for r in all_rows if r["scaffold"] == scaffold)
        if count > 0:
            successes = sum(
                1 for r in all_rows if r["scaffold"] == scaffold and r["success"] == 1
            )
            logger.info(
                f"  {scaffold}: {count} trials, {successes} successes ({100 * successes / count:.1f}%)"
            )

    logger.info("\n" + "=" * 60)
    logger.info("Summary by verbosity:")
    logger.info("=" * 60)
    for verbosity in VERBOSITIES:
        count = sum(1 for r in all_rows if r["verbosity"] == verbosity)
        if count > 0:
            successes = sum(
                1 for r in all_rows if r["verbosity"] == verbosity and r["success"] == 1
            )
            logger.info(
                f"  {verbosity}: {count} trials, {successes} successes ({100 * successes / count:.1f}%)"
            )


if __name__ == "__main__":
    main()
