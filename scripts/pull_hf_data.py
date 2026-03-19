from pathlib import Path

import fire
from datasets import load_dataset
from loguru import logger
from report_constants import (
    AGENTS,
    ALLOWED_CATEGORIES,
    ALLOWED_MODELS,
    LEVEL_PREFIX,
    VERBOSITY,
)


def normalize_agent(agent: str) -> str:
    agent_key = agent.lower().replace("_", "").strip()
    if agent_key not in AGENTS:
        raise ValueError(f"Invalid agent '{agent}'. Allowed: {set(AGENTS.keys())}")
    return AGENTS[agent_key]


def normalize_level(level: str) -> str:
    if not level.startswith(LEVEL_PREFIX):
        level = f"{LEVEL_PREFIX}{level}"
    return level


def validate(model, category, verbosity):
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Invalid model '{model}'. Allowed: {ALLOWED_MODELS}")

    if category not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Allowed: {ALLOWED_CATEGORIES}"
        )

    if verbosity not in VERBOSITY:
        raise ValueError(f"Invalid verbosity '{verbosity}'. Allowed: {VERBOSITY}")


def download(
    model: str,
    env: str,
    level: str,
    category: str,
    agent: str,
    verbosity: str,
    report_type: str,
    output_dir: str,
):
    """
    Download a subset from the HF dataset repo and save it locally.
    """

    validate(model, category, verbosity)

    level = normalize_level(level)
    agent = normalize_agent(agent)

    if report_type not in {"traces", "overall_reports", "task_reports"}:
        raise ValueError("type must be one of: traces, overall_reports, task_reports")

    subset_path = f"{model}-{env}-{level}-{category}-{agent}-{verbosity}-{report_type}"

    if report_type == "traces":
        dataset_repo = "jablonkagroup/corral-traces"
    else:
        dataset_repo = "jablonkagroup/corral-reports"

    logger.info(f"Dataset repo : {dataset_repo}")
    logger.info(f"Subset path  : {subset_path}")

    output_path = Path(output_dir) / subset_path
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_repo, subset_path)

    dataset.save_to_disk(str(output_path))

    logger.success(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    fire.Fire(download)
