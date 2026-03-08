"""
Generate a report on subtask classification coverage with improved pattern matching.

This script:
1. Loads all subtasks from benchmark data
2. Attempts to classify them using subtask_category_tags.json with pattern matching
3. Creates a detailed report of coverage and missing classifications
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
from plot_utils import load_reports_data

# Add analysis to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))


def load_category_tags():
    """Load the subtask category tags from JSON file."""
    tags_path = REPO_ROOT / "analysis" / "subtask_category_tags.json"
    with tags_path.open() as f:
        return json.load(f)


def get_env_key_mapping():
    """Map environment names in data to keys in category tags."""
    return {
        "spectra": "sptectra",
        "retro": "retrosynthesis",
        "afm": "afm",
        "catalyst": "catalyst",
        "md": "md",
        "ml": "ml",
        "resistor": "resistor",
    }


def classify_subtask(subtask, environment, category_tags):
    """
    Classify a subtask based on category tags with pattern matching.

    Args:
        subtask: Full subtask name
        environment: Environment name from data
        category_tags: Category tags dictionary

    Returns:
        Category string or None if not found
    """
    # Map environment name to tag key
    env_mapping = get_env_key_mapping()
    tag_env_key = env_mapping.get(environment, environment)

    # Get tags for this environment
    env_tags = category_tags.get(tag_env_key, {})
    if not env_tags:
        return None

    # AFM: pattern matching for level-based tasks
    if environment == "afm":
        # Match level 1-4 subtasks
        if "subtask_level_" in subtask:
            # Extract base name
            base_name = subtask.split("_level_")[0]
            if base_name + "_level_1" in env_tags:
                return env_tags[base_name + "_level_1"]

        # Match experiment level tasks
        if subtask.startswith("afm_experiment_level_"):
            return "experiment_execution"

        # Match higher level tasks
        if "relationship_level" in subtask or "roughness_subtask_level" in subtask:
            return "validation"

        # Direct match
        return env_tags.get(subtask)

    # Catalyst: extract generic pattern
    if environment == "catalyst":
        # Workflow tasks are validation
        if subtask.endswith("_workflow"):
            return "validation"

        # Extract pattern (e.g., cu20_retrieve_structure -> retrieve_structure)
        parts = subtask.split("_", 1)
        if len(parts) > 1:
            pattern = parts[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    # MD: nested structure by task type
    if environment == "md":
        # Extract task type and subtask name
        # e.g., silicon_melting_1_subtask_diffusivity or aluminum_surface_1_subtask_bulk_energy_minimisation
        for task_type in ["melting", "quenching", "surface_energy", "surface"]:
            if task_type in subtask:
                task_tags = env_tags.get(task_type, {})
                if "subtask_" in subtask:
                    subtask_name = subtask.split("subtask_")[-1]
                    # Map variations
                    if subtask_name == "diffusion_coefficient":
                        subtask_name = "diffusivity"
                    if subtask_name == "tg_calculation":
                        subtask_name = "tg_detection"
                    elif subtask_name == "equilibration":
                        subtask_name = "structure_retrieval"
                    if subtask_name in task_tags:
                        return task_tags[subtask_name]

        # Top-level task names (silicon_melting_1, etc.) are validation
        if re.match(r"^\w+_(melting|quenching|surface_energy)_\d+$", subtask):
            return "validation"

        return None

    # ML: pattern matching
    if environment == "ml":
        # Workflow tasks are validation
        if subtask.startswith("ml_"):
            return "validation"

        # Extract generic pattern
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            task_name = parts[1]
            # Match batch_retrieve
            if (
                task_name.startswith("batch_retrieve_")
                and "batch_retrieve_*" in env_tags
            ):
                return env_tags["batch_retrieve_*"]
            # Direct match
            if task_name in env_tags:
                return env_tags[task_name]
        return None

    # Resistor: pattern matching
    if environment == "resistor":
        # Top-level tasks are validation
        if re.match(r"^task_\d+$", subtask):
            return "validation"

        # Extract pattern (e.g., task_0_subnet_1 -> subnet_1)
        parts = subtask.split("_", 2)
        if len(parts) >= 3 and parts[0] == "task" and parts[1].isdigit():
            pattern = "_".join(parts[2:])
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    # Retrosynthesis: pattern matching
    if environment == "retro":
        # Top-level tasks are validation
        if re.match(r"^make_\d+_lvl\d+$", subtask):
            return "validation"

        # Extract pattern (e.g., make_1_lvl1-apply_template-1 -> apply_template-1)
        if "-" in subtask:
            pattern = subtask.split("-", 1)[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    # Spectra: pattern matching
    if environment == "spectra":
        # Top-level tasks are validation
        if "_subtask_" not in subtask:
            return "validation"

        # Extract subtask number (e.g., ..._subtask_1 -> subtask_1)
        if "_subtask_" in subtask:
            subtask_num = "subtask_" + subtask.split("_subtask_")[-1]
            if subtask_num in env_tags:
                return env_tags[subtask_num]
        return None

    return None


def main():
    """Generate classification report."""

    # Load data
    logger.info("Loading benchmark data...")
    reports_df = load_reports_data()

    # IMPORTANT: Filter to SUBTASKS ONLY (category='subtask')
    reports_df = reports_df[reports_df["category"] == "subtask"].copy()
    logger.info(f"Filtered to {len(reports_df)} rows with category='subtask'")

    logger.info("Loading category tags...")
    category_tags = load_category_tags()

    # Extract all subtasks
    all_subtasks = {}  # {environment: set of subtasks}
    for _, row in reports_df.iterrows():
        env = row["environment"]
        task_results = row["Task Results"]
        if isinstance(task_results, dict):
            if env not in all_subtasks:
                all_subtasks[env] = set()
            all_subtasks[env].update(task_results.keys())

    # Classify subtasks
    classified = defaultdict(lambda: defaultdict(list))
    unclassified = defaultdict(list)

    for env, subtasks in sorted(all_subtasks.items()):
        for subtask in sorted(subtasks):
            category = classify_subtask(subtask, env, category_tags)
            if category:
                classified[env][category].append(subtask)
            else:
                unclassified[env].append(subtask)

    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("SUBTASK CLASSIFICATION REPORT (IMPROVED)")
    logger.info("=" * 80)

    # Summary statistics
    total_subtasks = sum(len(tasks) for tasks in all_subtasks.values())
    total_classified = sum(
        len(tasks) for env in classified.values() for tasks in env.values()
    )
    total_unclassified = sum(len(tasks) for tasks in unclassified.values())

    logger.info("\nSUMMARY:")
    logger.info(f"  Total unique subtasks: {total_subtasks}")
    logger.info(
        f"  Classified: {total_classified} ({100 * total_classified / total_subtasks:.1f}%)"
    )
    logger.info(
        f"  Unclassified: {total_unclassified} ({100 * total_unclassified / total_subtasks:.1f}%)"
    )

    # Category distribution
    category_counts = defaultdict(int)
    for env in classified.values():
        for category, tasks in env.items():
            category_counts[category] += len(tasks)

    logger.info("\nCATEGORY DISTRIBUTION:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(
            f"  {category}: {count} subtasks ({100 * count / total_classified:.1f}%)"
        )

    # Per-environment breakdown
    logger.info("\n\nPER-ENVIRONMENT BREAKDOWN:")
    logger.info("=" * 80)

    for env in sorted(all_subtasks.keys()):
        total = len(all_subtasks[env])
        classified_count = sum(len(tasks) for tasks in classified[env].values())
        unclassified_count = len(unclassified[env])

        logger.info(f"\n{env.upper()}: {total} total subtasks")
        logger.info(
            f"  Classified: {classified_count}/{total} ({100 * classified_count / total:.1f}%)"
        )
        logger.info(
            f"  Unclassified: {unclassified_count}/{total} ({100 * unclassified_count / total:.1f}%)"
        )

        if classified[env]:
            logger.info("  Categories:")
            for category, tasks in sorted(classified[env].items()):
                logger.info(f"    - {category}: {len(tasks)} subtasks")

        if unclassified[env] and len(unclassified[env]) <= 10:
            logger.info("  Unclassified subtasks:")
            for task in sorted(unclassified[env]):
                logger.info(f"    - {task}")

    # Save detailed report to JSON
    report_path = REPO_ROOT / "plots" / "panel_2" / "classification_report_v2.json"
    report_data = {
        "summary": {
            "total_subtasks": total_subtasks,
            "classified": total_classified,
            "unclassified": total_unclassified,
            "classification_rate": total_classified / total_subtasks,
        },
        "category_counts": dict(category_counts),
        "by_environment": {
            env: {
                "total": len(all_subtasks[env]),
                "classified": dict(classified[env].items()),
                "unclassified": unclassified[env],
            }
            for env in all_subtasks
        },
    }

    with report_path.open("w") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"\n\nDetailed report saved to: {report_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
