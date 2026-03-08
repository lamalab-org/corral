"""
Generate a report on subtask classification coverage.

This script:
1. Loads all subtasks from benchmark data
2. Attempts to classify them using subtask_category_tags.json
3. Creates a detailed report of coverage and missing classifications
"""

import json
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


def normalize_subtask_name(subtask, environment):
    """
    Normalize a subtask name to match against category tags.

    Args:
        subtask: Full subtask name from data
        environment: Environment name

    Returns:
        Normalized subtask name that can be matched against tags
    """
    # For most environments, we need to extract the generic pattern

    # AFM: keep as-is for level 1 subtasks
    if environment == "afm":
        return subtask

    # Catalyst: extract generic pattern (e.g., cu20_retrieve_structure -> retrieve_structure)
    if environment == "catalyst":
        parts = subtask.split("_", 1)
        if len(parts) > 1:
            return parts[1]
        return subtask

    # MD: extract generic pattern after material name
    if environment == "md":
        # e.g., aluminum_surface_1_subtask_bulk_energy_minimisation -> bulk_energy_minimisation
        # or silicon_melting_1_subtask_diffusivity -> diffusivity
        if "subtask_" in subtask:
            return subtask.split("subtask_")[-1]
        return subtask

    # ML: extract generic pattern
    if environment == "ml":
        # e.g., 0_batch_retrieve_oxide_polymorphs -> batch_retrieve_*
        # e.g., 0_evaluate_model_performance -> evaluate_model_performance
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            task_name = parts[1]
            # Replace specific material with wildcard
            if "batch_retrieve" in task_name:
                return "batch_retrieve_*"
            return task_name
        return subtask

    # Resistor: extract generic pattern
    if environment == "resistor":
        # e.g., task_0_subnet_1 -> subnet_1
        # e.g., task_0_final_assembly -> final_assembly
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0] == "task":
            parts2 = parts[1].split("_", 1)
            if len(parts2) > 1 and parts2[0].isdigit():
                return parts2[1]
        return subtask

    # Retrosynthesis: extract generic pattern
    if environment == "retro":
        # e.g., make_1_lvl1-apply_template-1 -> apply_template-1
        # e.g., make_1_lvl1-build_complete_route -> build_complete_route
        if "-" in subtask:
            return subtask.split("-", 1)[1]
        return subtask

    # Spectra: extract generic pattern
    if environment == "spectra":
        # e.g., 00_0000_orgsyn_000_0000_subtask_1 -> subtask_1
        if "_subtask_" in subtask:
            return "subtask_" + subtask.split("_subtask_")[-1]
        return subtask

    return subtask


def classify_subtask(subtask, environment, category_tags):
    """
    Classify a subtask based on category tags.

    Args:
        subtask: Full subtask name
        environment: Environment name
        category_tags: Category tags dictionary

    Returns:
        Category string or None if not found
    """
    # Get tags for this environment
    env_tags = category_tags.get(environment, {})
    if not env_tags:
        return None

    # For MD, tags are nested by task type
    if environment == "md":
        # Try to match task type first
        for task_type, task_tags in env_tags.items():
            if task_type in subtask.lower():
                # Normalize and match
                normalized = normalize_subtask_name(subtask, environment)
                if normalized in task_tags:
                    return task_tags[normalized]
        return None

    # Direct lookup
    if subtask in env_tags:
        return env_tags[subtask]

    # Try normalized name
    normalized = normalize_subtask_name(subtask, environment)
    if normalized in env_tags:
        return env_tags[normalized]

    return None


def main():
    """Generate classification report."""

    # Load data
    logger.info("Loading benchmark data...")
    reports_df = load_reports_data()

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
    logger.info("SUBTASK CLASSIFICATION REPORT")
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
                # Show first 3 examples
                for task in sorted(tasks)[:3]:
                    logger.info(f"        {task}")
                if len(tasks) > 3:
                    logger.info(f"        ... and {len(tasks) - 3} more")

        if unclassified[env]:
            logger.info("  Unclassified subtasks:")
            for task in sorted(unclassified[env])[:5]:
                logger.info(f"    - {task}")
            if len(unclassified[env]) > 5:
                logger.info(f"    ... and {len(unclassified[env]) - 5} more")

    # Save detailed report to JSON
    report_path = REPO_ROOT / "plots" / "panel_2" / "classification_report.json"
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
