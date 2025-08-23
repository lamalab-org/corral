"""
Parse the results from the results JSON file.
The script extract the relevant information (model, agent_type, env, verbosity_level, and if it is chained or single) from the JSON filename.
It avoids the bulk modulus files, since we did not consider them in our analysis.
It averages or sums the results for each environment with the same model, agent_type, and verbosity_level.
The MD environments are all joined under the "MD" key.
Pass@ks and Pass^ks are the averaged metrics.
total_tool_calls, successful_tool_calls, failed_tool_calls, completion_tokens, and total_tool_execution_duration are also included in the results by adding them from all the tasks.
The results are saved to processed_results.json.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

# Define the regex patterns for matching the required filename formats
pattern1 = r"^([\w-]+)-([\w-]+)-([\w-]+)-([\w-]+)_verbosity\.json$"  # Original pattern
pattern2 = r"^([\w-]+)-([\w-]+)-([\w-]+)-([\w-]+)-(?:task|subtask)\.json$"  # New pattern for task or subtask


def extract_json_files(path: str) -> list[str]:
    """Extract JSON files matching the specified patterns from directory."""
    # Walk through the directory recursively and use list comprehension
    return [
        str(Path(root) / file)
        for root, _dirs, files in os.walk(path)
        for file in files
        if re.match(pattern1, file) or re.match(pattern2, file)
    ]


def extract_filename_components(filename: str) -> dict[str, str] | None:
    """Extract model, agent_type, env, and verbosity_level from filename."""
    basename = Path(filename).name

    # Try pattern1: model-agent_type-env-verbosity_level_verbosity.json
    match1 = re.match(pattern1, basename)
    if match1:
        return {
            "model": match1.group(1),
            "agent_type": match1.group(2),
            "env": match1.group(3),
            "verbosity_level": match1.group(4),
        }

    # Try pattern2: model-agent_type-env-verbosity_level-task.json or model-agent_type-env-verbosity_level-subtask.json
    match2 = re.match(pattern2, basename)
    if match2:
        return {
            "model": match2.group(1),
            "agent_type": match2.group(2),
            "env": match2.group(3),
            "verbosity_level": match2.group(4),
        }

    return None


def is_chained(file_path: str, filename: str) -> bool:
    """
    Determine if the task is chained based on file path or filename.
    Returns True if 'chained' is in the file path or 'subtask' is in the filename.
    """
    file_path_lower = file_path.lower()
    filename_lower = filename.lower()

    return "chained" in file_path_lower or "subtask" in filename_lower


def extract_metrics_from_data(data: dict) -> dict[str, Any]:
    """Extract specific metrics from the JSON data."""
    metrics = {}

    try:
        # Extract the required metrics
        metrics["pass@1"] = data.get("metrics", {}).get("pass@1")
        metrics["pass@2"] = data.get("metrics", {}).get("pass@2")
        metrics["pass@3"] = data.get("metrics", {}).get("pass@3")
        metrics["pass@4"] = data.get("metrics", {}).get("pass@4")
        metrics["pass@5"] = data.get("metrics", {}).get("pass@5")
        metrics["pass^1"] = data.get("metrics", {}).get("pass^1")
        metrics["pass^2"] = data.get("metrics", {}).get("pass^2")
        metrics["pass^3"] = data.get("metrics", {}).get("pass^3")
        metrics["pass^4"] = data.get("metrics", {}).get("pass^4")
        metrics["pass^5"] = data.get("metrics", {}).get("pass^5")
        metrics["total_tool_calls"] = data.get("metrics", {}).get("total_tool_calls")
        metrics["successful_tool_calls"] = data.get("metrics", {}).get(
            "successful_tool_calls"
        )
        metrics["failed_tool_calls"] = data.get("metrics", {}).get("failed_tool_calls")

        # Extract completion tokens from nested structure
        token_usage = data.get("metrics", {}).get("total_token_usage", {})
        metrics["completion_tokens"] = token_usage.get("completion_tokens")

        metrics["total_tool_execution_duration"] = data.get("metrics", {}).get(
            "total_tool_execution_duration"
        )

    except (KeyError, TypeError) as e:
        logger.info(f"Error extracting metrics: {e}")
        return {}

    return metrics


def extract_data_from_json(file_path: str) -> dict | None:
    """Load and return JSON data from file."""
    try:
        with Path(file_path).open() as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return None
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None


def aggregate_results(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate results with the same model, agent_type, env, verbosity_level, and chained.

    Averages: pass@1, pass@2, pass@3, pass@4, pass@5, pass^1, pass^2, pass^3, pass^4, pass^5
    Sums: total_tool_calls, successful_tool_calls, failed_tool_calls, completion_tokens, total_tool_execution_duration
    """
    # Group results by the key fields
    groups = defaultdict(list)

    for result in data:
        # Create a key from the grouping fields
        key = (
            result.get("model"),
            result.get("agent_type"),
            result.get("env"),
            result.get("verbosity_level"),
            result.get("chained"),
        )
        groups[key].append(result)

    aggregated_results = []

    for key, group in groups.items():
        if len(group) == 1:
            # Single result, but still mark it as processed through aggregation
            result = group[0].copy()
            result["source_files"] = [result.get("file_path")]
            aggregated_results.append(result)
        else:
            # Multiple results, aggregate them
            logger.info(
                f"Aggregating {len(group)} results for: model={key[0]}, agent_type={key[1]}, env={key[2]}, verbosity_level={key[3]}, chained={key[4]}"
            )

            # Initialize aggregated result with the first item's metadata
            aggregated = {
                "model": key[0],
                "agent_type": key[1],
                "env": key[2],
                "verbosity_level": key[3],
                "chained": key[4],
                "file_path": f"[Aggregated from {len(group)} files]",
            }

            # Fields to average
            avg_fields = [
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
            # Fields to sum
            sum_fields = [
                "total_tool_calls",
                "successful_tool_calls",
                "failed_tool_calls",
                "completion_tokens",
                "total_tool_execution_duration",
            ]

            # Calculate averages
            for field in avg_fields:
                values = [
                    item.get(field) for item in group if item.get(field) is not None
                ]
                if values:
                    aggregated[field] = sum(values) / len(values)
                else:
                    aggregated[field] = None

            # Calculate sums
            for field in sum_fields:
                values = [
                    item.get(field) for item in group if item.get(field) is not None
                ]
                if values:
                    aggregated[field] = sum(values)
                else:
                    aggregated[field] = None

            # Add metadata about the aggregation
            aggregated["aggregated_count"] = len(group)
            aggregated["source_files"] = [item.get("file_path") for item in group]

            aggregated_results.append(aggregated)

    logger.info(
        f"\nAggregation complete: {len(data)} original results → {len(aggregated_results)} aggregated results"
    )

    return aggregated_results


def average_env_results(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Average results across surface_energy, quenching, and melting environments.
    Drops entries with env equal to bulk_modulus.

    Groups by: model, agent_type, verbosity_level, chained
    Averages: pass@1, pass@2, pass@3, pass@4, pass@5, pass^1, pass^2, pass^3, pass^4, pass^5
    Sums: total_tool_calls, successful_tool_calls, failed_tool_calls, completion_tokens, total_tool_execution_duration
    """
    # Target environments to include in averaging
    target_envs = {"surface_energy", "quenching", "melting"}

    # Filter out bulk_modulus and keep only target environments
    filtered_data = [result for result in data if result.get("env") in target_envs]

    logger.info(
        f"Filtered {len(data)} results → {len(filtered_data)} results (removed bulk_modulus entries)"
    )

    if not filtered_data:
        logger.info("No results to average after filtering")
        return []

    # Group results by model, agent_type, verbosity_level, chained (excluding env)
    groups = defaultdict(list)

    for result in filtered_data:
        key = (
            result.get("model"),
            result.get("agent_type"),
            result.get("verbosity_level"),
            result.get("chained"),
        )
        groups[key].append(result)

    averaged_results = []

    for key, group in groups.items():
        # Check which environments are represented in this group
        envs_in_group = {result.get("env") for result in group}

        logger.info(
            f"Averaging {len(group)} results for: model={key[0]}, agent_type={key[1]}, verbosity_level={key[2]}, chained={key[3]}"
        )
        logger.info(f"  Environments included: {', '.join(sorted(envs_in_group))}")

        # Initialize averaged result
        averaged = {
            "model": key[0],
            "agent_type": key[1],
            "env": "averaged_" + "_".join(sorted(envs_in_group)),  # Combined env name
            "verbosity_level": key[2],
            "chained": key[3],
            "file_path": f"[Averaged across {len(envs_in_group)} environments from {len(group)} files]",
        }

        # Fields to average
        avg_fields = [
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
        # Fields to sum
        sum_fields = [
            "total_tool_calls",
            "successful_tool_calls",
            "failed_tool_calls",
            "completion_tokens",
            "total_tool_execution_duration",
        ]

        # Calculate averages
        for field in avg_fields:
            values = [item.get(field) for item in group if item.get(field) is not None]
            if values:
                averaged[field] = sum(values) / len(values)
            else:
                averaged[field] = None

        # Calculate sums
        for field in sum_fields:
            values = [item.get(field) for item in group if item.get(field) is not None]
            if values:
                averaged[field] = sum(values)
            else:
                averaged[field] = None

        # Add metadata about the averaging
        averaged["env_averaged_count"] = len(group)
        averaged["environments_included"] = sorted(envs_in_group)
        averaged["source_files"] = [item.get("file_path") for item in group]

        averaged_results.append(averaged)

    logger.info(
        f"\nEnvironment averaging complete: {len(filtered_data)} filtered results → {len(averaged_results)} averaged results"
    )

    return averaged_results


def process_json_files(directory: str) -> list[dict[str, Any]]:
    """Process all JSON files and extract required data."""
    json_files = extract_json_files(directory)
    extracted_data = []

    logger.info(f"Found {len(json_files)} JSON files matching pattern")

    for file_path in json_files:
        logger.info(f"\nProcessing: {file_path}")

        # Extract filename components
        filename_components = extract_filename_components(file_path)
        if not filename_components:
            logger.warning(
                f"Could not extract components from filename: {Path(file_path).name}"
            )
            continue

        # Load JSON data
        data = extract_data_from_json(file_path)
        if not data:
            continue

        # Extract metrics
        metrics = extract_metrics_from_data(data)
        if not metrics:
            logger.warning("Could not extract metrics from data")
            continue

        # Determine if this is a chained task
        chained = is_chained(file_path, Path(file_path).name)

        # Combine all extracted information
        result = {
            **filename_components,  # model, agent_type, env, verbosity_level
            **metrics,  # all the metrics
            "chained": chained,  # boolean indicating if task is chained
            "file_path": file_path,  # for reference
        }

        extracted_data.append(result)

        # Print extracted data for this file
        logger.info("Extracted data:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")

    return extracted_data


def save_extracted_data(
    data: list[dict[str, Any]], output_file: str = "extracted_data.json"
):
    """Save extracted data to a JSON file."""
    try:
        with Path(output_file).open("w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"\nExtracted data saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")


def combine_all_results(extracted_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Combine all results: original, aggregated, and environment-averaged.
    Only drops bulk_modulus entries from the final output.
    Ensures that original files don't appear if they've been aggregated or environment-averaged.
    """
    # First, aggregate results that need aggregation
    aggregated_data = aggregate_results(extracted_data)

    # Get environment-averaged data from aggregated results
    env_averaged_data = average_env_results(aggregated_data)

    # Track ALL files that have been processed
    # Start by tracking all files that went through aggregation
    all_aggregated_files = set()
    for result in aggregated_data:
        source_files = result.get("source_files", [])
        for file_path in source_files:
            all_aggregated_files.add(file_path)

    # Track which files were used in environment averaging (highest priority)
    env_averaged_source_files = set()
    for result in env_averaged_data:
        source_files = result.get("source_files", [])
        for file_path in source_files:
            env_averaged_source_files.add(file_path)

    final_results = []

    # Add environment-averaged data (these have priority)
    final_results.extend(
        [
            result
            for result in env_averaged_data
            if result.get("env") != "bulk_modulus"
            and "bulk_modulus" not in result.get("env", "")
        ]
    )

    # Add aggregated data that wasn't environment-averaged
    for result in aggregated_data:
        source_files = result.get("source_files", [])
        # Skip if ANY of this data was used in environment averaging
        if (
            not any(
                file_path in env_averaged_source_files for file_path in source_files
            )
            and result.get("env") != "bulk_modulus"
        ):
            final_results.append(result)

    # Add original data that wasn't processed at all
    for result in extracted_data:
        file_path = result.get("file_path")
        # Skip if this data was processed through aggregation or environment averaging
        if (
            file_path not in all_aggregated_files
            and result.get("env") != "bulk_modulus"
        ):
            final_results.append(result)

    return final_results


def apply_env_mapping(
    data: list[dict[str, Any]], mappings: dict[str, str]
) -> list[dict[str, Any]]:
    """
    Apply environment name mappings to the data.

    Args:
        data: List of result dictionaries
        mappings: Dictionary mapping original env names to new names

    Returns:
        Updated data with mapped environment names
    """
    mapped_data = []

    for result in data:
        result_copy = result.copy()
        env = result_copy.get("env")

        if env and env in mappings:
            result_copy["env"] = mappings[env]
            logger.info(f"Mapped environment: {env} → {mappings[env]}")

        mapped_data.append(result_copy)

    return mapped_data


# Main execution
if __name__ == "__main__":
    # Process the JSON files
    extracted_data = process_json_files("../../reports")

    # Apply all transformations and combine results
    if extracted_data:
        logger.info(f"\n{'='*50}")
        logger.info("APPLYING ALL TRANSFORMATIONS")
        logger.info(f"{'='*50}")

        # Get intermediate results for reporting
        aggregated_data = aggregate_results(extracted_data)
        env_averaged_data = average_env_results(aggregated_data)

        # Combine all results (original + aggregated + env-averaged, minus bulk_modulus)
        final_data = combine_all_results(extracted_data)
    else:
        aggregated_data = []
        env_averaged_data = []
        final_data = []

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY:")
    logger.info(f"  Original files processed: {len(extracted_data)}")
    logger.info(f"  After aggregation: {len(aggregated_data)}")
    logger.info(f"  Environment-averaged entries: {len(env_averaged_data)}")
    logger.info(f"  Final combined data (excluding bulk_modulus): {len(final_data)}")
    logger.info(f"{'='*50}")

    # Save only the final combined data
    if final_data:
        # Apply environment name mapping
        mappings = {
            "averaged_melting_quenching_surface_energy": "MD",
            "spectra_chained_env": "Spectra",
            "spectra_env": "Spectra",
            "catalyst_env": "OpenCatalyst",
            "ml_env": "ML",
        }

        logger.info(f"\n{'='*50}")
        logger.info("APPLYING ENVIRONMENT MAPPINGS")
        logger.info(f"{'='*50}")

        final_data = apply_env_mapping(final_data, mappings)

        save_extracted_data(final_data, "processed_results.json")

        # Print breakdown of final data
        original_count = sum(
            1
            for item in final_data
            if not item.get("aggregated_count") and not item.get("env_averaged_count")
        )
        aggregated_count = sum(
            1
            for item in final_data
            if item.get("aggregated_count") and not item.get("env_averaged_count")
        )
        env_averaged_count = sum(
            1 for item in final_data if item.get("env_averaged_count")
        )

        logger.info("\nFinal data breakdown:")
        logger.info(f"  Original entries: {original_count}")
        logger.info(f"  Aggregated entries: {aggregated_count}")
        logger.info(f"  Environment-averaged entries: {env_averaged_count}")
        logger.info(f"  Total: {len(final_data)}")

        # Print a sample of the final data structure
        logger.info("\nSample final data structure:")
        sample = final_data[0]
        for key, value in sample.items():
            if key == "source_files" and isinstance(value, list) and len(value) > 2:
                logger.info(
                    f"  {key}: [{value[0]}, ..., {value[-1]}] ({len(value)} files)"
                )
            elif key == "environments_included":
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")
    else:
        logger.info("\nNo data to save after processing.")
