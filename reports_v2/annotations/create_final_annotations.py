"""
Create Final Annotations with Majority Voting

This script takes the cleaned annotations parquet file, groups entries by fileId,
discards files with only one annotator, applies majority voting on markers for
annotatable nodes, and saves the final consolidated annotations.
"""

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def solve_majority(
    file_id: str,
    node_id: str,
    marker_votes: dict[str, list[str]],
    total_annotators: int,
) -> list[str]:
    """
    Placeholder function to resolve cases where no majority is found for markers.

    This function is called when markers don't have a clear majority agreement.
    Override this function to implement custom resolution logic.

    Args:
        file_id: The file identifier
        node_id: The node identifier
        marker_votes: Dictionary mapping markers to list of annotators who voted for them
        total_annotators: Total number of annotators for this file

    Returns:
        List of markers to use (empty list by default)
    """
    logger.warning(
        f"No majority found for file '{file_id}', node '{node_id}'. "
        f"Marker votes: {marker_votes}. Total annotators: {total_annotators}"
    )
    # Default behavior: return empty list (no markers)
    # This can be customized to implement specific resolution strategies
    return []


def parse_json_if_needed(value: Any) -> Any:
    """Parse JSON string if needed"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def apply_majority_voting_per_marker(
    annotations: list[dict[str, Any]],
    file_id: str,
    node_id: str,
) -> tuple[list[str], bool]:
    """
    Apply per-marker majority voting for a node.

    For each marker present across annotators, check if it has majority support
    (more than half of annotators, or for 2 annotators, both must agree).

    Args:
        annotations: List of annotations from different annotators for this node
        file_id: The file identifier (for logging)
        node_id: The node identifier (for logging)

    Returns:
        Tuple of (list of agreed markers, boolean indicating if solve_majority was called)
    """
    total_annotators = len(annotations)

    # For 2 annotators, both must agree (threshold = 1, so > 1 means both agree)
    # For 3+ annotators, majority means > half
    majority_threshold = total_annotators / 2

    # Collect all markers from all annotators and count votes
    marker_votes = defaultdict(list)
    for ann in annotations:
        annotator = ann["annotator"]
        markers = ann["markers"] if ann["markers"] else []
        for marker in markers:
            marker_votes[marker].append(annotator)

    # Find markers that have majority support
    majority_markers = []
    non_majority_markers = {}

    for marker, voters in marker_votes.items():
        vote_count = len(voters)
        if total_annotators == 2:
            # For 2 annotators, both must agree (present in both)
            if vote_count == 2:
                majority_markers.append(marker)
            else:
                non_majority_markers[marker] = voters
        else:
            # For 3+ annotators, majority means more than half
            if vote_count > majority_threshold:
                majority_markers.append(marker)
            else:
                non_majority_markers[marker] = voters

    # Check if we need to call solve_majority
    # This happens when there are markers that were voted but no majority was reached
    needs_resolution = len(non_majority_markers) > 0

    if needs_resolution:
        # Call solve_majority for non-resolved markers
        resolved_markers = solve_majority(
            file_id, node_id, dict(marker_votes), total_annotators
        )
        # Combine majority markers with any resolved markers
        all_markers = list(set(majority_markers + resolved_markers))
    else:
        all_markers = majority_markers

    return sorted(all_markers), needs_resolution


def get_final_annotations(parquet_path: str, output_path: str) -> pd.DataFrame:
    """
    Process annotations and create final consolidated annotations with majority voting.

    Args:
        parquet_path: Path to the input parquet file with cleaned annotations
        output_path: Path to save the final annotations parquet file

    Returns:
        DataFrame with final consolidated annotations
    """
    logger.info(f"Loading data from {parquet_path}...")
    df_base = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df_base)} entries")

    # Group by fileId
    grouped = df_base.groupby("fileId")
    logger.info(f"Found {len(grouped)} unique files")

    final_annotations = []
    stats = {
        "total_files": len(grouped),
        "files_with_single_annotator": 0,
        "files_processed": 0,
        "nodes_processed": 0,
        "nodes_with_solve_majority_call": 0,
    }

    for file_id, group in grouped:
        # Only process files with 2 or more annotators
        if len(group) < 2:
            logger.debug(f"Skipping {file_id} - only {len(group)} annotator(s)")
            stats["files_with_single_annotator"] += 1
            continue

        stats["files_processed"] += 1
        annotators = group["annotator"].tolist()
        logger.info(f"Processing {file_id} with {len(group)} annotators: {annotators}")

        # Get the first row as the base for non-node fields
        # All annotators should have the same values for these fields
        base_row = group.iloc[0].to_dict()

        # Parse fileNodes for each annotator
        annotators_data = []
        for _, row in group.iterrows():
            annotator = row["annotator"]
            file_nodes = parse_json_if_needed(row["fileNodes"])
            annotators_data.append({"annotator": annotator, "fileNodes": file_nodes})

        # Build a mapping of node_id -> list of (annotator, markers, node_data)
        node_annotations = defaultdict(list)

        # Also keep track of node order and non-annotatable nodes from first annotator
        first_annotator_nodes = annotators_data[0]["fileNodes"]
        node_order = [node.get("id") for node in first_annotator_nodes]

        for annotator_data in annotators_data:
            annotator = annotator_data["annotator"]
            file_nodes = annotator_data["fileNodes"]

            for node in file_nodes:
                node_id = node.get("id")
                annotatable = node.get("annotatable", False)
                markers = node.get("markers", [])

                node_annotations[node_id].append(
                    {
                        "annotator": annotator,
                        "markers": markers,
                        "node": node,
                        "annotatable": annotatable,
                    }
                )

        # Process each node and apply majority voting for annotatable nodes
        final_file_nodes = []

        for node_id in node_order:
            if node_id not in node_annotations:
                continue

            annotations_for_node = node_annotations[node_id]

            # Get base node structure (from first annotator)
            base_node = deepcopy(annotations_for_node[0]["node"])
            annotatable = annotations_for_node[0]["annotatable"]

            stats["nodes_processed"] += 1

            if annotatable:
                # Apply majority voting for markers
                final_markers, called_solve_majority = apply_majority_voting_per_marker(
                    annotations_for_node,
                    file_id,
                    node_id,
                )

                if called_solve_majority:
                    stats["nodes_with_solve_majority_call"] += 1

                # Update the node with the final markers
                base_node["markers"] = final_markers

                # Remove notes field if present (it's annotator-specific)
                if "notes" in base_node:
                    del base_node["notes"]

            final_file_nodes.append(base_node)

        # Create final annotation entry (without annotator field)
        final_entry = {
            key: value
            for key, value in base_row.items()
            if key not in ["annotator", "fileNodes"]
        }
        final_entry["fileNodes"] = final_file_nodes

        final_annotations.append(final_entry)

    # Create DataFrame from final annotations
    final_df = pd.DataFrame(final_annotations)

    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    logger.info(f"Saved final annotations to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL ANNOTATIONS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files in input: {stats['total_files']}")
    logger.info(
        f"Files with single annotator (discarded): {stats['files_with_single_annotator']}"
    )
    logger.info(f"Files processed (2+ annotators): {stats['files_processed']}")
    logger.info(f"Total nodes processed: {stats['nodes_processed']}")
    logger.info(
        f"Nodes where solve_majority was called: {stats['nodes_with_solve_majority_call']}"
    )
    logger.info("=" * 70)

    return final_df


def main():
    """Main execution function"""
    # Set up paths
    parquet_path = Path(__file__).parent / "data" / "cleaned_corral_annotations.parquet"
    output_path = Path(__file__).parent / "data" / "final_annotations.parquet"

    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return

    # Create final annotations
    final_df = get_final_annotations(str(parquet_path), str(output_path))

    logger.info(f"\nFinal annotations saved to: {output_path}")
    logger.info(f"Total entries in final file: {len(final_df)}")


if __name__ == "__main__":
    main()
