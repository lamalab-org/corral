"""
Majority Voting Analysis for Cleaned Corral Annotations

This script performs majority voting on annotations from multiple annotators,
calculating agreement metrics and exporting results to JSON files.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class MajorityVotingAnalyzer:
    """Analyzes annotations using majority voting"""

    def __init__(self, parquet_path: str):
        """
        Initialize the analyzer

        Args:
            parquet_path: Path to the parquet file containing annotations
        """
        self.parquet_path = Path(parquet_path)
        self.df = None
        self.results = {
            "total_files": 0,
            "files_with_multiple_annotators": 0,
            "total_annotatable_nodes": 0,
            "nodes_with_majority": 0,
            "nodes_with_absolute_agreement": 0,
            "nodes_with_disagreement": 0,
            "nodes_with_partial_majority": 0,
            "nodes_with_any_agreement": 0,
            "percentage_majority": 0.0,
            "percentage_absolute_agreement": 0.0,
            "percentage_disagreement": 0.0,
            "percentage_partial_majority": 0.0,
            "percentage_any_agreement": 0.0,
        }
        self.majority_annotations = []
        self.disagreement_annotations = []
        self.partial_majority_annotations = []
        # Track marker selection stats per (env, level)
        # Structure: {(env, level): {marker: {"accepted": count, "discarded": count}}}
        self.marker_selection_stats = defaultdict(
            lambda: defaultdict(lambda: {"accepted": 0, "discarded": 0})
        )

    def load_data(self) -> pd.DataFrame:
        """Load the parquet file"""
        logger.info(f"Loading data from {self.parquet_path}...")
        self.df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(self.df)} entries")
        return self.df

    def _parse_json_if_needed(self, value: Any) -> Any:
        """Parse JSON string if needed"""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    def analyze(self) -> dict[str, Any]:
        """
        Perform majority voting analysis

        Returns:
            Dictionary containing analysis results
        """
        if self.df is None:
            self.load_data()

        # Group by fileId
        grouped = self.df.groupby("fileId")
        self.results["total_files"] = len(grouped)

        logger.info("Starting majority voting analysis...")

        for file_id, group in grouped:
            # Only process files with 2 or more annotators
            if len(group) < 2:
                logger.debug(f"Skipping {file_id} - only {len(group)} annotator(s)")
                continue

            self.results["files_with_multiple_annotators"] += 1
            logger.info(
                f"Processing {file_id} with {len(group)} annotators: {group['annotator'].tolist()}"
            )

            # Process this file
            self._process_file(file_id, group)

        # Calculate percentages
        if self.results["total_annotatable_nodes"] > 0:
            total = self.results["total_annotatable_nodes"]
            self.results["percentage_majority"] = (
                self.results["nodes_with_majority"] / total
            ) * 100
            self.results["percentage_absolute_agreement"] = (
                self.results["nodes_with_absolute_agreement"] / total
            ) * 100
            self.results["percentage_disagreement"] = (
                self.results["nodes_with_disagreement"] / total
            ) * 100
            self.results["percentage_partial_majority"] = (
                self.results["nodes_with_partial_majority"] / total
            ) * 100
            self.results["nodes_with_any_agreement"] = (
                self.results["nodes_with_majority"]
                + self.results["nodes_with_partial_majority"]
            )
            self.results["percentage_any_agreement"] = (
                self.results["nodes_with_any_agreement"] / total
            ) * 100

        logger.info("Analysis complete!")
        return self.results

    def _process_file(self, file_id: str, group: pd.DataFrame) -> None:
        """
        Process a single file with multiple annotators

        Args:
            file_id: The file identifier
            group: DataFrame containing all annotations for this file
        """
        # Extract environment and level from the first row (all rows should have the same values)
        environment = group.iloc[0]["environment"]
        level = group.iloc[0]["level"]

        # Parse fileNodes for each annotator
        annotators_data = []
        for _, row in group.iterrows():
            annotator = row["annotator"]
            file_nodes = self._parse_json_if_needed(row["fileNodes"])
            annotators_data.append({"annotator": annotator, "fileNodes": file_nodes})

        # Build a mapping of node_id -> list of (annotator, markers)
        node_annotations = defaultdict(list)

        for annotator_data in annotators_data:
            annotator = annotator_data["annotator"]
            file_nodes = annotator_data["fileNodes"]

            for node in file_nodes:
                node_id = node.get("id")
                annotatable = node.get("annotatable", False)
                markers = node.get("markers", [])

                if annotatable:
                    node_annotations[node_id].append(
                        {
                            "annotator": annotator,
                            "markers": markers,
                            "node": node,
                        }
                    )

        # Process each annotatable node
        for node_id, annotations in node_annotations.items():
            self.results["total_annotatable_nodes"] += 1
            self._vote_on_node(file_id, node_id, annotations, environment, level)
            self._vote_on_node_per_marker(
                file_id, node_id, annotations, environment, level
            )

    def _track_marker_selection(
        self,
        environment: str,
        level: int,
        accepted_markers: list[str],
        discarded_markers: list[str],
    ) -> None:
        """
        Track accepted and discarded markers per environment and level.

        Args:
            environment: The task environment (e.g., 'md', 'ml', 'catalyst')
            level: The task level (1, 2, 3, or 4)
            accepted_markers: List of markers that were accepted
            discarded_markers: List of markers that were discarded
        """
        key = f"{environment}_level_{level}"

        for marker in accepted_markers:
            self.marker_selection_stats[key][marker]["accepted"] += 1

        for marker in discarded_markers:
            self.marker_selection_stats[key][marker]["discarded"] += 1

    def _vote_on_node(
        self,
        file_id: str,
        node_id: str,
        annotations: list[dict[str, Any]],
        environment: str,
        level: int,
    ) -> None:
        """
        Perform majority voting on a single node

        Args:
            file_id: The file identifier
            node_id: The node identifier
            annotations: List of annotations from different annotators
            environment: The task environment
            level: The task level
        """
        # Collect all markers from all annotators
        all_markers = []
        all_individual_markers = set()  # Track all individual markers seen
        for ann in annotations:
            markers = ann["markers"]
            # Each marker set is treated as a single vote
            all_markers.append(tuple(sorted(markers)) if markers else ())
            all_individual_markers.update(markers)

        # Count occurrences of each marker set
        marker_counts = Counter(all_markers)
        total_annotators = len(annotations)

        # Check for absolute agreement (all annotators agree)
        if len(marker_counts) == 1:
            self.results["nodes_with_absolute_agreement"] += 1
            self.results["nodes_with_majority"] += 1

            # Store the agreed annotation
            agreed_markers = (
                list(next(iter(marker_counts.keys()))) if marker_counts else []
            )

            # Track marker selection: all agreed markers are accepted, none discarded
            self._track_marker_selection(
                environment,
                level,
                accepted_markers=agreed_markers,
                discarded_markers=[],
            )

            self.majority_annotations.append(
                {
                    "fileId": file_id,
                    "nodeId": node_id,
                    "markers": agreed_markers,
                    "vote_type": "absolute_agreement",
                    "annotators": [ann["annotator"] for ann in annotations],
                    "vote_count": total_annotators,
                    "total_annotators": total_annotators,
                }
            )
            return

        # Find the majority
        most_common_markers, max_count = marker_counts.most_common(1)[0]

        # Check if there's a clear majority (more than half)
        if max_count > total_annotators / 2:
            self.results["nodes_with_majority"] += 1

            accepted_markers = list(most_common_markers)
            # Discarded markers are those that appeared in votes but not in the winning set
            discarded_markers = list(all_individual_markers - set(accepted_markers))

            # Track marker selection
            self._track_marker_selection(
                environment,
                level,
                accepted_markers=accepted_markers,
                discarded_markers=discarded_markers,
            )

            # Store the majority annotation
            self.majority_annotations.append(
                {
                    "fileId": file_id,
                    "nodeId": node_id,
                    "markers": accepted_markers,
                    "vote_type": "majority",
                    "annotators": [ann["annotator"] for ann in annotations],
                    "vote_count": max_count,
                    "total_annotators": total_annotators,
                    "all_votes": {str(list(k)): v for k, v in marker_counts.items()},
                }
            )
        else:
            # Disagreement - no clear majority
            self.results["nodes_with_disagreement"] += 1

            # For disagreements, all markers are considered discarded
            # (no consensus, so none are selected)
            self._track_marker_selection(
                environment,
                level,
                accepted_markers=[],
                discarded_markers=list(all_individual_markers),
            )

            # Store the disagreement
            self.disagreement_annotations.append(
                {
                    "fileId": file_id,
                    "nodeId": node_id,
                    "vote_type": "disagreement",
                    "annotators": [ann["annotator"] for ann in annotations],
                    "total_annotators": total_annotators,
                    "all_votes": {str(list(k)): v for k, v in marker_counts.items()},
                }
            )

    def _vote_on_node_per_marker(
        self,
        file_id: str,
        node_id: str,
        annotations: list[dict[str, Any]],
        environment: str,
        level: int,
    ) -> None:
        """
        Perform per-marker majority voting on a single node.
        This method finds markers that have majority agreement even if not all markers agree.

        Args:
            file_id: The file identifier
            node_id: The node identifier
            annotations: List of annotations from different annotators
            environment: The task environment
            level: The task level
        """
        # Collect all individual markers from all annotators
        marker_votes = defaultdict(list)

        for ann in annotations:
            annotator = ann["annotator"]
            markers = ann["markers"]
            for marker in markers:
                marker_votes[marker].append(annotator)

        total_annotators = len(annotations)
        majority_threshold = total_annotators / 2

        # Find markers that have majority support
        majority_markers = []
        marker_vote_details = {}

        for marker, voters in marker_votes.items():
            vote_count = len(voters)
            marker_vote_details[marker] = {"count": vote_count, "annotators": voters}
            if vote_count > majority_threshold:
                majority_markers.append(marker)

        # Check if this is already a full majority (all markers agree)
        # by checking if we already added it to majority_annotations
        is_already_full_majority = any(
            ann["fileId"] == file_id and ann["nodeId"] == node_id
            for ann in self.majority_annotations
        )

        # Only process if this wasn't already counted as absolute agreement or full majority
        if not is_already_full_majority:
            if majority_markers:
                # Case 1: Some markers have majority support (partial majority)
                self.results["nodes_with_partial_majority"] += 1

                # Track marker selection for partial majority
                # Accepted: markers with majority support
                # Discarded: markers that appeared but didn't get majority
                all_markers_seen = set(marker_votes.keys())
                discarded_markers = list(all_markers_seen - set(majority_markers))

                self._track_marker_selection(
                    environment,
                    level,
                    accepted_markers=majority_markers,
                    discarded_markers=discarded_markers,
                )

                # Store the partial majority annotation
                self.partial_majority_annotations.append(
                    {
                        "fileId": file_id,
                        "nodeId": node_id,
                        "markers": sorted(majority_markers),
                        "vote_type": "partial_majority",
                        "annotators": [ann["annotator"] for ann in annotations],
                        "total_annotators": total_annotators,
                        "marker_vote_details": marker_vote_details,
                    }
                )
            else:
                # Case 2: No markers have majority support - all markers discarded
                # This should be treated as disagreement (no agreement on any marker)
                # Only count if there were actually some markers to begin with
                if marker_votes:
                    # Don't increment disagreement counter again if already counted in _vote_on_node
                    # Just track the marker selection: all markers seen are discarded
                    all_markers_seen = set(marker_votes.keys())

                    self._track_marker_selection(
                        environment,
                        level,
                        accepted_markers=[],
                        discarded_markers=list(all_markers_seen),
                    )

    def print_summary(self) -> None:
        """Print a summary of the analysis results"""
        logger.info("\n" + "=" * 70)
        logger.info("MAJORITY VOTING ANALYSIS SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nTotal files: {self.results['total_files']}")
        logger.info(
            f"Files with multiple annotators: {self.results['files_with_multiple_annotators']}"
        )
        logger.info(
            f"\nTotal annotatable nodes (from multi-annotator files): {self.results['total_annotatable_nodes']}"
        )

        logger.info("\n" + "-" * 70)
        logger.info("AGREEMENT METRICS")
        logger.info("-" * 70)

        logger.info(
            f"\nNodes with majority agreement: {self.results['nodes_with_majority']} "
            f"({self.results['percentage_majority']:.2f}%)"
        )
        logger.info(
            f"  ├─ Absolute agreement (all agreed): {self.results['nodes_with_absolute_agreement']} "
            f"({self.results['percentage_absolute_agreement']:.2f}%)"
        )
        logger.info(
            f"  └─ Majority (not absolute): {self.results['nodes_with_majority'] - self.results['nodes_with_absolute_agreement']}"
        )

        logger.info(
            f"\nNodes with partial majority (per-marker): {self.results['nodes_with_partial_majority']} "
            f"({self.results['percentage_partial_majority']:.2f}%)"
        )

        logger.info(
            f"\nNodes with any agreement (full + partial): {self.results['nodes_with_any_agreement']} "
            f"({self.results['percentage_any_agreement']:.2f}%)"
        )

        logger.info(
            f"\nNodes with disagreement: {self.results['nodes_with_disagreement']} "
            f"({self.results['percentage_disagreement']:.2f}%)"
        )

        logger.info(
            f"\nNodes with any agreement (full + partial): {self.results['nodes_with_any_agreement']} "
            f"({self.results['percentage_any_agreement']:.2f}%)"
        )

        logger.info(
            f"\nNodes with disagreement: {self.results['nodes_with_disagreement']} "
            f"({self.results['percentage_disagreement']:.2f}%)"
        )

        logger.info("\n" + "=" * 70)
        logger.info(
            f"\nNodes with disagreement: {self.results['nodes_with_disagreement']} "
            f"({self.results['percentage_disagreement']:.2f}%)"
        )

        logger.info("\n" + "=" * 70)

    def save_results(
        self, output_dir: str | None = None
    ) -> tuple[Path, Path, Path, Path, Path]:
        """
        Save analysis results to JSON files

        Args:
            output_dir: Directory to save results (default: same as parquet file)

        Returns:
            Tuple of paths to the saved files (summary, majority, disagreement, partial_majority, marker_selection)
        """
        if output_dir is None:
            output_dir = self.parquet_path.parent

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary results
        summary_path = output_dir / "majority_voting_summary.json"
        with summary_path.open("w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        # Save majority annotations
        majority_path = output_dir / "majority_voting_annotations.json"
        with majority_path.open("w") as f:
            json.dump(self.majority_annotations, f, indent=2)
        logger.info(
            f"Saved {len(self.majority_annotations)} majority annotations to {majority_path}"
        )

        # Save disagreement annotations
        disagreement_path = output_dir / "disagreement_annotations.json"
        with disagreement_path.open("w") as f:
            json.dump(self.disagreement_annotations, f, indent=2)
        logger.info(
            f"Saved {len(self.disagreement_annotations)} disagreement annotations to {disagreement_path}"
        )

        # Save partial majority annotations
        partial_majority_path = output_dir / "partial_majority_annotations.json"
        with partial_majority_path.open("w") as f:
            json.dump(self.partial_majority_annotations, f, indent=2)
        logger.info(
            f"Saved {len(self.partial_majority_annotations)} partial majority annotations to {partial_majority_path}"
        )

        # Save marker selection stats per environment and level
        # Convert defaultdict to regular dict for JSON serialization
        marker_selection_data = {
            env_level: dict(markers)
            for env_level, markers in self.marker_selection_stats.items()
        }

        # Compute overall stats per environment (across all levels)
        env_overall_stats = defaultdict(
            lambda: defaultdict(lambda: {"accepted": 0, "discarded": 0})
        )
        for env_level, markers in self.marker_selection_stats.items():
            # Extract environment from the key (format: "env_level_X")
            env = "_".join(env_level.split("_level_")[0].split("_"))
            for marker, stats in markers.items():
                env_overall_stats[f"{env}_overall"][marker]["accepted"] += stats[
                    "accepted"
                ]
                env_overall_stats[f"{env}_overall"][marker]["discarded"] += stats[
                    "discarded"
                ]

        # Compute fully overall stats (across all environments and levels)
        fully_overall_stats = defaultdict(lambda: {"accepted": 0, "discarded": 0})
        for markers in self.marker_selection_stats.values():
            for marker, stats in markers.items():
                fully_overall_stats[marker]["accepted"] += stats["accepted"]
                fully_overall_stats[marker]["discarded"] += stats["discarded"]

        # Combine all stats
        combined_marker_selection_data = {
            **marker_selection_data,
            **{
                env_overall: dict(markers)
                for env_overall, markers in env_overall_stats.items()
            },
            "fully_overall": dict(fully_overall_stats),
        }

        marker_selection_path = output_dir / "marker_selection_stats.json"
        with marker_selection_path.open("w") as f:
            json.dump(combined_marker_selection_data, f, indent=2, sort_keys=True)
        logger.info(
            f"Saved marker selection stats for {len(marker_selection_data)} env/level combinations, "
            f"{len(env_overall_stats)} environment overalls, and 1 fully overall to {marker_selection_path}"
        )

        return (
            summary_path,
            majority_path,
            disagreement_path,
            partial_majority_path,
            marker_selection_path,
        )


def main():
    """Main execution function"""
    # Set up paths
    parquet_path = Path(__file__).parent / "data" / "cleaned_corral_annotations.parquet"

    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return

    # Create analyzer
    analyzer = MajorityVotingAnalyzer(str(parquet_path))

    # Run analysis
    analyzer.analyze()

    # Print summary
    analyzer.print_summary()

    # Save results
    output_dir = parquet_path.parent
    (
        summary_path,
        majority_path,
        disagreement_path,
        partial_majority_path,
        marker_selection_path,
    ) = analyzer.save_results(str(output_dir))

    logger.info("\nResults saved to:")
    logger.info(f"  - Summary: {summary_path}")
    logger.info(f"  - Majority annotations: {majority_path}")
    logger.info(f"  - Disagreement annotations: {disagreement_path}")
    logger.info(f"  - Partial majority annotations: {partial_majority_path}")
    logger.info(f"  - Marker selection stats: {marker_selection_path}")


if __name__ == "__main__":
    main()
