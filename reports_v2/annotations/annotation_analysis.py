"""
Annotation Analysis Script

This script analyzes inter-annotator agreement and label usage patterns
from the cleaned annotation data. It calculates various agreement metrics,
co-occurrence patterns, and provides recommendations for label consolidation.

Usage:
    python annotation_analysis.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

# All possible labels
ALL_LABELS = [
    "validation_attempt",
    "backtrack_trigger",
    "planning_statement",
    "reasoning_statement",
    "correct_submission",
    "neutral",
    "iteration_limit",
    "missing_validation",
    "unnecessary_tool_use",
    "non_sense",
    "loop_instance",
    "hallucination",
    "wrong_planning",
    "wrong_reasoning",
    "syntax_error",
    "early_final_answer",
    "give_up",
    "inefficient_tool_call",
    "misunderstood_tool",
]

# Label categories
POSITIVE_LABELS = [
    "validation_attempt",
    "backtrack_trigger",
    "planning_statement",
    "reasoning_statement",
    "correct_submission",
]

NEUTRAL_LABELS = ["neutral", "iteration_limit"]

NEGATIVE_LABELS = [
    "missing_validation",
    "unnecessary_tool_use",
    "non_sense",
    "loop_instance",
    "hallucination",
    "wrong_planning",
    "wrong_reasoning",
    "syntax_error",
    "early_final_answer",
    "give_up",
    "inefficient_tool_call",
    "misunderstood_tool",
]


def load_data(parquet_path: str) -> pd.DataFrame:
    """Load and prepare the annotation data."""
    return pd.read_parquet(parquet_path)


def parse_file_nodes(file_nodes: Any) -> list[dict]:
    """Parse fileNodes from JSON string or return as-is if already a list."""
    if isinstance(file_nodes, str):
        return json.loads(file_nodes)
    return file_nodes


def get_annotatable_nodes(nodes: list[dict]) -> list[dict]:
    """Filter to only annotatable nodes (those with markers field)."""
    return [n for n in nodes if n.get("annotatable", False)]


def filter_multi_annotator_files(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only include files with multiple annotators."""
    file_annotator_counts = df.groupby("fileId")["annotator"].nunique()
    multi_annotator_files = file_annotator_counts[file_annotator_counts > 1].index
    return df[df["fileId"].isin(multi_annotator_files)]


def node_to_binary_vector(
    markers: list[str], all_labels: list[str] = ALL_LABELS
) -> list[int]:
    """Convert a list of markers to a binary vector."""
    return [1 if label in markers else 0 for label in all_labels]


def calculate_percent_agreement(vector1: list[int], vector2: list[int]) -> float:
    """Calculate simple percent agreement between two binary vectors."""
    if len(vector1) != len(vector2):
        return 0.0
    agreements = sum(1 for a, b in zip(vector1, vector2, strict=False) if a == b)
    return agreements / len(vector1) if len(vector1) > 0 else 0.0


def calculate_cohens_kappa_for_label(
    annotator1_labels: list[list[str]],
    annotator2_labels: list[list[str]],
    label: str,
) -> float | None:
    """
    Calculate Cohen's Kappa for a specific label between two annotators.

    Args:
        annotator1_labels: List of marker lists from annotator 1 (one per node)
        annotator2_labels: List of marker lists from annotator 2 (one per node)
        label: The specific label to calculate agreement for

    Returns:
        Cohen's Kappa score or None if cannot be calculated
    """
    # Convert to binary: 1 if label present, 0 if not
    y1 = [1 if label in markers else 0 for markers in annotator1_labels]
    y2 = [1 if label in markers else 0 for markers in annotator2_labels]

    # Need at least 2 samples and both classes present to calculate kappa
    if len(y1) < 2 or len(set(y1)) < 2 or len(set(y2)) < 2:
        return None

    try:
        return cohen_kappa_score(y1, y2)
    except Exception:
        return None


def calculate_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for multiple annotators.

    Args:
        ratings_matrix: Matrix of shape (n_subjects, n_categories) where each entry
                       is the count of annotators who assigned that category.

    Returns:
        Fleiss' Kappa score
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Assume same number of raters per subject

    if n_raters <= 1 or n_subjects == 0:
        return 0.0

    # Calculate P_i (agreement for each subject)
    P_i = (np.sum(ratings_matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Calculate P_j (proportion of assignments to each category)
    p_j = np.sum(ratings_matrix, axis=0) / (n_subjects * n_raters)
    P_e = np.sum(p_j**2)

    # Calculate Fleiss' Kappa
    if P_e == 1:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def calculate_krippendorff_alpha(reliability_data: np.ndarray) -> float:
    """
    Calculate Krippendorff's Alpha for multiple annotators with binary data.

    Args:
        reliability_data: Matrix of shape (n_annotators, n_items) with binary values.
                         Use np.nan for missing values.

    Returns:
        Krippendorff's Alpha score
    """
    # Remove items with only one annotation
    valid_cols = np.sum(~np.isnan(reliability_data), axis=0) >= 2
    data = reliability_data[:, valid_cols]

    if data.shape[1] == 0:
        return 0.0

    n_annotators, n_items = data.shape

    # Count coincidence matrix for binary data (0 and 1)
    coincidence = np.zeros((2, 2))

    for item in range(n_items):
        values = data[:, item]
        valid_values = values[~np.isnan(values)]
        n_valid = len(valid_values)

        if n_valid < 2:
            continue

        for i in range(len(valid_values)):
            for j in range(len(valid_values)):
                if i != j:
                    v_i, v_j = int(valid_values[i]), int(valid_values[j])
                    coincidence[v_i, v_j] += 1 / (n_valid - 1)

    # Calculate observed disagreement
    total = np.sum(coincidence)
    if total == 0:
        return 0.0

    n_c = np.sum(coincidence, axis=1)

    # Binary nominal metric: 1 if different, 0 if same
    D_o = coincidence[0, 1] + coincidence[1, 0]
    D_e = 2 * n_c[0] * n_c[1] / total if total > 0 else 0

    if D_e == 0:
        return 1.0

    return 1 - (D_o / D_e)


def get_pairwise_node_labels(
    df: pd.DataFrame, file_id: str
) -> dict[tuple[str, str], list[tuple[list[str], list[str]]]]:
    """
    Get aligned node labels for all annotator pairs for a given file.

    Returns a dict mapping (annotator1, annotator2) to list of (labels1, labels2) tuples.
    """
    file_annotations = df[df["fileId"] == file_id]
    annotators = file_annotations["annotator"].unique()

    if len(annotators) < 2:
        return {}

    # Parse nodes for each annotator
    annotator_nodes = {}
    for _, row in file_annotations.iterrows():
        annotator = row["annotator"]
        nodes = parse_file_nodes(row["fileNodes"])
        annotatable = get_annotatable_nodes(nodes)
        annotator_nodes[annotator] = {
            n["id"]: n.get("markers", []) for n in annotatable
        }

    # Align nodes for each pair
    result = {}
    for ann1, ann2 in combinations(annotators, 2):
        nodes1 = annotator_nodes[ann1]
        nodes2 = annotator_nodes[ann2]

        # Find common node IDs
        common_ids = set(nodes1.keys()) & set(nodes2.keys())

        aligned_labels = [
            (nodes1[node_id], nodes2[node_id])
            for node_id in sorted(
                common_ids, key=lambda x: int(x) if x.isdigit() else x
            )
        ]

        if aligned_labels:
            result[(ann1, ann2)] = aligned_labels

    return result


def calculate_agreement_metrics_for_group(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate agreement metrics for a group of annotations.

    Returns dict with:
        - per_label_kappa: Cohen's Kappa per label (averaged across pairs)
        - overall_kappa: Overall average Kappa
        - percent_agreement: Per-label percent agreement
        - fleiss_kappa: Fleiss' Kappa (if 3+ annotators)
        - krippendorff_alpha: Per-label Krippendorff's Alpha
    """
    # Filter to multi-annotator files
    filtered_df = filter_multi_annotator_files(df)

    if len(filtered_df) == 0:
        return {
            "per_label_kappa": {},
            "overall_kappa": None,
            "percent_agreement": {},
            "fleiss_kappa": None,
            "krippendorff_alpha": {},
            "n_files": 0,
            "n_annotator_pairs": 0,
        }

    # Collect all pairwise label comparisons
    all_pairwise_labels = defaultdict(list)  # label -> [(y1, y2), ...]

    file_ids = filtered_df["fileId"].unique()
    n_annotator_pairs = 0

    for file_id in file_ids:
        pairwise = get_pairwise_node_labels(filtered_df, file_id)

        for aligned_labels in pairwise.values():
            n_annotator_pairs += 1

            for labels1, labels2 in aligned_labels:
                for label in ALL_LABELS:
                    y1 = 1 if label in labels1 else 0
                    y2 = 1 if label in labels2 else 0
                    all_pairwise_labels[label].append((y1, y2))

    # Calculate per-label Cohen's Kappa
    per_label_kappa = {}
    per_label_agreement = {}
    per_label_alpha = {}

    for label in ALL_LABELS:
        pairs = all_pairwise_labels[label]

        if len(pairs) < 2:
            per_label_kappa[label] = None
            per_label_agreement[label] = None
            per_label_alpha[label] = None
            continue

        y1_all = [p[0] for p in pairs]
        y2_all = [p[1] for p in pairs]

        # Percent agreement
        agreements = sum(1 for a, b in zip(y1_all, y2_all, strict=False) if a == b)
        per_label_agreement[label] = agreements / len(pairs)

        # Cohen's Kappa (needs variance in both)
        if len(set(y1_all)) >= 2 and len(set(y2_all)) >= 2:
            try:
                per_label_kappa[label] = cohen_kappa_score(y1_all, y2_all)
            except Exception:
                per_label_kappa[label] = None
        else:
            per_label_kappa[label] = None

        # Krippendorff's Alpha (approximation using pairwise data)
        reliability_matrix = np.array([y1_all, y2_all], dtype=float)
        per_label_alpha[label] = calculate_krippendorff_alpha(reliability_matrix)

    # Overall Kappa (average of per-label kappas)
    valid_kappas = [k for k in per_label_kappa.values() if k is not None]
    overall_kappa = np.mean(valid_kappas) if valid_kappas else None

    return {
        "per_label_kappa": per_label_kappa,
        "overall_kappa": overall_kappa,
        "percent_agreement": per_label_agreement,
        "krippendorff_alpha": per_label_alpha,
        "n_files": len(file_ids),
        "n_annotator_pairs": n_annotator_pairs,
    }


def calculate_cooccurrence_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Calculate label co-occurrence matrix.

    Returns:
        - DataFrame with co-occurrence counts
        - Dict with co-occurrence pairs sorted by count
    """
    cooccurrence = Counter()
    label_counts = Counter()

    for _, row in df.iterrows():
        nodes = parse_file_nodes(row["fileNodes"])
        annotatable = get_annotatable_nodes(nodes)

        for node in annotatable:
            markers = node.get("markers", [])
            for m in markers:
                label_counts[m] += 1

            if len(markers) > 1:
                for pair in combinations(sorted(markers), 2):
                    cooccurrence[pair] += 1

    # Create matrix
    matrix = pd.DataFrame(0, index=ALL_LABELS, columns=ALL_LABELS)

    for (l1, l2), count in cooccurrence.items():
        if l1 in ALL_LABELS and l2 in ALL_LABELS:
            matrix.loc[l1, l2] = count
            matrix.loc[l2, l1] = count

    # Diagonal = individual counts
    for label, count in label_counts.items():
        if label in ALL_LABELS:
            matrix.loc[label, label] = count

    # Sort pairs by count
    sorted_pairs = dict(sorted(cooccurrence.items(), key=lambda x: -x[1]))

    return matrix, sorted_pairs


def calculate_jaccard_similarity(
    cooccurrence_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Jaccard similarity between all label pairs.
    """
    jaccard = pd.DataFrame(0.0, index=ALL_LABELS, columns=ALL_LABELS)

    for l1 in ALL_LABELS:
        for l2 in ALL_LABELS:
            if l1 == l2:
                jaccard.loc[l1, l2] = 1.0
            else:
                count_l1 = cooccurrence_matrix.loc[l1, l1]
                count_l2 = cooccurrence_matrix.loc[l2, l2]
                count_both = cooccurrence_matrix.loc[l1, l2]

                union = count_l1 + count_l2 - count_both
                jaccard.loc[l1, l2] = count_both / union if union > 0 else 0.0

    return jaccard


def calculate_annotator_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate label usage profile per annotator.

    Returns DataFrame with normalized label usage per annotator.
    """
    label_cols = [
        c
        for c in df.columns
        if c.endswith("_count")
        and c != "positive_marker_count"
        and c != "negative_marker_count"
        and c != "neutral_marker_count"
    ]

    profiles = df.groupby("annotator")[label_cols].sum()

    # Normalize by total labels used
    total_per_annotator = profiles.sum(axis=1)
    return profiles.div(total_per_annotator, axis=0)


def calculate_annotator_agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pairwise agreement between all annotators.
    """
    filtered_df = filter_multi_annotator_files(df)
    annotators = df["annotator"].unique()

    agreement_matrix = pd.DataFrame(0.0, index=annotators, columns=annotators)
    pair_counts = pd.DataFrame(0, index=annotators, columns=annotators)

    file_ids = filtered_df["fileId"].unique()

    for file_id in file_ids:
        pairwise = get_pairwise_node_labels(filtered_df, file_id)

        for (ann1, ann2), aligned_labels in pairwise.items():
            if not aligned_labels:
                continue

            # Calculate average agreement across all nodes and labels
            total_agreement = 0
            total_comparisons = 0

            for labels1, labels2 in aligned_labels:
                vec1 = node_to_binary_vector(labels1)
                vec2 = node_to_binary_vector(labels2)
                total_agreement += calculate_percent_agreement(vec1, vec2)
                total_comparisons += 1

            if total_comparisons > 0:
                avg_agreement = total_agreement / total_comparisons
                agreement_matrix.loc[ann1, ann2] += avg_agreement
                agreement_matrix.loc[ann2, ann1] += avg_agreement
                pair_counts.loc[ann1, ann2] += 1
                pair_counts.loc[ann2, ann1] += 1

    # Average the agreements
    for ann1 in annotators:
        for ann2 in annotators:
            if pair_counts.loc[ann1, ann2] > 0:
                agreement_matrix.loc[ann1, ann2] /= pair_counts.loc[ann1, ann2]
            if ann1 == ann2:
                agreement_matrix.loc[ann1, ann2] = 1.0

    return agreement_matrix


def generate_recommendations(
    agreement_metrics: dict,
    cooccurrence_pairs: dict,
    annotator_profiles: pd.DataFrame,
    label_counts: dict,
) -> dict[str, list]:
    """
    Generate recommendations for label consolidation.

    Decision Framework:
    - Merge: High co-occurrence (>70%) between two labels
    - Drop: Label used <50 times total OR low agreement (<0.4 Kappa)
    - Clarify: One annotator uses label 10x more than others
    - Keep: Good agreement and reasonable usage
    """
    recommendations = {
        "merge": [],
        "drop": [],
        "clarify": [],
        "keep": [],
    }

    per_label_kappa = agreement_metrics.get("per_label_kappa", {})

    # Check for labels to merge based on co-occurrence
    total_label_counts = defaultdict(int)
    for (l1, l2), count in cooccurrence_pairs.items():
        total_label_counts[l1] += count
        total_label_counts[l2] += count

    # Check high co-occurrence pairs
    checked_pairs = set()
    for (l1, l2), count in list(cooccurrence_pairs.items())[:20]:  # Top 20 pairs
        if (l1, l2) in checked_pairs or (l2, l1) in checked_pairs:
            continue
        checked_pairs.add((l1, l2))

        # Calculate conditional probability
        count_l1 = label_counts.get(l1, 1)
        count_l2 = label_counts.get(l2, 1)

        cond_prob_l1_given_l2 = count / count_l2 if count_l2 > 0 else 0
        cond_prob_l2_given_l1 = count / count_l1 if count_l1 > 0 else 0

        if cond_prob_l1_given_l2 > 0.5 or cond_prob_l2_given_l1 > 0.5:
            recommendations["merge"].append(
                {
                    "labels": f"{l1}|{l2}",
                    "label_1": l1,
                    "label_2": l2,
                    "cooccurrence_count": count,
                    "cond_prob_l1_given_l2": cond_prob_l1_given_l2,
                    "cond_prob_l2_given_l1": cond_prob_l2_given_l1,
                    "reason": "High conditional probability of co-occurrence",
                }
            )

    # Check for labels to drop or clarify
    for label in ALL_LABELS:
        count = label_counts.get(label, 0)
        kappa = per_label_kappa.get(label)

        # Check if one annotator dominates usage
        if label.replace("_", "_") + "_count" in annotator_profiles.columns:
            col = label + "_count"
            if col in annotator_profiles.columns:
                usage = annotator_profiles[col]
                max_usage = usage.max()
                mean_usage = usage.mean()

                if max_usage > 3 * mean_usage and mean_usage > 0:
                    recommendations["clarify"].append(
                        {
                            "label": label,
                            "reason": f"High variance in annotator usage (max: {max_usage:.3f}, mean: {mean_usage:.3f})",
                            "dominant_annotator": usage.idxmax(),
                        }
                    )

        # Drop criteria
        if count < 50:
            recommendations["drop"].append(
                {
                    "label": label,
                    "count": count,
                    "reason": "Rarely used (<50 instances)",
                }
            )
        elif kappa is not None and kappa < 0.4:
            recommendations["drop"].append(
                {
                    "label": label,
                    "kappa": kappa,
                    "reason": "Low inter-annotator agreement (Kappa < 0.4)",
                }
            )
        elif kappa is not None and kappa >= 0.6:
            recommendations["keep"].append(
                {
                    "label": label,
                    "kappa": kappa,
                    "count": count,
                    "reason": "Good agreement and usage",
                }
            )

    return recommendations


def analyze_by_environment_and_level(df: pd.DataFrame) -> dict[str, dict]:
    """
    Perform analysis grouped by environment and level.
    """
    results = {}

    # Get unique environment-level combinations
    groups = df.groupby(["environment", "level"]).size().reset_index()

    for _, group_row in groups.iterrows():
        env = group_row["environment"]
        level = group_row["level"]

        group_df = df[(df["environment"] == env) & (df["level"] == level)]

        # Filter to multi-annotator files
        filtered_df = filter_multi_annotator_files(group_df)

        if len(filtered_df) == 0:
            continue

        key = f"{env}_level_{level}"

        # Calculate agreement metrics
        agreement = calculate_agreement_metrics_for_group(filtered_df)

        # Calculate co-occurrence for this group
        cooccur_matrix, cooccur_pairs = calculate_cooccurrence_matrix(filtered_df)

        # Calculate annotator profiles for this group
        profiles = calculate_annotator_profiles(filtered_df)

        # Get label counts
        label_counts = {}
        for label in ALL_LABELS:
            count_col = label + "_count"
            if count_col in filtered_df.columns:
                label_counts[label] = filtered_df[count_col].sum()

        results[key] = {
            "environment": env,
            "level": level,
            "n_annotations": len(group_df),
            "n_multi_annotator": len(filtered_df),
            "annotators": list(filtered_df["annotator"].unique()),
            "agreement_metrics": agreement,
            "top_cooccurrences": {
                f"{k[0]}|{k[1]}": v for k, v in list(cooccur_pairs.items())[:10]
            },
            "annotator_profiles": profiles.to_dict() if len(profiles) > 0 else {},
            "label_counts": label_counts,
        }

    return results


def format_results_report(results: dict, output_path: Path) -> None:
    """
    Format and save the analysis results as a markdown report.
    """
    lines = [
        "# Annotation Analysis Results",
        "",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # Overall summary
    if "overall" in results:
        overall = results["overall"]
        lines.extend(
            [
                "## Overall Summary",
                "",
                f"- **Total annotations analyzed**: {overall.get('n_annotations', 'N/A')}",
                f"- **Files with multiple annotators**: {overall.get('n_multi_annotator_files', 'N/A')}",
                f"- **Unique annotators**: {overall.get('n_annotators', 'N/A')}",
                "",
            ]
        )

    # Agreement metrics summary
    if "agreement_summary" in results:
        agreement = results["agreement_summary"]
        lines.extend(
            [
                "## Inter-Annotator Agreement Summary",
                "",
                "### Per-Label Cohen's Kappa",
                "",
                "| Label | Kappa | Interpretation |",
                "|-------|-------|----------------|",
            ]
        )

        per_label = agreement.get("per_label_kappa", {})
        for label in ALL_LABELS:
            kappa = per_label.get(label)
            if kappa is not None:
                if kappa >= 0.8:
                    interp = "Almost perfect"
                elif kappa >= 0.6:
                    interp = "Substantial"
                elif kappa >= 0.4:
                    interp = "Moderate"
                elif kappa >= 0.2:
                    interp = "Fair"
                else:
                    interp = "Slight"
                lines.append(f"| {label} | {kappa:.3f} | {interp} |")
            else:
                lines.append(f"| {label} | N/A | Insufficient data |")

        lines.extend(["", ""])

    # Per environment-level results
    if "by_environment_level" in results:
        lines.extend(
            [
                "## Results by Environment and Level",
                "",
            ]
        )

        for _key, data in sorted(results["by_environment_level"].items()):
            env = data["environment"]
            level = data["level"]

            lines.extend(
                [
                    f"### {env.upper()} - Level {level}",
                    "",
                    f"- **Annotations**: {data['n_annotations']}",
                    f"- **Multi-annotator files**: {data['n_multi_annotator']}",
                    f"- **Annotators**: {', '.join(data['annotators'])}",
                    "",
                ]
            )

            # Agreement for this group
            agreement = data.get("agreement_metrics", {})
            if agreement.get("per_label_kappa"):
                lines.append("**Agreement Metrics (Top 5 by Kappa):**")
                lines.append("")
                lines.append("| Label | Kappa | % Agreement |")
                lines.append("|-------|-------|-------------|")

                kappas = agreement["per_label_kappa"]
                agreements = agreement.get("percent_agreement", {})

                # Sort by kappa, filter valid
                sorted_labels = sorted(
                    [(label, k) for label, k in kappas.items() if k is not None],
                    key=lambda x: -x[1],
                )[:5]

                for label, kappa in sorted_labels:
                    pct = agreements.get(label, 0) * 100 if agreements.get(label) else 0
                    lines.append(f"| {label} | {kappa:.3f} | {pct:.1f}% |")

                lines.extend(["", ""])

    # Recommendations
    if "recommendations" in results:
        recs = results["recommendations"]
        lines.extend(
            [
                "## Recommendations",
                "",
            ]
        )

        if recs.get("merge"):
            lines.extend(
                [
                    "### Labels to Consider Merging",
                    "",
                ]
            )
            for item in recs["merge"][:5]:
                l1 = item.get("label_1", "")
                l2 = item.get("label_2", "")
                lines.append(f"- **{l1}** ↔ **{l2}**: {item['reason']}")
            lines.append("")

        if recs.get("drop"):
            lines.extend(
                [
                    "### Labels to Consider Dropping",
                    "",
                ]
            )
            lines.extend(
                f"- **{item['label']}**: {item['reason']}" for item in recs["drop"][:5]
            )
            lines.append("")

        if recs.get("clarify"):
            lines.extend(
                [
                    "### Labels Needing Clarification",
                    "",
                ]
            )
            lines.extend(
                f"- **{item['label']}**: {item['reason']}"
                for item in recs["clarify"][:5]
            )
            lines.append("")

        if recs.get("keep"):
            lines.extend(
                [
                    "### Labels Working Well",
                    "",
                ]
            )
            lines.extend(
                f"- **{item['label']}**: Kappa = {item.get('kappa', 'N/A'):.3f}"
                for item in recs["keep"][:5]
            )
            lines.append("")

    # Co-occurrence summary
    if "cooccurrence_top_pairs" in results:
        lines.extend(
            [
                "## Label Co-occurrence Analysis",
                "",
                "### Top Co-occurring Label Pairs",
                "",
                "| Label 1 | Label 2 | Count |",
                "|---------|---------|-------|",
            ]
        )

        for key, count in list(results["cooccurrence_top_pairs"].items())[:10]:
            parts = key.split("|")
            if len(parts) == 2:
                l1, l2 = parts
                lines.append(f"| {l1} | {l2} | {count} |")

        lines.append("")

    # Annotator profiles
    if "annotator_agreement_matrix" in results:
        lines.extend(
            [
                "## Annotator Agreement Matrix",
                "",
                "Pairwise percent agreement between annotators:",
                "",
            ]
        )

        matrix = results["annotator_agreement_matrix"]
        if isinstance(matrix, pd.DataFrame):
            lines.append(matrix.round(3).to_markdown())
        lines.append("")

    # Write to file
    with Path(output_path).open("w") as f:
        f.write("\n".join(lines))


def main(output_dir: str = "reports_v2/annotations/analysis_results"):
    """Main analysis function."""
    # Setup paths
    base_path = Path(__file__).parent
    data_path = base_path / "data" / "cleaned_corral_annotations.parquet"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Annotation Analysis")
    logger.info("=" * 60)

    # Load data
    logger.info("\n1. Loading data...")
    annotations_df = load_data(data_path)
    logger.info("   - Total annotations: %d", len(annotations_df))
    logger.info("   - Unique annotators: %d", annotations_df["annotator"].nunique())
    logger.info(
        "   - Environments: %s", annotations_df["environment"].unique().tolist()
    )
    logger.info("   - Levels: %s", annotations_df["level"].unique().tolist())

    # Filter to multi-annotator files
    logger.info("\n2. Filtering to multi-annotator files...")
    multi_df = filter_multi_annotator_files(annotations_df)
    logger.info("   - Files with multiple annotators: %d", multi_df["fileId"].nunique())
    logger.info("   - Annotations from multi-annotator files: %d", len(multi_df))

    # Calculate overall agreement metrics
    logger.info("\n3. Calculating overall agreement metrics...")
    overall_agreement = calculate_agreement_metrics_for_group(multi_df)
    logger.info(
        "   - Overall average Kappa: %.3f", overall_agreement["overall_kappa"]
    ) if overall_agreement["overall_kappa"] else logger.info("   - Overall Kappa: N/A")

    # Calculate co-occurrence
    logger.info("\n4. Calculating label co-occurrence...")
    cooccur_matrix, cooccur_pairs = calculate_cooccurrence_matrix(multi_df)
    logger.info(
        "   - Top co-occurring pair: %s",
        next(iter(cooccur_pairs.items())) if cooccur_pairs else "N/A",
    )

    # Calculate Jaccard similarity (used per group)
    logger.info("\n5. Calculating Jaccard similarity...")

    # Calculate annotator profiles (used per group)
    logger.info("\n6. Calculating annotator profiles...")

    # Calculate annotator agreement matrix (used per group)
    logger.info("\n7. Calculating annotator agreement matrix...")

    # Get label counts (overall)
    label_counts = {
        label: int(multi_df[f"{label}_count"].sum())
        if f"{label}_count" in multi_df.columns
        else 0
        for label in ALL_LABELS
    }

    # Analyze by environment and level
    logger.info("\n8. Analyzing by environment and level...")

    # Build structured output JSON
    output_json = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_annotations": len(annotations_df),
            "multi_annotator_files": int(multi_df["fileId"].nunique()),
            "n_annotators": int(annotations_df["annotator"].nunique()),
            "annotators": annotations_df["annotator"].unique().tolist(),
            "environments": annotations_df["environment"].unique().tolist(),
            "levels": [int(x) for x in annotations_df["level"].unique().tolist()],
        },
        "overall": {
            "agreement_metrics": {
                "overall_kappa": overall_agreement["overall_kappa"],
                "per_label_kappa": overall_agreement["per_label_kappa"],
                "percent_agreement": overall_agreement["percent_agreement"],
                "krippendorff_alpha": overall_agreement["krippendorff_alpha"],
                "n_files": overall_agreement["n_files"],
                "n_annotator_pairs": overall_agreement["n_annotator_pairs"],
            },
            "label_counts": label_counts,
            "cooccurrence_top_pairs": [
                {"label_1": k[0], "label_2": k[1], "count": v}
                for k, v in list(cooccur_pairs.items())[:20]
            ],
        },
        "by_environment_level": {},
    }

    # Analyze each environment-level group
    groups = annotations_df.groupby(["environment", "level"]).size().reset_index()

    for _, group_row in groups.iterrows():
        env = group_row["environment"]
        level = int(group_row["level"])

        group_df = annotations_df[
            (annotations_df["environment"] == env) & (annotations_df["level"] == level)
        ]
        filtered_df = filter_multi_annotator_files(group_df)

        if len(filtered_df) == 0:
            continue

        key = f"{env}_level_{level}"

        # Calculate agreement metrics for this group
        agreement = calculate_agreement_metrics_for_group(filtered_df)

        # Calculate co-occurrence for this group
        group_cooccur_matrix, group_cooccur_pairs = calculate_cooccurrence_matrix(
            filtered_df
        )

        # Calculate Jaccard similarity for this group
        group_jaccard = calculate_jaccard_similarity(group_cooccur_matrix)

        # Calculate annotator profiles for this group
        group_profiles = calculate_annotator_profiles(filtered_df)

        # Calculate annotator agreement matrix for this group
        group_ann_agreement = calculate_annotator_agreement_matrix(filtered_df)

        # Get label counts for this group
        group_label_counts = {
            label: int(filtered_df[f"{label}_count"].sum())
            if f"{label}_count" in filtered_df.columns
            else 0
            for label in ALL_LABELS
        }

        # Generate recommendations for this group
        group_recommendations = generate_recommendations(
            agreement, group_cooccur_pairs, group_profiles, group_label_counts
        )

        output_json["by_environment_level"][key] = {
            "environment": env,
            "level": level,
            "n_annotations": len(group_df),
            "n_multi_annotator": len(filtered_df),
            "annotators": list(filtered_df["annotator"].unique()),
            "agreement_metrics": {
                "overall_kappa": agreement["overall_kappa"],
                "per_label_kappa": agreement["per_label_kappa"],
                "percent_agreement": agreement["percent_agreement"],
                "krippendorff_alpha": agreement["krippendorff_alpha"],
                "n_files": agreement["n_files"],
                "n_annotator_pairs": agreement["n_annotator_pairs"],
            },
            "label_counts": group_label_counts,
            "cooccurrence_top_pairs": [
                {"label_1": k[0], "label_2": k[1], "count": v}
                for k, v in list(group_cooccur_pairs.items())[:10]
            ],
            "jaccard_similarity_top_pairs": _get_top_jaccard_pairs(group_jaccard, n=10),
            "annotator_agreement_matrix": {
                ann1: {
                    ann2: float(group_ann_agreement.loc[ann1, ann2])
                    for ann2 in group_ann_agreement.columns
                }
                for ann1 in group_ann_agreement.index
            },
            "annotator_profiles": group_profiles.to_dict()
            if len(group_profiles) > 0
            else {},
            "recommendations": group_recommendations,
        }

    logger.info("   - Groups analyzed: %d", len(output_json["by_environment_level"]))

    # Save the single JSON output
    logger.info("\n10. Saving results...")
    json_path = output_path / "annotation_analysis.json"

    with json_path.open("w") as f:
        json.dump(output_json, f, indent=2, default=str)
    logger.info("   - JSON results: %s", json_path)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)

    # Print summary
    logger.info("\n### Quick Summary ###")
    if overall_agreement["overall_kappa"]:
        logger.info("\nOverall Kappa: %.3f", overall_agreement["overall_kappa"])
    else:
        logger.info("\nOverall Kappa: N/A")

    logger.info("\nTop 5 labels by agreement (Kappa):")
    kappas = [
        (lab, k)
        for lab, k in overall_agreement["per_label_kappa"].items()
        if k is not None
    ]
    for label, kappa in sorted(kappas, key=lambda x: -x[1])[:5]:
        logger.info("  - %s: %.3f", label, kappa)

    logger.info("\nTop 5 labels by disagreement (lowest Kappa):")
    for label, kappa in sorted(kappas, key=lambda x: x[1])[:5]:
        logger.info("  - %s: %.3f", label, kappa)

    return output_json


def _get_top_jaccard_pairs(jaccard_df: pd.DataFrame, n: int = 20) -> list[dict]:
    """Extract top Jaccard similarity pairs (excluding self-comparisons)."""
    pairs = []
    for i, l1 in enumerate(jaccard_df.index):
        for j, l2 in enumerate(jaccard_df.columns):
            if i < j:  # Upper triangle only, excluding diagonal
                sim = jaccard_df.loc[l1, l2]
                if sim > 0:
                    pairs.append(
                        {"label_1": l1, "label_2": l2, "similarity": float(sim)}
                    )

    # Sort by similarity descending
    pairs.sort(key=lambda x: -x["similarity"])
    return pairs[:n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze annotation agreement and patterns"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports_v2/annotations/analysis_results",
        help="Directory to save output files",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir)
