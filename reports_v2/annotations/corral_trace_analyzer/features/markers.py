"""
Marker-based features
"""

from typing import Any

import pandas as pd

from .base import BaseFeatureExtractor
from .constant import ALL_MARKERS


class MarkerFeatures(BaseFeatureExtractor):
    """Extract marker-based features"""

    ALL_FEATURES = ALL_MARKERS

    def __init__(
        self, node_types: list[str] | None = None, features: str | list[str] = "all"
    ):
        """
        Initialize marker feature extractor

        Args:
            node_types: List of node types to include. Default is ['assistant']
                       since markers only exist on assistant nodes.
            features: "all" or a list of feature names to extract.
        """
        # Default to assistant nodes only (where markers exist)
        if node_types is None:
            node_types = ["assistant"]
        super().__init__(node_types=node_types)

        if features == "all":
            self.feature_names = self.ALL_FEATURES
        else:
            self.feature_names = [f for f in self.ALL_FEATURES if f in features]

    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract marker features"""

        # Filter steps by node type (default: assistant nodes only)
        filtered_steps_df = self.filter_steps_by_node_type(steps_df)

        features_list = []

        for trace_id in traces_df["trace_id"].unique():
            trace_data = traces_df[traces_df["trace_id"] == trace_id].iloc[0]
            trace_steps = filtered_steps_df[
                filtered_steps_df["trace_id"] == trace_id
            ].sort_values("step_index")
            trace_tools = tools_df[tools_df["trace_id"] == trace_id]

            features = self._extract_for_trace(
                trace_id, trace_data, trace_steps, trace_tools
            )
            features_list.append(features)

        if not features_list:
            return pd.DataFrame(
                index=pd.Index([], name="trace_id"), columns=self.feature_names
            )

        df_ = pd.DataFrame(features_list).set_index("trace_id")

        # Return only requested features, handling cases where a feature might not be computed for any trace
        return_cols = [col for col in self.feature_names if col in df_.columns]
        return df_[return_cols]

    def _extract_for_trace(
        self,
        trace_id: str,
        trace_data: pd.Series,
        steps_df: pd.DataFrame,
        tools_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Extract features for a single trace"""

        features = {"trace_id": trace_id}

        # Total markers
        total_markers = (
            trace_data["positive_marker_count"]
            + trace_data["negative_marker_count"]
            + trace_data["neutral_marker_count"]
        )
        features["total_markers"] = total_markers

        # Marker balance
        features["marker_balance"] = (
            trace_data["positive_marker_count"] - trace_data["negative_marker_count"]
        )

        # Marker ratios
        if total_markers > 0:
            features["marker_balance_ratio"] = (
                features["marker_balance"] / total_markers
            )
            features["positive_marker_ratio"] = (
                trace_data["positive_marker_count"] / total_markers
            )
            features["negative_marker_ratio"] = (
                trace_data["negative_marker_count"] / total_markers
            )
            features["neutral_marker_ratio"] = (
                trace_data["neutral_marker_count"] / total_markers
            )
        else:
            features["marker_balance_ratio"] = 0.0
            features["positive_marker_ratio"] = 0.0
            features["negative_marker_ratio"] = 0.0
            features["neutral_marker_ratio"] = 0.0

        # Ratios to total steps
        total_steps = len(steps_df)
        if total_steps > 0:
            features["validation_to_total_ratio"] = (
                trace_data["validation_attempt_count"] / total_steps
            )
            features["backtrack_to_total_ratio"] = (
                trace_data["backtrack_trigger_count"] / total_steps
            )
            features["planning_to_total_ratio"] = (
                trace_data["planning_statement_count"] / total_steps
            )
            features["reasoning_to_total_ratio"] = (
                trace_data["reasoning_statement_count"] / total_steps
            )
        else:
            features["validation_to_total_ratio"] = 0.0
            features["backtrack_to_total_ratio"] = 0.0
            features["planning_to_total_ratio"] = 0.0
            features["reasoning_to_total_ratio"] = 0.0

        # Recovery behavior after errors
        features["validation_after_error_rate"] = self._calculate_marker_after_error(
            steps_df, tools_df, "validation_attempt"
        )
        features["backtrack_after_error_rate"] = self._calculate_marker_after_error(
            steps_df, tools_df, "backtrack_trigger"
        )

        # Problem markers (count of various negative indicators)
        problem_markers = (
            trace_data["unnecessary_tool_use_count"]
            + trace_data["non_sense_count"]
            + trace_data["loop_instance_count"]
            + trace_data["hallucination_count"]
            + trace_data["wrong_planning_count"]
            + trace_data["wrong_reasoning_count"]
            + trace_data["syntax_error_count"]
            + trace_data["early_final_answer_count"]
            + trace_data["give_up_count"]
            + trace_data["inefficient_tool_call_count"]
            + trace_data["misunderstood_tool_count"]
        )
        features["problem_marker_count"] = problem_markers
        features["problem_marker_ratio"] = (
            problem_markers / total_steps if total_steps > 0 else 0.0
        )

        # Recovery markers (validation + backtrack)
        recovery_markers = (
            trace_data["validation_attempt_count"]
            + trace_data["backtrack_trigger_count"]
        )
        features["recovery_marker_count"] = recovery_markers
        features["recovery_marker_ratio"] = (
            recovery_markers / total_steps if total_steps > 0 else 0.0
        )

        return features

    def _calculate_marker_after_error(
        self, steps_df: pd.DataFrame, tools_df: pd.DataFrame, marker_name: str
    ) -> float:
        """
        Calculate rate of a specific marker appearing after errors

        Args:
            steps_df: Steps dataframe
            tools_df: Tools dataframe
            marker_name: Name of marker to look for

        Returns:
            Ratio of errors followed by the marker
        """
        if len(tools_df) == 0:
            return 0.0

        error_tools = tools_df[~tools_df["is_success"]]
        if len(error_tools) == 0:
            return 0.0

        # Count how many errors are followed by the marker within next 3 steps
        marker_after_error = 0

        for tool_idx in error_tools["tool_index"]:
            # Look for marker in next few steps after this tool call
            # Tool calls and steps may not align perfectly, so we look in a window
            next_steps = steps_df[steps_df["step_index"] > tool_idx]
            next_steps = next_steps.head(3)  # Look at next 3 steps

            # Check if marker appears in any of these steps
            has_marker = any(
                marker_name in markers
                for markers in next_steps["markers"]
                if isinstance(markers, list)
            )

            if has_marker:
                marker_after_error += 1

        return marker_after_error / len(error_tools)
