"""
Temporal and sequence-based features
"""

from typing import Any

import numpy as np
import pandas as pd

from corral_trace_analyzer.config import TEMPORAL_WINDOWS

from .base import BaseFeatureExtractor
from .constant import ALL_TEMPORAL


class TemporalFeatures(BaseFeatureExtractor):
    """Extract temporal and sequence-based features"""

    ALL_FEATURES = ALL_TEMPORAL

    def __init__(
        self, node_types: list | None = None, features: str | list[str] = "all"
    ):
        """
        Initialize temporal feature extractor

        Args:
            node_types: List of node types to include. Default is None (all types)
                       since temporal analysis needs full sequence including
                       assistant, tool, user, and system nodes for accurate
                       step counts and timing.
            features: "all" or a list of feature names to extract.
        """
        # Default to ALL node types for temporal analysis
        # We need all nodes to accurately count steps and track sequence
        super().__init__(node_types=node_types)

        if features == "all":
            self.feature_names = self.ALL_FEATURES
        else:
            self.feature_names = [f for f in self.ALL_FEATURES if f in features]

        # Dynamically add windowed features to feature_names if not already there
        for start, end in TEMPORAL_WINDOWS:
            label = f"{start}_{end}" if end != float("inf") else f"{start}_plus"
            pos_feature = f"positive_markers_steps_{label}"
            neg_feature = f"negative_markers_steps_{label}"
            if features == "all":
                if pos_feature not in self.feature_names:
                    self.feature_names.append(pos_feature)
                if neg_feature not in self.feature_names:
                    self.feature_names.append(neg_feature)
            else:
                if pos_feature in features and pos_feature not in self.feature_names:
                    self.feature_names.append(pos_feature)
                if neg_feature in features and neg_feature not in self.feature_names:
                    self.feature_names.append(neg_feature)

    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract temporal features"""

        features_list = []

        for trace_id in traces_df["trace_id"].unique():
            trace_steps = steps_df[steps_df["trace_id"] == trace_id].sort_values(
                "step_index"
            )
            trace_tools = tools_df[tools_df["trace_id"] == trace_id]

            features = self._extract_for_trace(trace_id, trace_steps, trace_tools)
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
        self, trace_id: str, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Extract features for a single trace"""

        features = {"trace_id": trace_id}

        # Basic counts
        features["step_count_total"] = len(steps_df)
        features["assistant_steps_count"] = (steps_df["node_type"] == "assistant").sum()
        features["tool_steps_count"] = (steps_df["node_type"] == "tool").sum()
        features["annotatable_steps_count"] = steps_df["annotatable"].sum()

        # Steps to first events (counting all nodes)
        features["steps_to_first_tool_call"] = self._steps_to_first(
            steps_df, lambda row: row["node_type"] == "tool"
        )

        features["steps_to_first_error"] = self._steps_to_first_tool_error(tools_df)

        features["steps_to_first_positive_marker"] = self._steps_to_first(
            steps_df, lambda row: row["has_positive"]
        )

        features["steps_to_first_negative_marker"] = self._steps_to_first(
            steps_df, lambda row: row["has_negative"]
        )

        features["steps_to_first_planning"] = self._steps_to_first(
            steps_df, lambda row: row["has_planning"]
        )

        features["steps_to_first_reasoning"] = self._steps_to_first(
            steps_df, lambda row: row["has_reasoning"]
        )

        # Assistant-only step counts (counting only assistant nodes)
        assistant_steps = steps_df[steps_df["node_type"] == "assistant"]

        features["assistant_steps_to_first_tool_call"] = (
            self._assistant_steps_to_first_tool_call(steps_df, tools_df)
        )

        features["assistant_steps_to_first_error"] = (
            self._assistant_steps_to_first_tool_error(steps_df, tools_df)
        )

        features["assistant_steps_to_first_positive_marker"] = self._steps_to_first(
            assistant_steps, lambda row: row["has_positive"]
        )

        features["assistant_steps_to_first_negative_marker"] = self._steps_to_first(
            assistant_steps, lambda row: row["has_negative"]
        )

        features["assistant_steps_to_first_planning"] = self._steps_to_first(
            assistant_steps, lambda row: row["has_planning"]
        )

        features["assistant_steps_to_first_reasoning"] = self._steps_to_first(
            assistant_steps, lambda row: row["has_reasoning"]
        )

        # Marker trajectory slope (positive ratio over time)
        features["marker_trajectory_slope"] = self._calculate_marker_trajectory_slope(
            steps_df
        )

        # Early negative rate (first 10 steps)
        early_steps = steps_df[steps_df["step_index"] < 10]
        if len(early_steps) > 0:
            features["early_negative_rate"] = early_steps["has_negative"].sum() / len(
                early_steps
            )
        else:
            features["early_negative_rate"] = 0.0

        # Windowed marker counts
        for window_start, window_end in TEMPORAL_WINDOWS:
            window_steps = steps_df[
                (steps_df["step_index"] >= window_start - 1)
                & (steps_df["step_index"] < window_end)
            ]

            window_label = (
                f"{window_start}_{window_end}"
                if window_end != float("inf")
                else f"{window_start}_plus"
            )

            features[f"positive_markers_steps_{window_label}"] = window_steps[
                "has_positive"
            ].sum()
            features[f"negative_markers_steps_{window_label}"] = window_steps[
                "has_negative"
            ].sum()

        # First step features
        features["has_planning_in_first_step"] = self._has_planning_in_first_step(
            steps_df
        )
        features["has_reasoning_in_first_step"] = self._has_reasoning_in_first_step(
            steps_df
        )
        features["has_tool_error_in_first_step"] = self._has_tool_error_in_first_step(
            tools_df
        )

        # Longest looping sequence
        features["longest_looping_sequence"] = self._longest_looping_sequence(steps_df)

        return features

    def _steps_to_first(self, steps_df: pd.DataFrame, condition_func) -> int:
        """
        Calculate steps to first occurrence of a condition

        Args:
            steps_df: DataFrame with steps
            condition_func: Function that takes a row and returns True/False

        Returns:
            Number of steps to first occurrence, or -1 if never occurs
        """
        matching_steps = steps_df[steps_df.apply(condition_func, axis=1)]

        if len(matching_steps) > 0:
            return matching_steps.iloc[0]["step_index"] + 1  # +1 to make it 1-indexed
        return -1

    def _steps_to_first_tool_error(self, tools_df: pd.DataFrame) -> int:
        """Calculate steps to first tool error"""
        if len(tools_df) == 0:
            return -1

        error_tools = tools_df[~tools_df["is_success"]]
        if len(error_tools) > 0:
            return error_tools.iloc[0]["tool_index"] + 1
        return -1

    def _calculate_marker_trajectory_slope(self, steps_df: pd.DataFrame) -> float:
        """
        Calculate the slope of positive marker ratio over time

        Returns:
            Slope of the linear fit, or 0 if not enough data
        """
        if len(steps_df) < 2:
            return 0.0

        # Calculate cumulative positive ratio at each step
        steps_with_markers = steps_df[steps_df["marker_count"] > 0].copy()

        if len(steps_with_markers) < 2:
            return 0.0

        steps_with_markers["cumulative_positive"] = steps_with_markers[
            "has_positive"
        ].cumsum()
        steps_with_markers["cumulative_total"] = range(1, len(steps_with_markers) + 1)
        steps_with_markers["positive_ratio"] = (
            steps_with_markers["cumulative_positive"]
            / steps_with_markers["cumulative_total"]
        )

        # Fit linear regression
        x = steps_with_markers["step_index"].to_numpy()
        y = steps_with_markers["positive_ratio"].to_numpy()

        if len(x) < 2:
            return 0.0

        # Simple linear regression
        return np.polyfit(x, y, 1)[0]

    def _has_planning_in_first_step(self, steps_df: pd.DataFrame) -> int:
        """
        Check if the first assistant step has a planning marker

        Returns:
            1 if first assistant step has planning, 0 otherwise
        """
        assistant_steps = steps_df[steps_df["node_type"] == "assistant"]
        if len(assistant_steps) > 0:
            first_step = assistant_steps.iloc[0]
            return 1 if first_step.get("has_planning", False) else 0
        return 0

    def _has_reasoning_in_first_step(self, steps_df: pd.DataFrame) -> int:
        """
        Check if the first assistant step has a reasoning marker

        Returns:
            1 if first assistant step has reasoning, 0 otherwise
        """
        assistant_steps = steps_df[steps_df["node_type"] == "assistant"]
        if len(assistant_steps) > 0:
            first_step = assistant_steps.iloc[0]
            return 1 if first_step.get("has_reasoning", False) else 0
        return 0

    def _has_tool_error_in_first_step(self, tools_df: pd.DataFrame) -> int:
        """
        Check if the first tool call resulted in an error

        Returns:
            1 if first tool call failed, 0 otherwise (including no tools)
        """
        if len(tools_df) > 0:
            first_tool = tools_df.iloc[0]
            return 0 if first_tool.get("is_success", True) else 1
        return 0

    def _longest_looping_sequence(self, steps_df: pd.DataFrame) -> int:
        """
        Calculate the longest consecutive sequence of steps with loop_instance marker

        Returns:
            Maximum count of consecutive steps with loop_instance marker
        """
        # Check if markers column contains loop_instance for each step
        has_loop = steps_df["markers"].apply(
            lambda m: "loop_instance" in m if isinstance(m, list) else False
        )

        max_consecutive = 0
        current_consecutive = 0

        for has_loop_marker in has_loop:
            if has_loop_marker:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _assistant_steps_to_first_tool_call(
        self, steps_df: pd.DataFrame, _tools_df: pd.DataFrame
    ) -> int:
        """
        Calculate assistant steps until first tool call

        Returns:
            Number of assistant steps (1-indexed) before first tool node appears,
            or -1 if no tool calls
        """
        # Find first tool node
        tool_steps = steps_df[steps_df["node_type"] == "tool"]
        if len(tool_steps) == 0:
            return -1

        first_tool_step_index = tool_steps.iloc[0]["step_index"]

        # Count assistant steps before this tool step
        assistant_steps_before = steps_df[
            (steps_df["node_type"] == "assistant")
            & (steps_df["step_index"] < first_tool_step_index)
        ]

        return (
            len(assistant_steps_before) + 1 if len(assistant_steps_before) >= 0 else -1
        )

    def _assistant_steps_to_first_tool_error(
        self, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> int:
        """
        Calculate assistant steps until first tool error

        Returns:
            Number of assistant steps (1-indexed) before first error tool node appears,
            or -1 if no errors
        """
        if len(tools_df) == 0:
            return -1

        # Find first error tool
        error_tools = tools_df[~tools_df["is_success"]]
        if len(error_tools) == 0:
            return -1

        # Get the step_index of the first error (tools have tool_index, need to map to step_index)
        # The tool node appears after the assistant step that called it
        first_error_tool_index = error_tools.iloc[0]["tool_index"]

        # Find the tool node in steps_df
        tool_nodes = steps_df[steps_df["node_type"] == "tool"].reset_index(drop=True)
        if first_error_tool_index < len(tool_nodes):
            first_error_step_index = tool_nodes.iloc[first_error_tool_index][
                "step_index"
            ]

            # Count assistant steps before this error tool step
            assistant_steps_before = steps_df[
                (steps_df["node_type"] == "assistant")
                & (steps_df["step_index"] < first_error_step_index)
            ]

            return (
                len(assistant_steps_before) + 1
                if len(assistant_steps_before) >= 0
                else -1
            )

        return -1
