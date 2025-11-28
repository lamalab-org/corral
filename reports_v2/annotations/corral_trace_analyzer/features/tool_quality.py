"""
Tool quality and usage pattern features
"""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor
from .constant import ALL_TOOL


class ToolQualityFeatures(BaseFeatureExtractor):
    """Extract tool quality and usage pattern features"""

    ALL_FEATURES = ALL_TOOL

    def __init__(
        self, node_types: list | None = None, features: str | list[str] = "all"
    ):
        """
        Initialize tool quality feature extractor

        Args:
            node_types: Not used for tool quality features since they use
                       the separate tools_df. Included for API consistency.
            features: "all" or a list of feature names to extract.
        """
        super().__init__(node_types=node_types)

        if features == "all":
            self.feature_names = self.ALL_FEATURES
        else:
            self.feature_names = [f for f in self.ALL_FEATURES if f in features]

    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract tool quality features"""

        features_list = []

        for trace_id in traces_df["trace_id"].unique():
            trace_tools = tools_df[tools_df["trace_id"] == trace_id]
            trace_steps = steps_df[steps_df["trace_id"] == trace_id]

            features = self._extract_for_trace(trace_id, trace_tools, trace_steps)
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
        self, trace_id: str, tools_df: pd.DataFrame, steps_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Extract features for a single trace"""

        features = {"trace_id": trace_id}

        if len(tools_df) == 0:
            # No tool calls - set defaults
            features.update(
                {
                    "retry_rate": 0.0,
                    "unique_tool_calls_ratio": 0.0,
                    "error_burstiness": 0,
                    "max_consecutive_failures": 0,
                    "max_consecutive_successes": 0,
                    "tool_diversity_entropy": 0.0,
                    "tool_latency_mean": 0.0,
                    "tool_latency_max": 0.0,
                    "tool_latency_std": 0.0,
                    "error_rate": 0.0,
                    "success_rate": 0.0,
                    "avg_tools_per_step": 0.0,
                    "tool_repetition_rate": 0.0,
                    "most_used_tool": None,
                    "most_used_tool_count": 0,
                    "most_used_tool_ratio": 0.0,
                }
            )
            return features

        # Basic rates
        total_calls = len(tools_df)
        successful_calls = tools_df["is_success"].sum()
        failed_calls = total_calls - successful_calls

        features["error_rate"] = failed_calls / total_calls
        features["success_rate"] = successful_calls / total_calls

        # Unique tool calls ratio
        unique_tools = tools_df["tool_name"].nunique()
        features["unique_tool_calls_ratio"] = unique_tools / total_calls

        # Retry rate (same tool called consecutively)
        tool_names = tools_df["tool_name"].to_numpy()
        consecutive_same = sum(
            1 for i in range(1, len(tool_names)) if tool_names[i] == tool_names[i - 1]
        )
        features["retry_rate"] = (
            consecutive_same / total_calls if total_calls > 1 else 0.0
        )

        # Tool repetition rate (any repeated tool, not necessarily consecutive)
        tool_counts = Counter(tool_names)
        repeated_calls = sum(count - 1 for count in tool_counts.values() if count > 1)
        features["tool_repetition_rate"] = repeated_calls / total_calls

        # Error burstiness (max consecutive failures)
        features["max_consecutive_failures"] = self._max_consecutive(
            tools_df["is_success"].values, False
        )
        features["max_consecutive_successes"] = self._max_consecutive(
            tools_df["is_success"].values, True
        )
        features["error_burstiness"] = features["max_consecutive_failures"]

        # Tool diversity entropy
        features["tool_diversity_entropy"] = self._calculate_entropy(tool_names)

        # Tool latency statistics
        latencies = tools_df["duration"].to_numpy()
        features["tool_latency_mean"] = np.mean(latencies)
        features["tool_latency_max"] = np.max(latencies)
        features["tool_latency_std"] = np.std(latencies) if len(latencies) > 1 else 0.0

        # Tools per step
        _tool_steps = len(steps_df[steps_df["node_type"] == "tool"])
        features["avg_tools_per_step"] = (
            total_calls / len(steps_df) if len(steps_df) > 0 else 0.0
        )

        # Most used tool
        most_common = tool_counts.most_common(1)
        if most_common:
            features["most_used_tool"] = most_common[0][0]
            features["most_used_tool_count"] = most_common[0][1]
            features["most_used_tool_ratio"] = most_common[0][1] / total_calls
        else:
            features["most_used_tool"] = None
            features["most_used_tool_count"] = 0
            features["most_used_tool_ratio"] = 0.0

        return features

    def _max_consecutive(self, sequence, value) -> int:
        """Calculate maximum consecutive occurrences of a value"""
        max_count = 0
        current_count = 0

        for item in sequence:
            if item == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _calculate_entropy(self, sequence) -> float:
        """Calculate Shannon entropy of a sequence"""
        counts = Counter(sequence)
        total = len(sequence)

        entropy = 0.0
        for count in counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)

        return entropy
