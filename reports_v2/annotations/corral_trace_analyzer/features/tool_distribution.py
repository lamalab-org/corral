"""
Tool distribution features - advanced metrics for tool usage patterns

These features capture how tools are distributed across a trace:
- Inequality metrics (Gini coefficient)
- Temporal patterns (switching rate, burst length)
- Concentration metrics
"""

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseFeatureExtractor


class ToolDistributionFeatures(BaseFeatureExtractor):
    """Extract tool distribution and usage pattern features"""

    def __init__(
        self,
        features: list[str] | None = None,
        node_types: list[str] | None = None,
    ):
        """
        Initialize tool distribution feature extractor

        Args:
            features: list of features to extract or "all"
            node_types: list of node types to include
        """
        super().__init__(node_types=node_types)

        available_features = [
            "tool_gini_coefficient",
            "tool_switching_rate",
            "max_tool_burst_length",
            "top3_tool_concentration",
            "rare_tool_ratio",
        ]

        if features == "all" or features is None:
            self.features_to_extract = available_features
        else:
            self.features_to_extract = [f for f in features if f in available_features]

        self.feature_names = self.features_to_extract.copy()

    def _calculate_gini_coefficient(self, tool_counts: np.ndarray) -> float:
        """
        Calculate Gini coefficient for tool usage inequality

        0 = perfect equality (all tools used equally)
        1 = perfect inequality (one tool dominates)

        Args:
            tool_counts: Array of tool usage counts

        Returns:
            Gini coefficient
        """
        if len(tool_counts) == 0 or tool_counts.sum() == 0:
            return 0.0

        sorted_counts = np.sort(tool_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)

        return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (
            n + 1
        ) / n

    def _calculate_switching_rate(self, tool_sequence: list[str]) -> float:
        """
        Calculate how often the agent switches between different tools

        High switching rate = exploration
        Low switching rate = exploitation (using same tool repeatedly)

        Args:
            tool_sequence: Ordered list of tool names

        Returns:
            Switching rate (0 to 1)
        """
        if len(tool_sequence) <= 1:
            return 0.0

        switches = sum(
            1
            for i in range(1, len(tool_sequence))
            if tool_sequence[i] != tool_sequence[i - 1]
        )

        return switches / (len(tool_sequence) - 1)

    def _calculate_max_burst_length(self, tool_sequence: list[str]) -> int:
        """
        Calculate maximum consecutive same-tool calls

        High burst = agent sticks with one tool for a long time
        Low burst = agent switches frequently

        Args:
            tool_sequence: Ordered list of tool names

        Returns:
            Maximum burst length
        """
        if len(tool_sequence) == 0:
            return 0

        max_burst = 1
        current_burst = 1

        for i in range(1, len(tool_sequence)):
            if tool_sequence[i] == tool_sequence[i - 1]:
                current_burst += 1
                max_burst = max(max_burst, current_burst)
            else:
                current_burst = 1

        return max_burst

    def _calculate_top_k_concentration(
        self, tool_counts: np.ndarray, k: int = 3
    ) -> float:
        """
        Calculate what percentage of tool calls go to the top k tools

        High concentration = few tools dominate usage
        Low concentration = usage spread across many tools

        Args:
            tool_counts: Array of tool usage counts
            k: Number of top tools to consider

        Returns:
            Concentration ratio (0 to 1)
        """
        if len(tool_counts) == 0 or tool_counts.sum() == 0:
            return 0.0

        sorted_counts = np.sort(tool_counts)[::-1]
        top_k_sum = sorted_counts[: min(k, len(sorted_counts))].sum()

        return top_k_sum / tool_counts.sum()

    def _calculate_rare_tool_ratio(
        self, tool_counts: np.ndarray, threshold: int = 2
    ) -> float:
        """
        Calculate percentage of tools that are used rarely (≤ threshold times)

        High ratio = many tools tried but not reused (exploration)
        Low ratio = tools are reused (exploitation)

        Args:
            tool_counts: Array of tool usage counts
            threshold: Maximum count to be considered "rare"

        Returns:
            Ratio of rare tools (0 to 1)
        """
        if len(tool_counts) == 0:
            return 0.0

        rare_tools = (tool_counts <= threshold).sum()
        return rare_tools / len(tool_counts)

    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract tool distribution features from tools_df

        Args:
            traces_df: Trace-level dataframe
            steps_df: Step-level dataframe
            tools_df: Tool-level dataframe

        Returns:
            DataFrame with tool distribution features (indexed by trace_id)
        """
        # Filter by node type if specified
        logger.info(steps_df.head)
        filtered_tools_df = self.filter_steps_by_node_type(tools_df)

        results = []

        for trace_id in traces_df["trace_id"]:
            trace_tools = filtered_tools_df[filtered_tools_df["trace_id"] == trace_id]

            features = {"trace_id": trace_id}

            if len(trace_tools) == 0:
                # No tools called - set all features to 0/NaN
                for feature in self.features_to_extract:
                    features[feature] = 0.0
            else:
                # Get tool sequence (ordered by tool_index)
                tool_sequence = trace_tools.sort_values("tool_index")[
                    "tool_name"
                ].tolist()

                # Get tool counts
                tool_counts = trace_tools["tool_name"].value_counts().to_numpy()

                # Calculate requested features
                if "tool_gini_coefficient" in self.features_to_extract:
                    features["tool_gini_coefficient"] = (
                        self._calculate_gini_coefficient(tool_counts)
                    )

                if "tool_switching_rate" in self.features_to_extract:
                    features["tool_switching_rate"] = self._calculate_switching_rate(
                        tool_sequence
                    )

                if "max_tool_burst_length" in self.features_to_extract:
                    features["max_tool_burst_length"] = (
                        self._calculate_max_burst_length(tool_sequence)
                    )

                if "top3_tool_concentration" in self.features_to_extract:
                    features["top3_tool_concentration"] = (
                        self._calculate_top_k_concentration(tool_counts, k=3)
                    )

                if "rare_tool_ratio" in self.features_to_extract:
                    features["rare_tool_ratio"] = self._calculate_rare_tool_ratio(
                        tool_counts, threshold=2
                    )

            results.append(features)

        return pd.DataFrame(results).set_index("trace_id")
