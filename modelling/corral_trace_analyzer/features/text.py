"""
Text and semantic features
"""

import re
from typing import Any

import numpy as np
import pandas as pd

from corral_trace_analyzer.config import (
    CONFIDENCE_MARKERS,
    SELF_CORRECTION_MARKERS,
    UNCERTAINTY_MARKERS,
    VERIFICATION_MARKERS,
)

from .base import BaseFeatureExtractor
from .constant import ALL_TEXT


class TextFeatures(BaseFeatureExtractor):
    """Extract text and semantic features"""

    ALL_FEATURES = ALL_TEXT

    def __init__(
        self, node_types: list | None = None, features: str | list[str] = "all"
    ):
        """
        Initialize text feature extractor

        Args:
            node_types: list of node types to include. Default is ['assistant']
                       since text analysis focuses on agent's messages.
            features: "all" or a list of feature names to extract.
        """
        # Default to assistant nodes only (agent's messages)
        if node_types is None:
            node_types = ["assistant"]
        super().__init__(node_types=node_types)

        if features == "all":
            self.feature_names = self.ALL_FEATURES
        else:
            self.feature_names = [f for f in self.ALL_FEATURES if f in features]

    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, _tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract text features"""

        # Filter steps by node type (default: assistant nodes only)
        filtered_steps_df = self.filter_steps_by_node_type(steps_df)

        features_list = []

        for trace_id in traces_df["trace_id"].unique():
            trace_steps = filtered_steps_df[filtered_steps_df["trace_id"] == trace_id]

            features = self._extract_for_trace(trace_id, trace_steps)
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
        self, trace_id: str, steps_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Extract features for a single trace"""

        features = {"trace_id": trace_id}

        if len(steps_df) == 0:
            # No steps - set defaults
            features.update(
                {
                    "avg_message_length": 0.0,
                    "avg_assistant_message_length": 0.0,
                    "message_length_std": 0.0,
                    "total_message_length": 0,
                    "uncertainty_marker_count": 0,
                    "uncertainty_marker_ratio": 0.0,
                    "confidence_marker_count": 0,
                    "confidence_marker_ratio": 0.0,
                    "verification_marker_count": 0,
                    "verification_marker_ratio": 0.0,
                    "self_correction_marker_count": 0,
                    "self_correction_marker_ratio": 0.0,
                    "has_thought_tags": False,
                    "thought_count": 0,
                    "avg_thought_length": 0.0,
                    "action_count": 0,
                    "thought_to_action_ratio": 0.0,
                    "avg_words_per_message": 0.0,
                    "code_block_count": 0,
                    "question_count": 0,
                }
            )
            return features

        # Message length statistics
        message_lengths = steps_df["message_length"].to_numpy()
        features["avg_message_length"] = np.mean(message_lengths)
        features["message_length_std"] = (
            np.std(message_lengths) if len(message_lengths) > 1 else 0.0
        )
        features["total_message_length"] = np.sum(message_lengths)

        # Assistant message length
        assistant_steps = steps_df[steps_df["node_type"] == "assistant"]
        if len(assistant_steps) > 0:
            features["avg_assistant_message_length"] = assistant_steps[
                "message_length"
            ].mean()
        else:
            features["avg_assistant_message_length"] = 0.0

        # Uncertainty markers
        features["uncertainty_marker_count"] = self._count_uncertainty_markers(steps_df)
        features["uncertainty_marker_ratio"] = (
            features["uncertainty_marker_count"] / len(assistant_steps)
            if len(assistant_steps) > 0
            else 0.0
        )

        # Confidence markers
        features["confidence_marker_count"] = self._count_confidence_markers(steps_df)
        features["confidence_marker_ratio"] = (
            features["confidence_marker_count"] / len(assistant_steps)
            if len(assistant_steps) > 0
            else 0.0
        )

        # Verification markers
        features["verification_marker_count"] = self._count_verification_markers(
            steps_df
        )
        features["verification_marker_ratio"] = (
            features["verification_marker_count"] / len(assistant_steps)
            if len(assistant_steps) > 0
            else 0.0
        )

        # Self-correction markers
        features["self_correction_marker_count"] = self._count_self_correction_markers(
            steps_df
        )
        features["self_correction_marker_ratio"] = (
            features["self_correction_marker_count"] / len(assistant_steps)
            if len(assistant_steps) > 0
            else 0.0
        )

        # Thought tags analysis (for ReAct-style agents)
        thought_info = self._analyze_thoughts_and_actions(assistant_steps)
        features["has_thought_tags"] = thought_info["has_thoughts"]
        features["thought_count"] = thought_info["thought_count"]
        features["avg_thought_length"] = thought_info["avg_thought_length"]
        features["action_count"] = thought_info["action_count"]
        features["thought_to_action_ratio"] = thought_info["thought_to_action_ratio"]

        # Word count
        total_words = sum(
            len(msg.split()) for msg in steps_df["message"] if isinstance(msg, str)
        )
        features["avg_words_per_message"] = total_words / len(steps_df)

        # Code blocks
        features["code_block_count"] = self._count_code_blocks(steps_df)

        # Questions
        features["question_count"] = self._count_questions(steps_df)

        return features

    def _count_uncertainty_markers(self, steps_df: pd.DataFrame) -> int:
        """Count uncertainty markers in messages"""
        count = 0

        for message in steps_df["message"]:
            if not isinstance(message, str):
                continue

            message_lower = message.lower()
            for marker in UNCERTAINTY_MARKERS:
                count += message_lower.count(marker.lower())

        return count

    def _count_confidence_markers(self, steps_df: pd.DataFrame) -> int:
        """Count confidence markers in messages"""
        count = 0

        for message in steps_df["message"]:
            if not isinstance(message, str):
                continue

            message_lower = message.lower()
            for marker in CONFIDENCE_MARKERS:
                count += message_lower.count(marker.lower())

        return count

    def _count_verification_markers(self, steps_df: pd.DataFrame) -> int:
        """Count verification/checking markers in messages"""
        count = 0

        for message in steps_df["message"]:
            if not isinstance(message, str):
                continue

            message_lower = message.lower()
            for marker in VERIFICATION_MARKERS:
                count += message_lower.count(marker.lower())

        return count

    def _count_self_correction_markers(self, steps_df: pd.DataFrame) -> int:
        """Count self-correction markers in messages"""
        count = 0

        for message in steps_df["message"]:
            if not isinstance(message, str):
                continue

            message_lower = message.lower()
            for marker in SELF_CORRECTION_MARKERS:
                count += message_lower.count(marker.lower())

        return count

    def _analyze_thoughts_and_actions(
        self, assistant_steps: pd.DataFrame
    ) -> dict[str, Any]:
        """Analyze thought and action tags in assistant messages"""

        result = {
            "has_thoughts": False,
            "thought_count": 0,
            "avg_thought_length": 0.0,
            "action_count": 0,
            "thought_to_action_ratio": 0.0,
        }

        if len(assistant_steps) == 0:
            return result

        thought_lengths = []

        for message in assistant_steps["message"]:
            if not isinstance(message, str):
                continue

            # Look for <thought> tags
            thoughts = re.findall(r"<thought>(.*?)</thought>", message, re.DOTALL)
            if thoughts:
                result["has_thoughts"] = True
                result["thought_count"] += len(thoughts)
                thought_lengths.extend([len(t.strip()) for t in thoughts])

            # Look for <action> tags
            actions = re.findall(r"<action>(.*?)</action>", message, re.DOTALL)
            result["action_count"] += len(actions)

        if thought_lengths:
            result["avg_thought_length"] = np.mean(thought_lengths)

        if result["action_count"] > 0:
            result["thought_to_action_ratio"] = (
                result["thought_count"] / result["action_count"]
            )

        return result

    def _count_code_blocks(self, steps_df: pd.DataFrame) -> int:
        """Count code blocks in messages"""
        count = 0

        for message in steps_df["message"]:
            if not isinstance(message, str):
                continue

            # Count markdown code blocks (```)
            count += message.count("```")

        return count // 2  # Divide by 2 since each block has opening and closing

    def _count_questions(self, steps_df: pd.DataFrame) -> int:
        """Count questions in messages"""
        count = 0

        for message in steps_df["message"]:
            if not isinstance(message, str):
                continue

            count += message.count("?")

        return count
