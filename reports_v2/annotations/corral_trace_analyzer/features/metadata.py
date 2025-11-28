"""
Metadata features - model, agent_type, level, environment

These are categorical/metadata features that describe the experimental setup.
For ML models, categorical features are one-hot encoded automatically during analysis.
"""

import pandas as pd
from loguru import logger

from .base import BaseFeatureExtractor


class MetadataFeatures(BaseFeatureExtractor):
    """Extract metadata features (model, agent_type, level, environment)"""

    def __init__(self, features: list[str] | None = None):
        """
        Initialize metadata feature extractor

        Args:
            features: list of features to extract. Options:
                     - "model": LLM model name (categorical)
                     - "agent_type": Agent type (categorical)
                     - "level": Difficulty level (numeric)
                     - "environment": Environment name (categorical)
                     Use "all" to extract all features.
        """
        super().__init__()

        available_features = ["model", "agent_type", "level", "environment"]

        if features == "all" or features is None:
            self.features_to_extract = available_features
        else:
            self.features_to_extract = [f for f in features if f in available_features]

        self.feature_names = self.features_to_extract.copy()

    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract metadata features from traces_df

        Args:
            traces_df: Trace-level dataframe
            steps_df: Step-level dataframe (not used)
            tools_df: Tool-level dataframe (not used)

        Returns:
            DataFrame with metadata features (indexed by trace_id)
        """
        result = pd.DataFrame(index=traces_df["trace_id"])

        # Extract requested features
        for feature in self.features_to_extract:
            if feature in traces_df.columns:
                result[feature] = traces_df.set_index("trace_id")[feature]
            else:
                # Feature not available, set to None
                result[feature] = None
                logger.warning(f"Warning: Feature '{feature}' not found in traces_df")

        return result

    def get_categorical_features(self) -> list[str]:
        """Return list of categorical feature names (for one-hot encoding)"""
        categorical = ["model", "agent_type", "environment"]
        return [f for f in self.features_to_extract if f in categorical]

    def get_numeric_features(self) -> list[str]:
        """Return list of numeric feature names"""
        numeric = ["level"]
        return [f for f in self.features_to_extract if f in numeric]
