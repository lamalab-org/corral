"""
Base class for feature extraction
"""

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""

    def __init__(self, node_types: list[str] | None = None):
        """
        Initialize feature extractor

        Args:
            node_types: List of node types to include (e.g., ['assistant'])
                       If None, includes all node types
        """
        self.feature_names: list[str] = []
        self.node_types = node_types

    @abstractmethod
    def extract(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract features from the dataframes

        Args:
            traces_df: Trace-level dataframe
            steps_df: Step-level dataframe
            tools_df: Tool-level dataframe

        Returns:
            DataFrame with extracted features (indexed by trace_id)
        """

    def filter_steps_by_node_type(
        self, steps_df: pd.DataFrame, node_types: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Filter steps dataframe by node types

        Args:
            steps_df: Step-level dataframe
            node_types: List of node types to include (e.g., ['assistant'])
                       If None, uses self.node_types. If that's also None, returns all.

        Returns:
            Filtered dataframe
        """
        types_to_filter = node_types if node_types is not None else self.node_types

        if types_to_filter is None:
            return steps_df

        return steps_df[steps_df["node_type"].isin(types_to_filter)]

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces"""
        return self.feature_names


class FeatureExtractor:
    """Main feature extractor that combines all feature extractors"""

    def __init__(self):
        from ..config import FEATURES
        from .markers import MarkerFeatures
        from .temporal import TemporalFeatures
        from .text import TextFeatures
        from .tool_quality import ToolQualityFeatures

        self.extractors = []

        extractor_map = {
            "temporal": TemporalFeatures,
            "tool_quality": ToolQualityFeatures,
            "markers": MarkerFeatures,
            "text": TextFeatures,
        }

        for name, klass in extractor_map.items():
            enabled_features = FEATURES.get(name)
            if enabled_features:  # Not None or empty list
                self.extractors.append(klass(features=enabled_features))

    def extract_all(
        self, traces_df: pd.DataFrame, steps_df: pd.DataFrame, tools_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract all features and combine into a single dataframe

        Args:
            traces_df: Trace-level dataframe
            steps_df: Step-level dataframe
            tools_df: Tool-level dataframe

        Returns:
            DataFrame with all extracted features
        """
        from ..config import EXCLUDE_FROM_ANALYSIS

        logger.info("Extracting all features...")

        # Start with a copy of traces_df, but exclude raw marker counts
        result_df = traces_df.set_index("trace_id").copy()

        # Remove columns that should be excluded from analysis
        cols_to_drop = [
            col for col in EXCLUDE_FROM_ANALYSIS if col in result_df.columns
        ]
        if cols_to_drop:
            logger.info(
                f"  Excluding {len(cols_to_drop)} raw count columns from analysis: {', '.join(cols_to_drop)}"
            )
            result_df = result_df.drop(columns=cols_to_drop)

        # Extract features from each extractor and merge
        for extractor in self.extractors:
            logger.info(f"  Extracting {extractor.__class__.__name__}...")
            features_df = extractor.extract(traces_df, steps_df, tools_df)

            # Merge features
            result_df = result_df.join(features_df, how="left")

        logger.info(f"Extracted {len(result_df.columns)} total features")
        return result_df.reset_index()

    def get_all_feature_names(self) -> dict[str, list[str]]:
        """Get all feature names grouped by extractor"""
        return {
            extractor.__class__.__name__: extractor.get_feature_names()
            for extractor in self.extractors
        }
