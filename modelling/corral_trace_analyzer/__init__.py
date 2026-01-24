"""
CORRAL Trace Analyzer

A library for analyzing agent benchmark traces with manual annotations.
Provides data loading, feature engineering, correlation analysis, and visualization.
Includes both trace-level and environment-level analysis capabilities.
"""

__version__ = "0.1.0"

# Trace-level analysis
from .analysis.binary_outcome import BinaryOutcomeAnalyzer
from .analysis.correlations import CorrelationAnalyzer
from .analysis.environment_analysis import EnvironmentAnalyzer
from .analysis.interactions import InteractionAnalyzer

# Environment-level analysis
from .data.environment_loader import EnvironmentDataLoader
from .data.loader import TraceDataLoader
from .features.base import FeatureExtractor
from .visualization.correlations import CorrelationVisualizer

__all__ = [
    # Trace-level
    "TraceDataLoader",
    "FeatureExtractor",
    "CorrelationAnalyzer",
    "InteractionAnalyzer",
    "BinaryOutcomeAnalyzer",
    "CorrelationVisualizer",
    # Environment-level
    "EnvironmentDataLoader",
    "EnvironmentAnalyzer",
]
