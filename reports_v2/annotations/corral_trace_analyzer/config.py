"""
Configuration constants for the trace analyzer
"""

# Analysis configuration
# Set to None to include all environments, or specify a list of environments to include
SELECTED_ENVIRONMENTS = (
    None  # Example: ["md", "retrosynthesis", "resistor", "afm", "catalyst", "ml"]
)

# Set to None to include all models, or specify a list
SELECTED_MODELS = None  # Example: ["gpt-4o"]

# Set to None to include all agent types, or specify a list
SELECTED_AGENT_TYPES = None  # Example: ["ReActAgent", "ToolCallingAgent"]

# Score type - affects which analyses to run
# "binary" for 0/1 outcomes, "continuous" for real-valued scores
SCORE_TYPE = "binary"  # Change to "continuous" if your scores are not 0/1

# ==========================================
# ENVIRONMENT-LEVEL ANALYSIS CONFIGURATION
# ==========================================

# Tool verbosity levels to include in environment-level analysis
# Set to None to include all, or specify a list: ["workflow", "brief", "comprehensive"]
SELECTED_VERBOSITY = None  # or ["workflow"]

# Which score metric to use as target variable for environment-level analysis
# Options: "average_score", "overall_success_rate", "pass@1", "pass@2", "pass@3", "pass@4", "pass@5"
#          "pass^1", "pass^2", "pass^3", "pass^4", "pass^5"
ENVIRONMENT_TARGET_METRIC = "average_score"

# Additional metrics to analyze (will compute correlations for all of these)
ENVIRONMENT_METRICS_TO_ANALYZE = [
    "average_score",
    "overall_success_rate",
    "pass@1",
    "pass@5",
]


# Node types
NODE_TYPES = None  # ["system", "user", "assistant", "tool"]

# Error types
ERROR_TYPES = ["invalid_tool", "invalid_args", "execution_error"]

# Text markers for analysis
# Uncertainty markers
UNCERTAINTY_MARKERS = [
    "maybe",
    "perhaps",
    "possibly",
    "might",
    "could",
    "I think",
    "I believe",
    "not sure",
    "uncertain",
    "unclear",
    "I dont know",
]

# Confidence markers (high certainty)
CONFIDENCE_MARKERS = [
    "definitely",
    "certainly",
    "obviously",
    "clearly",
    "sure that",
    "confident",
    "know that",
    "must be",
    "will",
    "without doubt",
    "absolutely",
    "undoubtedly",
    "guaranteed",
    "for sure",
    "no question",
    "indeed",
]

# Verification/checking markers
VERIFICATION_MARKERS = [
    "let me check",
    "verify",
    "confirm",
    "validate",
    "double-check",
    "make sure",
    "ensure",
    "test whether",
    "check if",
    "let me see",
    "need to verify",
    "checking",
    "examine",
    "inspect",
    "review",
    "cross-check",
]

# Self-correction markers
SELF_CORRECTION_MARKERS = [
    "wait",
    "actually",
    "correction",
    "mistake",
    "oops",
    "wrong",
    "error",
    "my bad",
    "apologies",
    "sorry",
    "let me fix",
    "let me correct",
    "instead",
    "rather",
    "on second thought",
    "let me try again",
    "re-do",
]

# Window sizes for temporal analysis
TEMPORAL_WINDOWS = [
    (1, 5),
    (6, 10),
    (11, 20),
    (21, float("inf")),
]

# Correlation significance level
ALPHA = 0.05

# Visualization settings
VIZ_CONFIG = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "whitegrid",
    "palette": "husl",
    "context": "notebook",
}

# ==========================================
# DATA FILTERING CONFIGURATION
# ==========================================
# Columns to exclude from feature analysis
# These are raw marker counts used to derive features, not features themselves
EXCLUDE_FROM_ANALYSIS = [
    # Raw marker counts (features are derived from these)
    # "positive_marker_count",
    # "negative_marker_count",
    # "neutral_marker_count",
    # "validation_attempt_count",
    # "backtrack_trigger_count",
    # "planning_statement_count",
    # "reasoning_statement_count",
    "correct_submission_count",
    # "neutral_count",
    "iteration_limit_count",
    # "missing_validation_count",
    # "unnecessary_tool_use_count",
    # "non_sense_count",
    # "loop_instance_count",
    # "hallucination_count",
    # "wrong_planning_count",
    # "wrong_reasoning_count",
    # "syntax_error_count",
    # "early_final_answer_count",
    # "give_up_count",
    # "inefficient_tool_call_count",
    # "misunderstood_tool_count",
    # "qa_score",(passed with metadata)
]

# ==========================================
# FEATURE EXTRACTION CONFIGURATION
# ==========================================
# For each feature extractor, specify which features to enable.
# Use 'all' to enable all features from an extractor.
# To disable an extractor, provide an empty list for its features.
# FEATURES = {
#     "markers": "all",
#     "temporal": "all",
#     "text": "all",
#     "tool_quality": "all",
# }
# Example for selecting specific features:
FEATURES = {
    "markers": [
        "total_markers",
        "marker_balance",
        "marker_balance_ratio",
        "positive_marker_ratio",
        "negative_marker_ratio",
        "neutral_marker_ratio",
        "validation_to_total_ratio",
        "backtrack_to_total_ratio",
        "planning_to_total_ratio",
        "reasoning_to_total_ratio",
        "validation_after_error_rate",
        "backtrack_after_error_rate",
        "problem_marker_count",
        "problem_marker_ratio",
        "recovery_marker_count",
        "recovery_marker_ratio",
    ],
    "temporal": [
        # Basic counts
        #      "step_count_total",  # Total nodes (all types)
        "assistant_steps_count",  # Assistant nodes only
        # "tool_steps_count",  # Tool nodes only
        "annotatable_steps_count",  # Annotatable nodes only
        # Steps to first events - ALL NODES counting (includes system, user, assistant, tool)
        "steps_to_first_tool_call",
        "steps_to_first_error",
        "steps_to_first_positive_marker",
        "steps_to_first_negative_marker",
        "steps_to_first_planning",
        "steps_to_first_reasoning",
        # Steps to first events - ASSISTANT NODES ONLY counting
        "assistant_steps_to_first_tool_call",
        "assistant_steps_to_first_error",
        "assistant_steps_to_first_positive_marker",
        "assistant_steps_to_first_negative_marker",
        "assistant_steps_to_first_planning",
        "assistant_steps_to_first_reasoning",
        # Marker trajectory analysis
        "marker_trajectory_slope",
        "early_negative_rate",
        # Windowed marker counts
        "positive_markers_steps_1_5",
        "negative_markers_steps_1_5",
        "positive_markers_steps_6_10",
        "negative_markers_steps_6_10",
        "positive_markers_steps_11_20",
        "negative_markers_steps_11_20",
        "positive_markers_steps_21_plus",
        "negative_markers_steps_21_plus",
        # First-step binary features
        "has_planning_in_first_step",
        "has_reasoning_in_first_step",
        "has_tool_error_in_first_step",
        # Loop detection
        "longest_looping_sequence",
    ],
    "text": [
        "avg_message_length",
        "avg_assistant_message_length",
        "message_length_std",
        "total_message_length",
        "uncertainty_marker_count",
        "uncertainty_marker_ratio",
        "confidence_marker_count",
        "confidence_marker_ratio",
        "verification_marker_count",
        "verification_marker_ratio",
        "self_correction_marker_count",
        "self_correction_marker_ratio",
        "has_thought_tags",
        "thought_count",
        "avg_thought_length",
        "action_count",
        "thought_to_action_ratio",
        "avg_words_per_message",
        "code_block_count",
        "question_count",
    ],
    "tool_quality": [
        "retry_rate",
        "unique_tool_calls_ratio",
        "error_burstiness",
        "max_consecutive_failures",
        "max_consecutive_successes",
        "tool_diversity_entropy",
        "tool_latency_mean",
        "tool_latency_max",
        "tool_latency_std",
        "error_rate",
        "success_rate",
        "avg_tools_per_step",
        "tool_repetition_rate",
        "most_used_tool",
        "most_used_tool_count",
        "most_used_tool_ratio",
    ],
    "tool_distribution": [
        "tool_gini_coefficient",
        "tool_switching_rate",
        "max_tool_burst_length",
        "top3_tool_concentration",
        "rare_tool_ratio",
    ],
}
