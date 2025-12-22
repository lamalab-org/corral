ALL_MARKERS = [
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
]

ALL_TEMPORAL = [
    # Total node counts (all node types)
    "step_count_total",
    # Steps to first events (counting all nodes)
    "steps_to_first_tool_call",
    "steps_to_first_error",
    "steps_to_first_positive_marker",
    "steps_to_first_negative_marker",
    "steps_to_first_planning",
    "steps_to_first_reasoning",
    # Assistant-only step counts (counting only assistant nodes)
    "assistant_steps_to_first_tool_call",
    "assistant_steps_to_first_error",
    "assistant_steps_to_first_positive_marker",
    "assistant_steps_to_first_negative_marker",
    "assistant_steps_to_first_planning",
    "assistant_steps_to_first_reasoning",
    # Marker trajectory
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
    # Node type counts
    "assistant_steps_count",
    "tool_steps_count",
    "annotatable_steps_count",
    # First step features
    "has_planning_in_first_step",
    "has_reasoning_in_first_step",
    "has_tool_error_in_first_step",
    "longest_looping_sequence",
]

ALL_TEXT = [
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
]

ALL_TOOL = [
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
]

ALL_TOOL_DISTRIBUTION = [
    "tool_gini_coefficient",
    "tool_switching_rate",
    "max_tool_burst_length",
    "top3_tool_concentration",
    "rare_tool_ratio",
]
