ALLOWED_MODELS = {
    "claude_sonnet_45",
    "gpt_4o",
    "gpt_oss_120b",
}

LEVEL_PREFIX = "level_"

ALLOWED_CATEGORIES = {
    "tasks",
    "subtasks",
}

AGENTS = {
    "react": "ReActAgent",
    "reactagent": "ReActAgent",
    "tool calling": "ToolCallingAgent",
    "toolcalling": "ToolCallingAgent",
    "toolcallingagent": "ToolCallingAgent",
}

VERBOSITY = {"brief", "comprehensive", "workflow"}

METRIC_KEYS = [
    "success_rate",
    "average_score",
    "pass@1",
    "pass@2",
    "pass@3",
    "pass@4",
    "pass@5",
    "pass^1",
    "pass^2",
    "pass^3",
    "pass^4",
    "pass^5",
]
