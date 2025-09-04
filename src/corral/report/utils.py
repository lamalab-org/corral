from corral.report.results import TaskTrialResult


def calculate_trial_tool_duration(trial: TaskTrialResult) -> float:
    """Calculate tool execution duration for a trial"""
    if "tool_calls" not in trial.tool_statistics:
        return 0.0

    duration = 0.0
    for tool_call in trial.tool_statistics["tool_calls"]:
        if tool_call.get("duration") is not None:
            duration += tool_call["duration"]
    return duration
