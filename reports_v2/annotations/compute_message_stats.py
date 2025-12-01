"""
Script to compute message statistics per agent configuration.

For each agent (react or tool), model, env, level and verbosity, computes:
- Total number of messages (excluding tool responses)
- Average number of messages per task

Messages are those that are NOT tool responses. Tool responses are identified by:
- Having a specific schema with 'tool_name', 'arguments', 'result', 'status', etc.
- Or being in the 'tool' role with tool result content
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

# Base directory for reports
REPORTS_BASE = Path(__file__).parent.parent

# Models to process
MODELS = ["claude_sonnet_45", "gpt-4o"]

# Environments (task types)
ENVIRONMENTS = ["afm", "catalyst", "md", "ml", "resistor", "retrosynthesis", "spectra"]


def is_tool_response(message: dict[str, Any]) -> bool:
    """
    Check if a message is a tool response.

    Tool responses can be identified by:
    1. Role is 'tool' (ToolCallingAgent format)
    2. Content contains the tool result schema with 'tool_name', 'arguments', 'result', 'status'
    """
    # Check for ToolCallingAgent format (role = 'tool')
    if message.get("role") == "tool":
        return True

    # Check for ReActAgent format - tool results are in user messages as "Observation: {...}"
    # The content contains a dict with 'tool_name', 'arguments', 'result', 'status'
    content = message.get("content", "")
    if isinstance(content, str):
        # Check if it looks like a tool observation
        if content.startswith("Observation:"):
            return True
        # Also check for the pattern where tool results are embedded
        if (
            ("'tool_name':" in content or '"tool_name":' in content)
            and ("'status':" in content or '"status":' in content)
            and ("'result':" in content or '"result":' in content)
        ):
            return True

    # Check the 'name' field which indicates tool result in some formats
    return bool(message.get("name"))


def count_non_tool_messages(messages: list[dict[str, Any]]) -> int:
    """
    Count messages that are not tool responses.
    """
    count = 0
    for msg in messages:
        if not is_tool_response(msg):
            count += 1
    return count


def parse_agent_log_dir_name(dirname: str) -> tuple[str, str, str]:
    """
    Parse the agent log directory name to extract agent type, model, and verbosity.

    Example: agent_logs-ReActAgent-claude-sonnet-4-5-20250929-brief
    Returns: (agent_type, model, verbosity)
    """
    # Pattern: agent_logs-{AgentType}-{model}-{verbosity}
    match = re.match(r"agent_logs-(\w+Agent)-(.+)-(\w+)$", dirname)
    if match:
        agent_type = match.group(1)
        # The model part is between agent type and verbosity
        remaining = match.group(2)
        verbosity = match.group(3)

        # The model name is everything except the last part (which is verbosity)
        # For claude: claude-sonnet-4-5-20250929
        # For gpt4o: gpt-4o-2024-08-06
        model = remaining

        return agent_type, model, verbosity
    return None, None, None


def process_agent_logs(agent_logs_dir: Path) -> dict[str, Any]:
    """
    Process all JSON files in an agent_logs directory.

    Returns:
        Dict with statistics for this configuration
    """
    stats = {
        "total_messages": 0,
        "total_tasks": 0,
        "tasks": [],
        "messages_per_task": [],
    }

    if not agent_logs_dir.exists():
        return stats

    for json_file in agent_logs_dir.glob("*.json"):
        try:
            with json_file.open() as f:
                data = json.load(f)

            messages = data.get("messages", [])
            non_tool_count = count_non_tool_messages(messages)

            stats["total_messages"] += non_tool_count
            stats["total_tasks"] += 1
            stats["messages_per_task"].append(non_tool_count)
            stats["tasks"].append(
                {
                    "file": json_file.name,
                    "task_id": data.get("task_id", ""),
                    "messages": non_tool_count,
                    "tool_verbosity": data.get("tool_verbosity", ""),
                }
            )
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    return stats


def compute_all_stats() -> list[dict[str, Any]]:
    """
    Compute message statistics for all models, environments, levels, and agent configurations.
    Processes both 'tasks' and 'subtasks' directories.

    Returns:
        List of dicts, each containing all metadata and statistics for a configuration.
    """
    all_stats = []

    for model in MODELS:
        model_dir = REPORTS_BASE / model
        if not model_dir.exists():
            continue

        for env in ENVIRONMENTS:
            env_dir = model_dir / env
            if not env_dir.exists():
                continue

            # Find all levels
            for level_dir in sorted(env_dir.glob("level_*")):
                level = level_dir.name  # e.g., "level_1"

                # Process both tasks and subtasks directories
                for task_type in ["tasks", "subtasks"]:
                    is_subtask = task_type == "subtasks"
                    tasks_dir = level_dir / task_type

                    if not tasks_dir.exists():
                        continue

                    # Find all agent_logs directories
                    for agent_logs_dir in tasks_dir.glob("agent_logs-*"):
                        if not agent_logs_dir.is_dir():
                            continue

                        agent_type, agent_model, verbosity = parse_agent_log_dir_name(
                            agent_logs_dir.name
                        )
                        if not agent_type:
                            continue

                        stats = process_agent_logs(agent_logs_dir)

                        if stats["total_tasks"] > 0:
                            stats["average_messages_per_task"] = (
                                stats["total_messages"] / stats["total_tasks"]
                            )
                        else:
                            stats["average_messages_per_task"] = 0

                        # Create flat record with all metadata
                        record = {
                            "model": model,
                            "model_version": agent_model,
                            "environment": env,
                            "level": level,
                            "agent_type": agent_type,
                            "verbosity": verbosity,
                            "subtask": is_subtask,
                            "total_messages": stats["total_messages"],
                            "total_tasks": stats["total_tasks"],
                            "average_messages_per_task": stats[
                                "average_messages_per_task"
                            ],
                            "messages_per_task": stats["messages_per_task"],
                            "tasks": stats["tasks"],
                        }

                        all_stats.append(record)

    return all_stats


def create_summary(all_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create a summary across all configurations.
    """
    summary = {
        "by_model": defaultdict(lambda: {"total_messages": 0, "total_tasks": 0}),
        "by_agent_type": defaultdict(lambda: {"total_messages": 0, "total_tasks": 0}),
        "by_verbosity": defaultdict(lambda: {"total_messages": 0, "total_tasks": 0}),
        "by_environment": defaultdict(lambda: {"total_messages": 0, "total_tasks": 0}),
        "by_level": defaultdict(lambda: {"total_messages": 0, "total_tasks": 0}),
        "by_subtask": defaultdict(lambda: {"total_messages": 0, "total_tasks": 0}),
        "configurations": [],
    }

    for stats in all_stats:
        model = stats["model"]
        env = stats["environment"]
        level = stats["level"]

        # Add to summary
        summary["by_model"][model]["total_messages"] += stats["total_messages"]
        summary["by_model"][model]["total_tasks"] += stats["total_tasks"]

        summary["by_agent_type"][stats["agent_type"]]["total_messages"] += stats[
            "total_messages"
        ]
        summary["by_agent_type"][stats["agent_type"]]["total_tasks"] += stats[
            "total_tasks"
        ]

        summary["by_verbosity"][stats["verbosity"]]["total_messages"] += stats[
            "total_messages"
        ]
        summary["by_verbosity"][stats["verbosity"]]["total_tasks"] += stats[
            "total_tasks"
        ]

        summary["by_environment"][env]["total_messages"] += stats["total_messages"]
        summary["by_environment"][env]["total_tasks"] += stats["total_tasks"]

        summary["by_level"][level]["total_messages"] += stats["total_messages"]
        summary["by_level"][level]["total_tasks"] += stats["total_tasks"]

        # Add by subtask
        subtask_key = "subtask" if stats.get("subtask", False) else "task"
        summary["by_subtask"][subtask_key]["total_messages"] += stats["total_messages"]
        summary["by_subtask"][subtask_key]["total_tasks"] += stats["total_tasks"]

        # Add configuration (without the per-task details)
        summary["configurations"].append(
            {
                "model": model,
                "model_version": stats["model_version"],
                "agent_type": stats["agent_type"],
                "verbosity": stats["verbosity"],
                "environment": env,
                "level": level,
                "subtask": stats.get("subtask", False),
                "total_messages": stats["total_messages"],
                "total_tasks": stats["total_tasks"],
                "average_messages_per_task": stats["average_messages_per_task"],
            }
        )

    # Compute averages
    for key in [
        "by_model",
        "by_agent_type",
        "by_verbosity",
        "by_environment",
        "by_level",
        "by_subtask",
    ]:
        for data in summary[key].values():
            if data["total_tasks"] > 0:
                data["average_messages_per_task"] = (
                    data["total_messages"] / data["total_tasks"]
                )
            else:
                data["average_messages_per_task"] = 0

    # Convert defaultdicts to regular dicts
    summary["by_model"] = dict(summary["by_model"])
    summary["by_agent_type"] = dict(summary["by_agent_type"])
    summary["by_verbosity"] = dict(summary["by_verbosity"])
    summary["by_environment"] = dict(summary["by_environment"])
    summary["by_level"] = dict(summary["by_level"])
    summary["by_subtask"] = dict(summary["by_subtask"])

    return summary


def main():
    logger.info("Computing message statistics...")

    # Compute all stats
    all_stats = compute_all_stats()

    # Create summary
    summary = create_summary(all_stats)

    # Output directory
    output_dir = REPORTS_BASE / "annotations"
    output_dir.mkdir(exist_ok=True)

    # Data directory
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Save detailed stats
    detailed_output = output_dir / "message_stats_detailed.json"
    with detailed_output.open("w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Detailed stats saved to: {detailed_output}")

    # Save summary
    summary_output = output_dir / "message_stats_summary.json"
    with summary_output.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_output}")

    # Also save to data directory
    detailed_data_output = data_dir / "message_stats_detailed.json"
    with detailed_data_output.open("w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Detailed stats also saved to: {detailed_data_output}")

    summary_data_output = data_dir / "message_stats_summary.json"
    with summary_data_output.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary also saved to: {summary_data_output}")

    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("MESSAGE STATISTICS SUMMARY")
    logger.info("=" * 80)

    logger.info("\nBy Model:")
    for model, data in summary["by_model"].items():
        logger.info(
            f"  {model}: {data['total_messages']} messages, {data['total_tasks']} tasks, "
            f"avg {data['average_messages_per_task']:.2f} messages/task"
        )

    logger.info("\nBy Agent Type:")
    for agent, data in summary["by_agent_type"].items():
        logger.info(
            f"  {agent}: {data['total_messages']} messages, {data['total_tasks']} tasks, "
            f"avg {data['average_messages_per_task']:.2f} messages/task"
        )

    logger.info("\nBy Verbosity:")
    for verb, data in summary["by_verbosity"].items():
        logger.info(
            f"  {verb}: {data['total_messages']} messages, {data['total_tasks']} tasks, "
            f"avg {data['average_messages_per_task']:.2f} messages/task"
        )

    logger.info("\nBy Environment:")
    for env, data in summary["by_environment"].items():
        logger.info(
            f"  {env}: {data['total_messages']} messages, {data['total_tasks']} tasks, "
            f"avg {data['average_messages_per_task']:.2f} messages/task"
        )

    logger.info("\nBy Level:")
    for level, data in summary["by_level"].items():
        logger.info(
            f"  {level}: {data['total_messages']} messages, {data['total_tasks']} tasks, "
            f"avg {data['average_messages_per_task']:.2f} messages/task"
        )

    logger.info("\nBy Task/Subtask:")
    for task_type, data in summary["by_subtask"].items():
        logger.info(
            f"  {task_type}: {data['total_messages']} messages, {data['total_tasks']} tasks, "
            f"avg {data['average_messages_per_task']:.2f} messages/task"
        )


if __name__ == "__main__":
    main()
