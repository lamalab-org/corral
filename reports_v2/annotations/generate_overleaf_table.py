"""Generate an Overleaf LaTeX table with main results from the benchmark.

This script reads the main results from JSON and Q&A summary files,
and generates a LaTeX table for the paper.

Averaging of duplicate entries:
    When multiple results share the same (environment, level, agent_type, verbosity, model)
    combination (e.g., multiple MD runs with identical configurations), their numeric
    metrics are averaged. The averaging is performed by summing all values for each
    metric and dividing by the count of matching entries. This applies to:
    - average_score (Overall)
    - pass@5
    - pass^5
    - total_tool_calls
    - failed_tool_calls
    - total_benchmark_time
"""

import contextlib
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

# Root paths
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = Path(__file__).parent / "data"
QA_PATH = ROOT_PATH / "qa"

# Global message stats lookup dictionaries
_MESSAGE_STATS_LOOKUP: dict | None = None
_MESSAGE_STATS_SUBTASK_LOOKUP: dict | None = None


def _load_message_stats(subtask: bool = False) -> dict:
    """Load message stats and build a lookup dictionary.

    Args:
        subtask: If True, load subtask message stats; otherwise load regular task stats.

    Returns:
        Dictionary with (model, env, level, agent_type, verbosity) -> total_messages mapping.
    """
    global _MESSAGE_STATS_LOOKUP, _MESSAGE_STATS_SUBTASK_LOOKUP  # noqa: PLW0603

    # Return cached lookup if available
    if subtask:
        if _MESSAGE_STATS_SUBTASK_LOOKUP is not None:
            return _MESSAGE_STATS_SUBTASK_LOOKUP
    else:
        if _MESSAGE_STATS_LOOKUP is not None:
            return _MESSAGE_STATS_LOOKUP

    message_stats_path = DATA_PATH / "message_stats_detailed.json"
    lookup = {}

    if not message_stats_path.exists():
        logger.warning(f"Message stats file not found: {message_stats_path}")
        if subtask:
            _MESSAGE_STATS_SUBTASK_LOOKUP = lookup
        else:
            _MESSAGE_STATS_LOOKUP = lookup
        return lookup

    with message_stats_path.open() as f:
        stats = json.load(f)

    # Build lookup dictionary
    # Map from message_stats agent_type to main_results agent_type
    agent_type_map = {
        "ReActAgent": "react",
        "ToolCallingAgent": "tool",
    }

    for entry in stats:
        # Filter by subtask flag
        is_subtask = entry.get("subtask", False)
        if is_subtask != subtask:
            continue

        agent_type = agent_type_map.get(
            entry.get("agent_type"), entry.get("agent_type")
        )
        key = (
            entry.get("model"),
            entry.get("environment"),
            entry.get("level"),
            agent_type,
            entry.get("verbosity"),
        )
        lookup[key] = entry.get("total_messages", 0)

    if subtask:
        _MESSAGE_STATS_SUBTASK_LOOKUP = lookup
    else:
        _MESSAGE_STATS_LOOKUP = lookup

    logger.info(
        f"Loaded {len(lookup)} {'subtask ' if subtask else ''}message stats entries"
    )
    return lookup


def get_message_count(
    model: str,
    env: str,
    level: int,
    agent_type: str,
    verbosity: str,
    subtask: bool = False,
) -> int | None:
    """Get total message count for a given configuration.

    Args:
        model: Model name (e.g., 'claude_sonnet_45', 'gpt-4o')
        env: Environment name (e.g., 'afm', 'catalyst')
        level: Level number (1, 2, etc.)
        agent_type: Agent type ('react' or 'tool')
        verbosity: Verbosity level ('brief', 'comprehensive', 'workflow')
        subtask: If True, get message count for subtasks; otherwise for regular tasks.

    Returns:
        Total message count or None if not found.
    """
    lookup = _load_message_stats(subtask=subtask)
    level_str = f"level_{level}"
    key = (model, env, level_str, agent_type, verbosity)
    return lookup.get(key)


def get_qa_score(env: str, model: str) -> float | None:
    """Get the Q&A overall score for a given environment and model.

    Args:
        env: Environment name (e.g., 'afm', 'catalyst', 'md', 'ml', 'resistor', 'retrosynthesis', 'spectra')
        model: Model name ('claude_sonnet_45' or 'gpt-4o')

    Returns:
        The overall Q&A score or None if not found.
    """
    # Map environment names to QA folder names
    env_to_qa_folder = {
        "afm": "afm_qa",
        "catalyst": "catalyst_qa",
        "md": "md_qa",
        "ml": "ml_qa",
        "resistor": "resistor_qa",
        "retrosynthesis": "retro_qa",
        "spectra": "spectra_qa",
    }

    # Map model names to QA subfolder names
    model_to_qa_folder = {
        "claude_sonnet_45": "claude",
        "gpt-4o": "gpt",
    }

    qa_folder = env_to_qa_folder.get(env)
    model_folder = model_to_qa_folder.get(model)

    if not qa_folder or not model_folder:
        return None

    summary_path = (
        QA_PATH
        / qa_folder
        / model_folder
        / "reports"
        / "topic_reports"
        / "summary.json"
    )

    if not summary_path.exists():
        return None

    try:
        with summary_path.open() as f:
            data = json.load(f)
        return data.get("overall_score")
    except (json.JSONDecodeError, KeyError):
        return None


def format_value(value, is_percentage: bool = False, decimals: int = 2) -> str:
    """Format a value for the LaTeX table.

    Args:
        value: The value to format
        is_percentage: If True, multiply by 100 and add % sign
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if value == "N/A" or value is None:
        return "---"

    try:
        val = float(value)
        if is_percentage:
            return f"{val * 100:.{decimals}f}"
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_time(seconds) -> str:
    """Format time in a human-readable way.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m" or "45m" or "30s")
    """
    if seconds == "N/A" or seconds is None:
        return "---"

    try:
        secs = float(seconds)
        if secs >= 3600:
            hours = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            return f"{hours}h{mins}m"
        elif secs >= 60:
            mins = int(secs // 60)
            return f"{mins}m"
        else:
            return f"{int(secs)}s"
    except (ValueError, TypeError):
        return str(seconds)


def get_model_display_name(model: str) -> str:
    """Get display name for a model."""
    if "claude" in model.lower():
        return "Claude"
    elif "gpt" in model.lower():
        return "GPT-4o"
    return model


def get_agent_display_name(agent_type: str, verbosity: str) -> str:
    """Get display name for an agent type with verbosity."""
    agent_name = (
        "ReAct"
        if agent_type == "react"
        else "Tool"
        if agent_type == "tool"
        else agent_type
    )
    verbosity_short = get_verbosity_short(verbosity)
    return f"{agent_name}-{verbosity_short}" if verbosity_short else agent_name


def get_verbosity_short(verbosity: str) -> str:
    """Get short form of verbosity."""
    if not verbosity or verbosity == "N/A":
        return ""
    mapping = {
        "brief": "B",
        "comprehensive": "C",
        "workflow": "W",
    }
    return mapping.get(verbosity.lower(), verbosity[0].upper())


def _average_duplicate_entries(main_results: list) -> list:
    """Average entries with the same (environment, level, agent_type, verbosity, model).

    When multiple entries share the same configuration key, their numeric metrics
    are averaged by summing all values and dividing by the count of matching entries.

    Args:
        main_results: List of result dictionaries

    Returns:
        List of result dictionaries with duplicates averaged
    """
    # Group entries by key
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for result in main_results:
        key = (
            result.get("environment", ""),
            result.get("level", 0),
            result.get("agent_type", ""),
            result.get("tool_verbosity", ""),
            result.get("model", ""),
        )
        grouped[key].append(result)

    # Metrics to average
    numeric_fields = [
        "average_score",
        "pass@5",
        "pass^5",
        "total_tool_calls",
        "failed_tool_calls",
        "total_benchmark_time",
    ]

    averaged_results = []
    for key, entries in grouped.items():
        if len(entries) == 1:
            averaged_results.append(entries[0])
        else:
            # Average numeric fields
            averaged_entry = entries[0].copy()
            count = len(entries)

            for field in numeric_fields:
                values = []
                for entry in entries:
                    val = entry.get(field)
                    if val is not None and val != "N/A":
                        with contextlib.suppress(ValueError, TypeError):
                            values.append(float(val))
                if values:
                    averaged_entry[field] = sum(values) / count

            logger.debug(
                f"Averaged {count} entries for {key[0]}_{key[1]} {key[2]}-{key[3]} {key[4]}"
            )
            averaged_results.append(averaged_entry)

    return averaged_results


def generate_latex_table(main_results: list, subtask: bool = False) -> str:
    """Generate the LaTeX table from main results.

    Args:
        main_results: List of result dictionaries
        subtask: If True, generate table for subtask results; otherwise for regular tasks.

    Returns:
        LaTeX table string
    """
    # Table header (exactly as provided by user, with Q&A added at the end, without Thoughts)
    caption = (
        "Performance comparison of agents across different subtasks"
        if subtask
        else "Performance comparison of agents across different tasks"
    )
    label = "tab:subtask-results" if subtask else "tab:main-results"
    table = rf"""\begin{{table}}[!ht]
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \resizebox{{\textwidth}}{{!}}{{%
    \begin{{tabular}}{{lllcccccccc}}
        \toprule
        \multirow{{2}}{{*}}{{\begin{{tabular}}[c]{{@{{}}l@{{}}}}Task/\\Env\end{{tabular}}}} & \multirow{{2}}{{*}}{{Agent}} & \multirow{{2}}{{*}}{{Model}} & \multicolumn{{3}}{{c}}{{Scores}} & \multirow{{2}}{{*}}{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Num\\msgs\end{{tabular}}}} & \multirow{{2}}{{*}}{{\begin{{tabular}}[c]{{@{{}}c@{{}}}}Exec\\time\end{{tabular}}}} & \multicolumn{{2}}{{c}}{{Tool calls}} & \multirow{{2}}{{*}}{{Q\&A}} \\
        \cline{{4-6}} \cline{{9-10}}
        & & & Overall & pass@5 & pass\textasciicircum{{}}5 & & & Total & Failed & \\
        \midrule
        \midrule
"""

    # Average entries with same (environment, level, agent_type, verbosity, model)
    averaged_results = _average_duplicate_entries(main_results)

    # Sort results by environment, level, model, agent_type, verbosity
    sorted_results = sorted(
        averaged_results,
        key=lambda x: (
            x.get("environment", ""),
            x.get("level", 0),
            x.get("model", ""),
            x.get("agent_type", ""),
            x.get("tool_verbosity", ""),
        ),
    )

    # Group by environment_level
    current_env_level = None

    for result in sorted_results:
        env = result.get("environment", "N/A")
        level = result.get("level", "N/A")
        env_level = f"{env}-{level}"

        model = result.get("model", "N/A")
        agent_type = result.get("agent_type", "N/A")
        verbosity = result.get("tool_verbosity", "N/A")

        # Get values
        overall_score = result.get("average_score", "N/A")
        pass_at_5 = result.get("pass@5", "N/A")
        pass_exp_5 = result.get("pass^5", "N/A")
        total_tool_calls = result.get("total_tool_calls", "N/A")
        failed_tool_calls = result.get("failed_tool_calls", "N/A")
        exec_time = result.get("total_benchmark_time", "N/A")

        # Get Q&A score
        qa_score = get_qa_score(env, model)

        # Format values
        overall_str = format_value(overall_score, is_percentage=True)
        pass_at_5_str = format_value(pass_at_5, is_percentage=True)
        pass_exp_5_str = format_value(pass_exp_5, is_percentage=True)
        exec_time_str = format_time(exec_time)
        qa_str = (
            format_value(qa_score, is_percentage=True)
            if qa_score is not None
            else "---"
        )

        # Format tool calls (round to int if float from averaging)
        total_tool_calls_str = (
            str(round(total_tool_calls))
            if isinstance(total_tool_calls, (int | float))
            else str(total_tool_calls)
        )
        failed_tool_calls_str = (
            str(round(failed_tool_calls))
            if isinstance(failed_tool_calls, (int | float))
            else str(failed_tool_calls)
        )

        # Model and agent display names
        model_display = get_model_display_name(model)
        agent_display = get_agent_display_name(agent_type, verbosity)

        # Format env_level display (only show if different from previous)
        if env_level != current_env_level:
            env_display = env_level
            current_env_level = env_level
        else:
            env_display = ""

        # Num messages - lookup from message stats
        msg_count = get_message_count(
            model, env, level, agent_type, verbosity, subtask=subtask
        )
        num_msgs = str(msg_count) if msg_count is not None else "---"

        # Add row
        row = (
            f"        {env_display} & {agent_display} & {model_display} & "
            f"{overall_str} & {pass_at_5_str} & {pass_exp_5_str} & {num_msgs} & "
            f"{exec_time_str} & {total_tool_calls_str} & {failed_tool_calls_str} & {qa_str} \\\\\n"
        )
        table += row

    # Table footer
    table += r"""        \bottomrule
    \end{tabular}}
\end{table}
"""

    return table


def main():
    """Main function to generate the LaTeX tables for both tasks and subtasks."""
    # Load and generate main results table
    main_results_path = DATA_PATH / "main_results.json"

    if not main_results_path.exists():
        logger.error(f"Error: {main_results_path} does not exist")
        logger.info(
            "Please run collect_all_tasks_info.py first to generate the main results."
        )
    else:
        with main_results_path.open() as f:
            main_results = json.load(f)

        logger.info(f"Loaded {len(main_results)} results from {main_results_path}")

        # Generate LaTeX table
        latex_table = generate_latex_table(main_results, subtask=False)

        # Save to file
        output_path = DATA_PATH / "main_results_table.tex"
        with output_path.open("w") as f:
            f.write(latex_table)

        logger.info(f"LaTeX table saved to: {output_path}")
        logger.info("Table preview:")
        logger.info("-" * 80)
        logger.info(latex_table)

    # Load and generate subtask results table
    subtask_results_path = DATA_PATH / "main_results_subtasks.json"

    if not subtask_results_path.exists():
        logger.warning(f"Subtask results file not found: {subtask_results_path}")
        logger.info("Run collect_all_tasks_info.py to generate subtask results.")
    else:
        with subtask_results_path.open() as f:
            subtask_results = json.load(f)

        logger.info(
            f"Loaded {len(subtask_results)} subtask results from {subtask_results_path}"
        )

        # Generate LaTeX table for subtasks
        subtask_latex_table = generate_latex_table(subtask_results, subtask=True)

        # Save to file
        subtask_output_path = DATA_PATH / "main_results_subtasks_table.tex"
        with subtask_output_path.open("w") as f:
            f.write(subtask_latex_table)

        logger.info(f"Subtask LaTeX table saved to: {subtask_output_path}")
        logger.info("Subtask table preview:")
        logger.info("-" * 80)
        logger.info(subtask_latex_table)


if __name__ == "__main__":
    main()
