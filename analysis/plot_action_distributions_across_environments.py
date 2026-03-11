"""Summarize how benchmark environments distribute agent actions across tool families.

The script reconstructs tool usage from nested trial traces, collapses the raw
tool vocabulary into a small set of analysis categories, and then compares the
mean category mix across environments. The grouped plots are intended to expose
whether environment differences persist after averaging over verbosity settings,
agent scaffolds, and models.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from contextlib import suppress
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "fig_4_app"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE_AGENT = OUT_DIR / "action_distribution_by_environment_and_agent.pdf"
OUT_FILE_MODEL = OUT_DIR / "action_distribution_by_environment_and_model.pdf"

ENV_LABELS = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retro",
    "spectra": "Spectra",
}
MODEL_LABELS = {
    "claude-4.5": "Claude 4.5",
    "gpt-4o": "GPT-4o",
    "gpt-oss-120b": "GPT-OSS-120B",
}
AGENT_TYPE_LABELS = {
    "react": "ReAct",
    "tool_calling": "Tool Calling",
}

ACTION_ORDER = [
    "retrieval",
    "file operations",
    "code execution",
    "experiment execution",
    "validation",
]
ACTION_LABELS = {
    "retrieval": "Retrieval",
    "file operations": "File Operations",
    "code execution": "Code Execution",
    "experiment execution": "Experiment Execution",
    "validation": "Validation",
}
ACTION_COLORS = {
    "retrieval": "#4C72B0",
    "file operations": "#8C8C8C",
    "code execution": "#55A868",
    "experiment execution": "#64B5CD",
    "validation": "#C44E52",
}

FILE_OPERATION_TOOLS = {
    "read_file",
    "write_file",
    "list_files",
    "file_info",
    "cat_files",
    "copy_file",
    "move_file",
    "mkdir",
    "grep",
    "file_write",
}
CODE_EXECUTION_TOOLS = {
    "execute_python_code",
    "execute_python_script",
    "code_executor",
    "exec",
    "exec_python",
    "exec_code",
    "run_in_terminal",
    "obtain_isomers_from_molecular_formula",
    "return_possible_fragments",
    "propose_simple_topology",
    "estimate_resistor_values",
    "choose_slab_text",
    "choose_adsorption_site_text",
    "sort_and_get_first_from_json",
    "select_polymorphs_with_strategy",
    "select_polymorphs_with_strategy_to_file",
    "finalize",
    "convert_structure_to_lammps_data",
    "prepare_tabular_dataset",
    "filter_json_with_strategy",
    "consolidate_polymorph_datasets",
    "create_slab_from_structure_text",
    "enumerate_slabs_text",
    "add_adsorbate_to_slab_text",
    "generate_reconstructed_slab",
    "find_all_unique_slabs_upto_millerindex",
    "generate_adsorbate_slab_configs",
    "apply_template",
    "deprotect_molecule",
    "detect_protection_groups",
    "detect_functional_groups",
    "map_reaction_smiles",
    "delta_to_wye_transform",
    "wye_to_delta_transform",
    "combine_subnetworks",
    "merge_subnetworks",
    "combine_subnetworks_manually",
    "merge_subnetworks_manually",
}
VALIDATION_TOOLS = {
    "analyze_measurements",
    "validate_smiles",
    "validate_measurements",
    "verify_step",
    "verify_route",
    "check_smiles_reaction_template_matching",
    "evaluate_xgboost_model",
}
EXPERIMENT_TOOLS = {
    "image_analyzer",
    "calculate_parallel_resistance",
    "calculate_series_resistance",
    "run_lammps",
    "train_xgboost_model",
    "perform_cross_validation",
    "simulate_circuit_resistance",
    "simulate_spectra",
    "carbon_nmr_spectra",
    "proton_nmr_spectra",
    "ir_spectra",
    "hsqc_nmr_spectra",
    "mass_spectrometry_spectra",
    "generate_test_measurements",
}
RETRIEVAL_TOOLS = {
    "document_retrieval",
    "is_buyable",
    "keyword_log_extractor",
    "visualisation_tool",
    "head_file",
    "head",
}
IGNORED_TOOL_NAMES = {
    "tool-calling-error",
    "react-error",
    "final_answer",
    "output",
    "none",
    "...",
}
RETRIEVAL_PREFIXES = (
    "get_",
    "retrieve_",
    "search_",
    "batch_retrieve_",
    "smiles_to_",
    "cas_to_",
)
CODE_EXECUTION_PREFIXES = (
    "create_",
    "convert_",
    "filter_",
    "consolidate_",
    "enumerate_",
    "apply_",
    "deprotect_",
    "detect_",
    "map_",
    "generate_",
    "add_",
    "propose_",
    "estimate_",
    "choose_",
    "select_",
)
VALIDATION_PREFIXES = ("validate_", "verify_", "check_")
EXPERIMENT_PREFIXES = ("run_", "simulate_", "train_", "evaluate_", "perform_")


def load_reports_df() -> pd.DataFrame:
    """Load the raw benchmark rows used for all downstream aggregations.

    Returns:
        pd.DataFrame: One row per JSONL record, including nested trial payloads
        that are unpacked later when action shares are computed.
    """
    records: list[dict] = []
    with DATA_PATH.open() as fh:
        for raw_line in fh:
            stripped_line = raw_line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return pd.DataFrame(records)


def ensure_task_results(task_results: object) -> dict:
    """Normalize a `Task Results` payload before iterating over trials.

    Some report rows store task results as serialized JSON, while others already
    provide a mapping. Returning an empty dictionary lets the caller skip
    malformed payloads without special-case branching.

    Args:
        task_results: Raw `Task Results` value read from a report row.

    Returns:
        dict: Parsed task-result mapping, or an empty dictionary when the input
        cannot be interpreted as a mapping.
    """
    if isinstance(task_results, dict):
        return task_results
    if isinstance(task_results, str):
        with suppress(json.JSONDecodeError):
            loaded = json.loads(task_results)
            if isinstance(loaded, dict):
                return loaded
    return {}


def extract_tool_name(tool_call: object) -> str | None:
    """Recover the most reliable tool identifier from a logged call payload.

    The benchmark traces are not schema-stable across agents, so the tool name
    may appear under several alternative keys.

    Args:
        tool_call: Raw object from a trial's `tool_calls` list.

    Returns:
        str | None: The extracted tool name, or `None` when no plausible name is
        present.
    """
    if not isinstance(tool_call, dict):
        return None

    for key in (
        "tool_name",
        "name",
        "tool",
        "function_name",
        "function",
        "tool_call",
    ):
        value = tool_call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        name = function_payload.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    return None


def normalize_tool_name(tool_name: str | None) -> str | None:
    """Clean logging artefacts from a candidate tool name.

    The traces sometimes include channel markers, bracketed wrappers, or entire
    assistant messages where a bare tool name was expected. This filter keeps
    only short, single-line identifiers that are suitable for heuristic
    classification.

    Args:
        tool_name: Candidate tool identifier extracted from a trace.

    Returns:
        str | None: Normalized tool name, or `None` when the value looks like a
        non-tool artefact.
    """
    if tool_name is None:
        return None

    normalized_name = tool_name.strip()
    if not normalized_name:
        return None

    normalized_name = normalized_name.split("<|channel|>", maxsplit=1)[0].strip()

    if (
        normalized_name.startswith("[")
        and normalized_name.endswith("]")
        and "\n" not in normalized_name
    ):
        normalized_name = normalized_name[1:-1].strip()

    if ".json" in normalized_name and len(normalized_name.split()) == 1:
        normalized_name = normalized_name.removesuffix(".json")

    lowered_name = normalized_name.lower()
    if lowered_name in IGNORED_TOOL_NAMES:
        return None

    if any(
        token in lowered_name
        for token in (
            "<thought>",
            "<final_answer>",
            "missing action",
            "tag before final answer",
        )
    ):
        return None

    if "\n" in normalized_name or len(normalized_name) > 80:
        return None

    return normalized_name


def extract_tool_name_from_message(message: object) -> str | None:
    """Infer a tool name from message logs when `tool_calls` is unavailable.

    Older traces sometimes encode tool execution only inside assistant messages
    or `Observation:` payloads. This fallback preserves those trials instead of
    dropping them from the action mix entirely.

    Args:
        message: Raw message object from a trial transcript.

    Returns:
        str | None: Normalized tool name recovered from the message, or `None`
        when the message does not encode a tool invocation.
    """
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str) and content.startswith("Observation:"):
        payload = content.split("Observation:", maxsplit=1)[1].strip()
        with suppress(SyntaxError, ValueError):
            observation = ast.literal_eval(payload)
            tool_name = extract_tool_name(observation)
            if tool_name is not None:
                return normalize_tool_name(tool_name)

    name = message.get("name")
    if isinstance(name, str) and name.strip() and name not in {"assistant", "user"}:
        return normalize_tool_name(name)

    return None


def iter_trial_tool_names(trial: dict) -> list[str]:
    """Collect tool names from a trial in preference order.

    Direct `tool_calls` data is used when available because it is less ambiguous
    than transcript reconstruction. Message parsing is only used as a fallback
    for traces that do not expose structured tool metadata.

    Args:
        trial: Trial payload from the nested benchmark report structure.

    Returns:
        list[str]: Tool names in the order they were recovered from the trace.
    """
    tool_names: list[str] = []

    tool_calls = trial.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            tool_name = normalize_tool_name(extract_tool_name(tool_call))
            if tool_name is not None:
                tool_names.append(tool_name)

    if tool_names:
        return tool_names

    messages = trial.get("messages")
    if isinstance(messages, list):
        for message in messages:
            tool_name = extract_tool_name_from_message(message)
            if tool_name is not None:
                tool_names.append(tool_name)

    return tool_names


def classify_tool(tool_name: str) -> str | None:
    """Assign a tool name to the coarse action taxonomy used in the figure.

    Exact-name lookups are preferred, with all code-producing helper families
    intentionally merged into one `code execution` bucket. Prefix and regex
    heuristics are only used to keep newer tool variants from appearing as
    uncategorized noise.

    Args:
        tool_name: Normalized tool identifier.

    Returns:
        str | None: Action category for the tool, or `None` when the heuristic
        mapping does not recognize it.
    """
    normalized_name = tool_name.strip().lower()

    if normalized_name in RETRIEVAL_TOOLS:
        return "retrieval"
    if normalized_name in FILE_OPERATION_TOOLS:
        return "file operations"
    if normalized_name in CODE_EXECUTION_TOOLS:
        return "code execution"
    if normalized_name in VALIDATION_TOOLS:
        return "validation"
    if normalized_name in EXPERIMENT_TOOLS:
        return "experiment execution"

    if normalized_name.startswith(RETRIEVAL_PREFIXES):
        return "retrieval"
    if normalized_name.startswith(VALIDATION_PREFIXES):
        return "validation"
    if normalized_name.startswith(CODE_EXECUTION_PREFIXES):
        return "code execution"
    if normalized_name.startswith(EXPERIMENT_PREFIXES):
        return "experiment execution"

    if re.search(
        r"(spectra|prediction|polymorph|catalog|template|surface_properties|thermo)",
        normalized_name,
    ):
        return "retrieval"
    if re.search(r"(transform|config|dataset|adsorbate|slab)", normalized_name):
        return "code execution"

    return None


def build_action_distribution_df(
    reports_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert raw reports into per-record action-share summaries.

    Each benchmark record contributes one share value per action category so the
    later averaging step weights records rather than absolute tool-call volume.
    Unknown tools are counted separately to make classification gaps visible in
    the logs.

    Args:
        reports_df: Top-level report table loaded from `reports.jsonl`.

    Returns:
        tuple[pd.DataFrame, dict[str, int]]: A long-form table with one row per
        record-category pair and a frequency table for uncategorized tool names.
    """
    rows: list[dict[str, object]] = []
    unknown_tool_counter: Counter[str] = Counter()

    for _, report in reports_df.iterrows():
        task_results = ensure_task_results(report.get("Task Results"))
        category_counter: Counter[str] = Counter()
        total_actions = 0

        for task_data in task_results.values():
            if not isinstance(task_data, dict):
                continue
            for trial in task_data.get("trials", []):
                if not isinstance(trial, dict):
                    continue
                for tool_name in iter_trial_tool_names(trial):
                    action_category = classify_tool(tool_name)
                    if action_category is None:
                        unknown_tool_counter[tool_name] += 1
                        continue
                    category_counter[action_category] += 1
                    total_actions += 1

        if total_actions == 0:
            continue

        base_metadata = {
            "environment": report.get("environment"),
            "agent_type": report.get("agent_type"),
            "model": report.get("model"),
            "Tool Verbosity": report.get("Tool Verbosity"),
            "level": report.get("level"),
            "category": report.get("category"),
        }
        rows.extend(
            {
                **base_metadata,
                "action_category": action_category,
                "share": category_counter[action_category] / total_actions,
                "count": category_counter[action_category],
                "total_actions": total_actions,
            }
            for action_category in ACTION_ORDER
        )

    return pd.DataFrame(rows), dict(unknown_tool_counter)


def make_pivot(
    distribution_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Average action shares at the requested grouping level.

    Missing categories are filled with zeros so stacked bars remain aligned even
    when a subgroup never uses one of the action families.

    Args:
        distribution_df: Long-form action-share table.
        group_cols: Columns that define each stacked-bar group.

    Returns:
        pd.DataFrame: Wide table with one column per action category ordered to
        match the plotting palette.
    """
    aggregated_df = (
        distribution_df.groupby([*group_cols, "action_category"], as_index=False)[
            "share"
        ]
        .mean()
        .pivot_table(index=group_cols, columns="action_category", values="share")
        .fillna(0.0)
    )

    for action_category in ACTION_ORDER:
        if action_category not in aggregated_df.columns:
            aggregated_df[action_category] = 0.0

    return aggregated_df[ACTION_ORDER]


def safe_float(value: object) -> float:
    """Coerce plotting values to floats while treating missing entries as zero.

    Args:
        value: Scalar-like value from a pivot table.

    Returns:
        float: Numeric value suitable for bar heights.
    """
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return 0.0
    return float(numeric_value)


def plot_grouped_stacked_distribution(
    distribution_df: pd.DataFrame,
    subgroup_col: str,
    subgroup_order: list[str],
    subgroup_labels: dict[str, str],
    out_file: Path,
    *,
    figsize: tuple[float, float],
) -> None:
    """Render environment-level action mixes split by a secondary subgroup.

    The plot uses stacked bars for action categories and letter annotations for
    subgroup identity so the same visual grammar works for both agent-type and
    model comparisons without overwhelming the legend.

    Args:
        distribution_df: Long-form action-share table.
        subgroup_col: Column used to split each environment into multiple bars.
        subgroup_order: Preferred plotting order for subgroup values.
        subgroup_labels: Human-readable labels for subgroup values.
        out_file: Destination PDF path.
        figsize: Figure size in inches.

    Returns:
        None: The function writes a figure to disk.
    """
    env_order = [
        env for env in ENV_LABELS if env in set(distribution_df["environment"])
    ]
    pivot_df = make_pivot(distribution_df, ["environment", subgroup_col])

    n_env = len(env_order)
    n_subgroups = len(subgroup_order)
    x_centers = np.arange(n_env, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / max(n_subgroups, 1)
    offsets = (np.arange(n_subgroups, dtype=float) - (n_subgroups - 1) / 2) * bar_width
    subgroup_markers = {
        subgroup: chr(ord("A") + subgroup_index)
        for subgroup_index, subgroup in enumerate(subgroup_order)
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for subgroup_index, subgroup in enumerate(subgroup_order):
        x_positions = x_centers + offsets[subgroup_index]
        bottom = np.zeros(n_env, dtype=float)

        for action_category in ACTION_ORDER:
            heights: list[float] = []
            for environment in env_order:
                index_key = (environment, subgroup)
                if index_key in pivot_df.index:
                    heights.append(safe_float(pivot_df.loc[index_key, action_category]))
                else:
                    heights.append(0.0)

            ax.bar(
                x_positions,
                heights,
                width=bar_width * 0.92,
                bottom=bottom,
                color=ACTION_COLORS[action_category],
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += np.array(heights, dtype=float)

    ax.set_xlim(-0.6, n_env - 0.4)
    ax.set_ylim(0, 1.06)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [ENV_LABELS.get(env, str(env).capitalize()) for env in env_order]
    )
    ax.set_xlabel("Environment")
    ax.set_ylabel("Average Action Share")
    if n_env > 0:
        range_frame(ax, np.array([0, 6]), np.array([0, 1.0]), pad=0.05)

    if n_env > 0 and n_subgroups > 1:
        example_positions = x_centers[0] + offsets
        for x_position, subgroup in zip(
            example_positions, subgroup_order, strict=False
        ):
            ax.text(
                x_position,
                1.015,
                subgroup_markers[subgroup],
                transform=ax.transData,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                clip_on=False,
            )

    subgroup_handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            label=(
                f"{subgroup_markers[subgroup]}: "
                f"{subgroup_labels.get(subgroup, str(subgroup))}"
            ),
        )
        for subgroup in subgroup_order
    ]

    category_handles = [
        Patch(
            facecolor=ACTION_COLORS[action_category],
            label=ACTION_LABELS[action_category],
        )
        for action_category in ACTION_ORDER
    ]

    if subgroup_handles:
        first_legend = ax.legend(
            handles=subgroup_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            frameon=False,
            title=subgroup_col.replace("_", " ").title(),
            handlelength=0,
            handletextpad=0,
        )
        ax.add_artist(first_legend)

    ax.legend(
        handles=category_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0 - len(subgroup_order) * 0.08 - 0.08),
        ncol=1,
        frameon=False,
        title="Action Category",
    )

    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {out_file}")


def main() -> None:
    """Build action-share summaries and write the comparison figures.

    Returns:
        None: The function saves the requested PDFs and logs any uncategorized
        tool names indirectly through the summary counts.
    """
    reports_df = load_reports_df()
    distribution_df, unknown_tools = build_action_distribution_df(reports_df)

    if distribution_df.empty:
        raise RuntimeError(
            "No tool-level action data could be extracted from reports.jsonl"
        )

    logger.info(
        "Built action distributions for {} benchmark records; {} tool names remained uncategorized.",
        distribution_df[
            [
                "environment",
                "agent_type",
                "model",
                "Tool Verbosity",
                "level",
                "category",
            ]
        ]
        .drop_duplicates()
        .shape[0],
        len(unknown_tools),
    )

    plot_grouped_stacked_distribution(
        distribution_df,
        "agent_type",
        [
            agent
            for agent in AGENT_TYPE_LABELS
            if agent in set(distribution_df["agent_type"])
        ],
        AGENT_TYPE_LABELS,
        OUT_FILE_AGENT,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
    )
    plot_grouped_stacked_distribution(
        distribution_df,
        "model",
        [model for model in MODEL_LABELS if model in set(distribution_df["model"])],
        MODEL_LABELS,
        OUT_FILE_MODEL,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
    )


if __name__ == "__main__":
    main()
