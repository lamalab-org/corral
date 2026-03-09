"""Plot action-category distributions across benchmark environments.

This script loads `analysis/results/data/reports.jsonl`, extracts per-tool actions from
nested trial data, maps tools into coarse action categories, and produces three
figures:

- environment comparison averaged over level, verbosity, agent type, and model
- environment comparison with separate bars for each agent type, averaged over
  level, verbosity, and model
- environment comparison with separate bars for each model, averaged over level,
  verbosity, and agent type

Action categories:
- retrieval
- file operations
- code execution
- experiment execution
- validation
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
from loguru import logger
from matplotlib.patches import Patch

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures"
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
    "tool_calling": "Tool calling",
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
    "file operations": "File ops",
    "code execution": "Code exec.",
    "experiment execution": "Experiment exec.",
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
}
VALIDATION_TOOLS = {
    "analyze_measurements",
    "validate_smiles",
    "validate_measurements",
    "verify_step",
    "verify_route",
    "check_smiles_reaction_template_matching",
}
HYPOTHESIS_TOOLS = {
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
}
EXPERIMENT_TOOLS = {
    "image_analyzer",
    "calculate_parallel_resistance",
    "calculate_series_resistance",
    "run_lammps",
    "train_xgboost_model",
    "evaluate_xgboost_model",
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
TRANSFORMATION_TOOLS = {
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
TRANSFORMATION_PREFIXES = (
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
)
VALIDATION_PREFIXES = ("validate_", "verify_", "check_")
HYPOTHESIS_PREFIXES = ("propose_", "estimate_", "choose_", "select_")
EXPERIMENT_PREFIXES = ("run_", "simulate_", "train_", "evaluate_", "perform_")


def load_reports_df() -> pd.DataFrame:
    """Load the combined benchmark reports."""
    records: list[dict] = []
    with DATA_PATH.open() as fh:
        for raw_line in fh:
            stripped_line = raw_line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return pd.DataFrame(records)


def ensure_task_results(task_results: object) -> dict:
    """Return task results as a dictionary."""
    if isinstance(task_results, dict):
        return task_results
    if isinstance(task_results, str):
        with suppress(json.JSONDecodeError):
            loaded = json.loads(task_results)
            if isinstance(loaded, dict):
                return loaded
    return {}


def extract_tool_name(tool_call: object) -> str | None:
    """Extract a tool name from a tool-call payload."""
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
    """Normalize noisy tool-name strings extracted from logs."""
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
    """Extract a tool name from a logged message if direct tool calls are absent."""
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
    """Return tool names used in a trial."""
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
    """Map a tool name into a coarse action category."""
    normalized_name = tool_name.strip().lower()

    if normalized_name in RETRIEVAL_TOOLS:
        return "retrieval"
    if normalized_name in FILE_OPERATION_TOOLS:
        return "file operations"
    if normalized_name in CODE_EXECUTION_TOOLS:
        return "code execution"
    if normalized_name in TRANSFORMATION_TOOLS:
        return "code execution"
    if normalized_name in VALIDATION_TOOLS:
        return "validation"
    if normalized_name in HYPOTHESIS_TOOLS:
        return "code execution"
    if normalized_name in EXPERIMENT_TOOLS:
        return "experiment execution"

    if normalized_name.startswith(RETRIEVAL_PREFIXES):
        return "retrieval"
    if normalized_name.startswith(TRANSFORMATION_PREFIXES):
        return "code execution"
    if normalized_name.startswith(VALIDATION_PREFIXES):
        return "validation"
    if normalized_name.startswith(HYPOTHESIS_PREFIXES):
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
    """Create one normalized action-distribution row per benchmark record."""
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
    """Aggregate mean shares and return a wide table."""
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


def plot_grouped_stacked_distribution(
    distribution_df: pd.DataFrame,
    subgroup_col: str,
    subgroup_order: list[str],
    subgroup_labels: dict[str, str],
    out_file: Path,
    *,
    figsize: tuple[float, float],
) -> None:
    """Plot grouped stacked bars split by either agent type or model."""
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
    hatches = ["", "//", "xx", "..", "\\"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for subgroup_index, subgroup in enumerate(subgroup_order):
        x_positions = x_centers + offsets[subgroup_index]
        bottom = np.zeros(n_env, dtype=float)

        for action_category in ACTION_ORDER:
            heights: list[float] = []
            for environment in env_order:
                index_key = (environment, subgroup)
                if index_key in pivot_df.index:
                    heights.append(float(pivot_df.loc[index_key, action_category]))
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
                hatch=hatches[subgroup_index % len(hatches)],
            )
            bottom += np.array(heights, dtype=float)

    ax.set_xlim(-0.6, n_env - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [ENV_LABELS.get(env, str(env).capitalize()) for env in env_order]
    )
    ax.set_xlabel("Environment")
    ax.set_ylabel("Average action share")

    subgroup_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatches[idx % len(hatches)],
            label=subgroup_labels.get(subgroup, str(subgroup)),
        )
        for idx, subgroup in enumerate(subgroup_order)
    ]
    category_handles = [
        Patch(
            facecolor=ACTION_COLORS[action_category],
            label=ACTION_LABELS[action_category],
        )
        for action_category in ACTION_ORDER
    ]

    first_legend = ax.legend(
        handles=subgroup_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        frameon=False,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=category_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0 - len(subgroup_order) * 0.08 - 0.06),
        ncol=1,
        frameon=False,
        title="Action category",
    )

    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {out_file}")


def main() -> None:
    """Load the reports, build action distributions, and save all figures."""
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
