"""Plot subtask heaviness and success rates as flowing ribbons.

Shows task workflow progression with:
- Color: Reasoning heaviness (reasoning > validation > code_execution > experiment_execution > retrieval)
- Thickness: Success rate (thicker = higher success)

Usage:
    python 4_task_heaviness.py --verbosity_strategy=average --level_strategy=default_map
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import Normalize
from scipy.constants import golden

lama_aesthetics.get_style("main")

# Add analysis and plot config to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import FONT_SIZES  # noqa: E402

# ==================== CONFIGURATION ====================

# Data paths
REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "reports.jsonl"
REASONING_PATH = REPO_ROOT / "analysis" / "reasoning.json"

# Figure sizing
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

# Heaviness scores for different reasoning types
HEAVINESS_SCORES = {
    "reasoning": 5,
    "validation": 4,
    "code_execution": 3,
    "experiment_execution": 2,
    "retrieval": 1,
    "unknown": 1,
}

# Default per-environment level selection
DEFAULT_ENV_LEVEL_MAP = {
    "afm": 1,
    "catalyst": 1,
    "md": 2,
    "ml": 1,
    "resistor": 1,
    "retro": 2,
    "spectra": 1,
    "wetlab": 2,
}

# Environment name normalization (maps from reports.jsonl to reasoning.json)
# Only needed for environments where the names differ
ENV_NAME_MAP = {
    "retro": "retrosynthesis",  # Reports: "retro" -> reasoning.json: "retrosynthesis"
    "spectra": "sptectra",  # Reports: "spectra" -> reasoning.json: "sptectra" (typo in json)
    # Note: "ml" matches in both, no mapping needed
}


# ==================== DATA LOADING ====================


def load_reports_data() -> pd.DataFrame:
    """Load main benchmark reports dataset."""
    if not REPORTS_PATH.exists():
        msg = f"Reports file not found: {REPORTS_PATH}"
        raise FileNotFoundError(msg)

    df = pd.read_json(REPORTS_PATH, lines=True)  # noqa: PD901
    logger.info(f"Loaded {len(df)} rows from reports.jsonl")
    return df


def load_reasoning_data() -> dict:
    """Load reasoning heaviness mapping."""
    if not REASONING_PATH.exists():
        msg = f"Reasoning file not found: {REASONING_PATH}"
        raise FileNotFoundError(msg)

    with REASONING_PATH.open() as f:
        reasoning_data = json.load(f)

    logger.info(f"Loaded reasoning data for {len(reasoning_data)} environments")
    return reasoning_data


# ==================== FILTERING FUNCTIONS ====================


def filter_by_verbosity(df: pd.DataFrame, verbosity_strategy: str) -> pd.DataFrame:
    """Filter dataframe by verbosity strategy."""
    if verbosity_strategy == "average":
        return df
    if verbosity_strategy in ["brief", "workflow", "comprehensive"]:
        return df[df["Tool Verbosity"] == verbosity_strategy]
    msg = f"Invalid verbosity_strategy: {verbosity_strategy}"
    raise ValueError(msg)


def filter_by_level(df: pd.DataFrame, level_strategy: str) -> pd.DataFrame:
    """Filter dataframe by level strategy."""
    if level_strategy == "all":
        return df

    if level_strategy == "default_map":
        mask = pd.Series(False, index=df.index)
        for env, level in DEFAULT_ENV_LEVEL_MAP.items():
            mask |= (df["environment"] == env) & (df["level"] == level)
        return df[mask]

    # Parse as integer level
    try:
        level_int = int(level_strategy)
        if level_int < 1:
            msg = f"Level must be >= 1, got: {level_int}"
            raise ValueError(msg)
        return df[df["level"] == level_int]
    except ValueError as e:
        if "invalid literal" in str(e):
            msg = f"Invalid level_strategy: {level_strategy}"
            raise ValueError(msg) from e
        raise


def filter_by_model(df: pd.DataFrame, model_strategy: str) -> pd.DataFrame:
    """Filter dataframe by model strategy."""
    if model_strategy == "average":
        return df
    # Specific model
    return df[df["model"] == model_strategy]


def filter_by_agent(df: pd.DataFrame, agent_strategy: str) -> pd.DataFrame:
    """Filter dataframe by agent strategy."""
    if agent_strategy == "average":
        return df
    if agent_strategy in ["react", "tool_calling"]:
        return df[df["agent_type"] == agent_strategy]
    msg = f"Invalid agent_strategy: {agent_strategy}"
    raise ValueError(msg)


# ==================== SUBTASK EXTRACTION ====================


def extract_subtask_data(df: pd.DataFrame) -> dict:
    """Extract subtask sequences and success rates from filtered data.

    Returns:
        Dict with {env_key: {"sequence": [...], "success_rates": [...]}}
    """
    # Filter to subtask category only
    df = df[df["category"] == "subtask"].copy()  # noqa: PD901

    if df.empty:
        logger.warning("No subtask data found after filtering")
        return {}

    result = {}

    # Group by environment and level
    for (env, level), env_df in df.groupby(["environment", "level"]):
        # Collect all subtask results from Task Results column
        all_subtask_data = defaultdict(list)  # subtask_name -> [scores]

        for _, row in env_df.iterrows():
            task_results = row["Task Results"]
            if not isinstance(task_results, dict):
                continue

            for subtask_name, subtask_result in task_results.items():
                if isinstance(subtask_result, dict):
                    score = subtask_result.get("Task Average Score", 0.0)

                    # For environments with multiple workflows, normalize to generic names
                    normalized_subtask = normalize_subtask_name(subtask_name, env)
                    all_subtask_data[normalized_subtask].append(score)

        if not all_subtask_data:
            continue

        # Sort subtask names and compute average success rates
        subtasks = sorted(all_subtask_data.keys())
        success_rates = [np.mean(all_subtask_data[st]) for st in subtasks]

        # Create environment key
        env_key = f"{env}_level_{level}"

        result[env_key] = {
            "sequence": subtasks,
            "success_rates": success_rates,
            "env": env,
            "level": level,
        }

    return result


def normalize_subtask_name(subtask_name: str, env: str) -> str:
    """Normalize subtask name by removing workflow-specific prefixes.

    This groups multiple workflows into generic subtask patterns that match reasoning.json.

    Args:
        subtask_name: Original subtask name from reports
        env: Environment name

    Returns:
        Normalized subtask name matching reasoning.json keys
    """
    # Catalyst: Remove element prefix (e.g., "cu20_add_adsorbate" -> "add_adsorbate")
    # Multiple elements tested (si, cu, fe, etc.) but all follow same subtask pattern
    if env == "catalyst":
        parts = subtask_name.split("_", 1)
        if len(parts) == 2 and parts[0] and parts[0][0].isalpha():
            return parts[1]

    # Resistor: Remove task ID prefix (e.g., "task_0_subnet_1" -> "subnet_1")
    # Multiple tasks but all follow same subtask pattern
    if env == "resistor" and subtask_name.startswith("task_"):
        parts = subtask_name.split("_", 2)
        if len(parts) >= 3:
            return "_".join(parts[2:])

    # Retrosynthesis: Remove make/level prefix (e.g., "make_1_lvl1-apply_template-1" -> "apply_template-1")
    # Multiple molecules to synthesize but all follow same template pattern
    if env == "retro" and "-" in subtask_name:
        parts = subtask_name.split("-")
        for i, part in enumerate(parts):
            if any(
                kw in part
                for kw in ["template_search", "apply_template", "build_complete_route"]
            ):
                return "-".join(parts[i:])

    # Spectra: Remove workflow ID prefix (e.g., "10_15227_orgsyn_084_0077_subtask_1" -> "subtask_1")
    # 20 different molecules but all follow same 10-subtask pattern
    if env == "spectra" and "subtask_" in subtask_name:
        idx = subtask_name.find("subtask_")
        return subtask_name[idx:]

    # ML: Remove task ID prefix and normalize variations
    # Multiple materials (oxide, nitride, sulphide) but all follow same ML workflow pattern
    if env == "ml":
        # Remove task number prefix (e.g., "0_batch_retrieve..." -> "batch_retrieve...")
        if "_" in subtask_name:
            parts = subtask_name.split("_", 1)
            if parts[0].isdigit() and len(parts) == 2:
                subtask_name = parts[1]

        # Normalize specific variations to match reasoning.json keys:
        # "batch_retrieve_*_polymorphs" -> "batch_retrieve_oxide_polymorphs" (generic)
        if subtask_name.startswith("batch_retrieve_") and "polymorphs" in subtask_name:
            return "batch_retrieve_oxide_polymorphs"

        # "prepare_ml_ready_dataset" -> "prepare_tabular_dataset"
        if "prepare_ml_ready" in subtask_name:
            return "prepare_tabular_dataset"

        # "train_xgboost_formation_energy_model" or similar -> "train_xgboost_model"
        if subtask_name.startswith("train_xgboost"):
            return "train_xgboost_model"

        # "evaluate_xgboost_model" variations -> "evaluate_xgboost_model"
        if "evaluate_xgboost" in subtask_name:
            return "evaluate_xgboost_model"

        return subtask_name

    # For other environments (like AFM, MD), return as is - they already match reasoning.json
    return subtask_name


def get_heaviness_for_subtask(
    env: str, subtask_name: str, reasoning_data: dict
) -> float:
    """Get heaviness score for a subtask.

    Args:
        env: Environment name
        subtask_name: Subtask name
        reasoning_data: Reasoning mapping loaded from reasoning.json

    Returns:
        Heaviness score (1-5)
    """
    # Normalize environment name (for looking up in reasoning.json)
    env_norm = ENV_NAME_MAP.get(env, env)

    if env_norm not in reasoning_data:
        return HEAVINESS_SCORES["unknown"]

    env_data = reasoning_data[env_norm]

    # Normalize subtask name (use original env name, not env_norm)
    normalized_name = normalize_subtask_name(subtask_name, env)

    # Handle nested structure (e.g., MD has subtypes)
    if isinstance(env_data, dict):
        # Check if normalized subtask is directly in env_data
        if normalized_name in env_data:
            reasoning_type = env_data[normalized_name]
            return HEAVINESS_SCORES.get(reasoning_type, HEAVINESS_SCORES["unknown"])

        # Check nested structures (for MD)
        for subtype_data in env_data.values():
            if isinstance(subtype_data, dict) and normalized_name in subtype_data:
                reasoning_type = subtype_data[normalized_name]
                return HEAVINESS_SCORES.get(reasoning_type, HEAVINESS_SCORES["unknown"])

    return HEAVINESS_SCORES["unknown"]


# ==================== PLOTTING ====================


def plot_subtask_heaviness(
    subtask_data: dict,
    reasoning_data: dict,
    output_path: Path,
    env_filter: str | None = None,
) -> None:
    """Create subtask heaviness ribbon plot.

    Args:
        subtask_data: Dict with {env_key: {"sequence": [...], "success_rates": [...]}}
        reasoning_data: Reasoning mapping
        output_path: Path to save the figure
        env_filter: Optional environment filter (e.g., "catalyst")
    """
    # Filter environments if requested
    if env_filter:
        subtask_data = {
            k: v for k, v in subtask_data.items() if env_filter.lower() in k.lower()
        }

    if not subtask_data:
        logger.warning("No subtask data to plot")
        return

    env_keys = sorted(subtask_data.keys())
    num_envs = len(env_keys)

    logger.info(f"Plotting {num_envs} environment sequences:")
    for env_key in env_keys:
        num_subtasks = len(subtask_data[env_key]["sequence"])
        logger.info(f"  {env_key}: {num_subtasks} subtasks")

    # Create figure
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, max(num_envs * 0.25 + 1, 3)))

    # Color map for heaviness
    cmap = sns.cubehelix_palette(as_cmap=True)
    norm_heavy = Normalize(vmin=1, vmax=5)

    for y_idx, env_key in enumerate(env_keys):
        data = subtask_data[env_key]
        subtasks = data["sequence"]
        success_rates = data["success_rates"]
        env = data["env"]

        num_subtasks = len(subtasks)
        if num_subtasks == 0:
            continue

        # Get heaviness values for each subtask
        heaviness_values = [
            get_heaviness_for_subtask(env, subtask, reasoning_data)
            for subtask in subtasks
        ]

        # Create smooth flowing ribbon
        segment_width = 1.0 / num_subtasks
        control_x = [(i + 0.5) * segment_width for i in range(num_subtasks)]

        # Add endpoints
        control_x = [0.0, *control_x, 1.0]
        control_scores = [success_rates[0], *success_rates, success_rates[-1]]
        control_heaviness = [
            heaviness_values[0],
            *heaviness_values,
            heaviness_values[-1],
        ]

        # Interpolate to create smooth curve
        num_interp_points = 200
        x_smooth = np.linspace(0, 1, num_interp_points)

        # Interpolate scores (thickness)
        scores_smooth = np.interp(x_smooth, control_x, control_scores)

        # Interpolate heaviness (color)
        heaviness_smooth = np.interp(x_smooth, control_x, control_heaviness)

        # Convert scores to half-widths (for ribbon)
        min_half_width = 0.02
        max_half_width = 0.18
        half_widths = min_half_width + (max_half_width - min_half_width) * scores_smooth

        # Draw ribbon as many small colored segments
        for j in range(len(x_smooth) - 1):
            x_seg = [x_smooth[j], x_smooth[j + 1], x_smooth[j + 1], x_smooth[j]]
            y_seg = [
                y_idx - half_widths[j],
                y_idx - half_widths[j + 1],
                y_idx + half_widths[j + 1],
                y_idx + half_widths[j],
            ]
            # Use average heaviness for this segment's color
            avg_heaviness = (heaviness_smooth[j] + heaviness_smooth[j + 1]) / 2
            color = cmap(norm_heavy(avg_heaviness))
            ax.fill(x_seg, y_seg, color=color, alpha=0.85, linewidth=0)

    # Styling
    ax.set_yticks(range(num_envs))
    ax.set_yticklabels(env_keys, fontsize=FONT_SIZES["tick_label"])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, num_envs - 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["First Subtask", "Last Subtask"], fontsize=FONT_SIZES["tick_label"]
    )
    ax.set_xlabel(
        "Normalized Workflow Progression",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_title(
        "Subtask Reasoning Heaviness (Color) and Success Score (Thickness)",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )

    # Colorbar for heaviness
    sm_heavy = plt.cm.ScalarMappable(cmap=cmap, norm=norm_heavy)
    cbar_heavy = fig.colorbar(sm_heavy, ax=ax, fraction=0.046, pad=0.04)
    cbar_heavy.set_label(
        "Reasoning Heaviness", fontsize=FONT_SIZES["axis_label"], fontweight="bold"
    )
    cbar_heavy.ax.tick_params(labelsize=FONT_SIZES["tick_label"])

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    fig.savefig(
        output_path.with_suffix(".png"),
        bbox_inches="tight",
        dpi=300,
    )
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


# ==================== MAIN ====================


def main(
    verbosity_strategy: str = "average",
    level_strategy: str = "default_map",
    model_strategy: str = "average",
    agent_strategy: str = "average",
    env_filter: str | None = None,
    output_filename: str = "subtask_heaviness.pdf",
) -> None:
    """Generate subtask heaviness ribbon plot.

    Args:
        verbosity_strategy: Which tool verbosity to use.
            Options: "average", "brief", "workflow", "comprehensive"
        level_strategy: Which level to use for environments.
            Options: "all" (average across all levels),
                     "default_map" (use per-environment mapping),
                     or specific level like "1", "2", "3", "4"
        model_strategy: Which model to use.
            Options: "average" (average all models),
                     "claude-4.5", "gpt-4o", "gpt-oss-120b"
        agent_strategy: Which agent type to use.
            Options: "average" (average both),
                     "react", "tool_calling"
        env_filter: Optional filter for specific environment (e.g., "catalyst", "md")
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Subtask Heaviness Plot Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Level strategy: {level_strategy}")
    logger.info(f"Model strategy: {model_strategy}")
    logger.info(f"Agent strategy: {agent_strategy}")
    if env_filter:
        logger.info(f"Environment filter: {env_filter}")
    logger.info("")

    # Load data
    logger.info("Loading datasets...")
    reports_df = load_reports_data()
    reasoning_data = load_reasoning_data()

    # Filter reports data
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(filtered_df, verbosity_strategy)
    filtered_df = filter_by_level(filtered_df, level_strategy)
    filtered_df = filter_by_model(filtered_df, model_strategy)
    filtered_df = filter_by_agent(filtered_df, agent_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    if level_strategy == "default_map":
        logger.info(f"Using default level map: {DEFAULT_ENV_LEVEL_MAP}")
    logger.info("")

    # Extract subtask data
    logger.info("Extracting subtask sequences and success rates...")
    subtask_data = extract_subtask_data(filtered_df)

    if not subtask_data:
        logger.error("No subtask data found after filtering!")
        return

    # Generate plot
    logger.info("Generating plot...")
    output_path = Path(output_filename)
    plot_subtask_heaviness(subtask_data, reasoning_data, output_path, env_filter)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Subtask heaviness plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
