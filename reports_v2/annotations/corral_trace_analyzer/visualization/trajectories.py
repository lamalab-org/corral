"""
Visualization for trace trajectories over time
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from ..config import VIZ_CONFIG


class TrajectoryVisualizer:
    """Visualize trace trajectories and temporal patterns"""

    def __init__(self, features_df: pd.DataFrame, steps_df: pd.DataFrame):
        """
        Initialize visualizer

        Args:
            features_df: DataFrame with features
            steps_df: DataFrame with step-level data
        """
        self.features_df = features_df
        self.steps_df = steps_df
        sns.set_style(VIZ_CONFIG["style"])
        sns.set_context(VIZ_CONFIG["context"])

    def plot_marker_trajectory(
        self,
        trace_id: str,
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot marker trajectory for a single trace

        Args:
            trace_id: Trace ID to visualize
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        trace_steps = self.steps_df[self.steps_df["trace_id"] == trace_id].sort_values(
            "step_index"
        )

        if len(trace_steps) == 0:
            raise ValueError(f"No steps found for trace_id: {trace_id}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot marker occurrences
        for idx, row in trace_steps.iterrows():
            step_idx = row["step_index"]

            if row["has_positive"]:
                ax1.scatter(step_idx, 1, color="green", s=100, marker="^", alpha=0.7)
            if row["has_negative"]:
                ax1.scatter(step_idx, 0, color="red", s=100, marker="v", alpha=0.7)
            if row["has_neutral"]:
                ax1.scatter(step_idx, 0.5, color="gray", s=50, marker="o", alpha=0.5)

        ax1.set_ylabel("Marker Type", fontsize=12)
        ax1.set_yticks([0, 0.5, 1])
        ax1.set_yticklabels(["Negative", "Neutral", "Positive"])
        ax1.set_title(
            f"Marker Trajectory for Trace: {trace_id}", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Plot cumulative marker balance
        cumulative_positive = trace_steps["has_positive"].cumsum()
        cumulative_negative = trace_steps["has_negative"].cumsum()
        balance = cumulative_positive - cumulative_negative

        ax2.plot(
            trace_steps["step_index"],
            balance,
            color="blue",
            linewidth=2,
            label="Marker Balance",
        )
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax2.fill_between(
            trace_steps["step_index"],
            0,
            balance,
            where=(balance >= 0),
            color="green",
            alpha=0.3,
            label="Positive Balance",
        )
        ax2.fill_between(
            trace_steps["step_index"],
            0,
            balance,
            where=(balance < 0),
            color="red",
            alpha=0.3,
            label="Negative Balance",
        )

        ax2.set_xlabel("Step Index", fontsize=12)
        ax2.set_ylabel("Cumulative Balance", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_average_marker_trajectory(
        self,
        group_col: str | None = None,
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot average marker trajectory across all traces

        Args:
            group_col: Optional column to group by
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if group_col and group_col in self.steps_df.columns:
            groups = self.steps_df[group_col].unique()

            for group in groups:
                group_steps = self.steps_df[self.steps_df[group_col] == group]

                # Calculate average positive ratio at each step index
                step_stats = (
                    group_steps.groupby("step_index")
                    .agg({"has_positive": "mean", "has_negative": "mean"})
                    .reset_index()
                )

                step_stats["balance"] = (
                    step_stats["has_positive"] - step_stats["has_negative"]
                )

                ax.plot(
                    step_stats["step_index"],
                    step_stats["balance"],
                    linewidth=2,
                    label=f"{group}",
                    alpha=0.8,
                )

        else:
            # Average across all traces
            step_stats = (
                self.steps_df.groupby("step_index")
                .agg({"has_positive": "mean", "has_negative": "mean"})
                .reset_index()
            )

            step_stats["balance"] = (
                step_stats["has_positive"] - step_stats["has_negative"]
            )

            ax.plot(
                step_stats["step_index"],
                step_stats["balance"],
                linewidth=2,
                color="blue",
                label="All Traces",
            )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Step Index", fontsize=12)
        ax.set_ylabel("Average Marker Balance", fontsize=12)
        ax.set_title("Average Marker Trajectory", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_step_type_distribution_over_time(
        self,
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot how step types are distributed over time

        Args:
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Group by step index and count node types
        step_type_counts = (
            self.steps_df.groupby(["step_index", "node_type"])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=figsize)

        step_type_counts.plot(kind="area", stacked=True, alpha=0.7, ax=ax)

        ax.set_xlabel("Step Index", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Step Type Distribution Over Time", fontsize=14, fontweight="bold")
        ax.legend(title="Node Type")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_success_failure_timeline(
        self,
        trace_id: str,
        tools_df: pd.DataFrame,
        figsize: tuple[int, int] = (12, 4),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot timeline of tool successes and failures

        Args:
            trace_id: Trace ID to visualize
            tools_df: DataFrame with tool call data
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        trace_tools = tools_df[tools_df["trace_id"] == trace_id].copy()

        if len(trace_tools) == 0:
            raise ValueError(f"No tool calls found for trace_id: {trace_id}")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot successes and failures
        for idx, row in trace_tools.iterrows():
            color = "green" if row["is_success"] else "red"
            marker = "o" if row["is_success"] else "x"

            ax.scatter(
                row["tool_index"], 0, color=color, s=200, marker=marker, alpha=0.7
            )

            # Add tool name as text
            ax.text(
                row["tool_index"],
                0.02,
                row["tool_name"],
                rotation=90,
                fontsize=8,
                ha="center",
                va="bottom",
            )

        ax.set_ylim(-0.05, 0.15)
        ax.set_xlabel("Tool Call Index", fontsize=12)
        ax.set_title(
            f"Tool Call Timeline for Trace: {trace_id}", fontsize=14, fontweight="bold"
        )
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Success"),
            Patch(facecolor="red", alpha=0.7, label="Failure"),
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_interactive_trace_timeline(self, trace_id: str) -> go.Figure:
        """
        Create interactive timeline visualization using Plotly

        Args:
            trace_id: Trace ID to visualize

        Returns:
            Plotly figure
        """
        trace_steps = self.steps_df[self.steps_df["trace_id"] == trace_id].sort_values(
            "step_index"
        )

        if len(trace_steps) == 0:
            raise ValueError(f"No steps found for trace_id: {trace_id}")

        # Create figure
        fig = go.Figure()

        # Add markers
        positive_steps = trace_steps[trace_steps["has_positive"]]
        negative_steps = trace_steps[trace_steps["has_negative"]]

        if len(positive_steps) > 0:
            fig.add_trace(
                go.Scatter(
                    x=positive_steps["step_index"],
                    y=[1] * len(positive_steps),
                    mode="markers",
                    name="Positive",
                    marker=dict(color="green", size=12, symbol="triangle-up"),
                    text=positive_steps["message"].str[:100],
                    hovertemplate="<b>Step %{x}</b><br>%{text}<extra></extra>",
                )
            )

        if len(negative_steps) > 0:
            fig.add_trace(
                go.Scatter(
                    x=negative_steps["step_index"],
                    y=[0] * len(negative_steps),
                    mode="markers",
                    name="Negative",
                    marker=dict(color="red", size=12, symbol="triangle-down"),
                    text=negative_steps["message"].str[:100],
                    hovertemplate="<b>Step %{x}</b><br>%{text}<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Interactive Trace Timeline: {trace_id}",
            xaxis_title="Step Index",
            yaxis_title="Marker Type",
            yaxis=dict(
                tickmode="array", tickvals=[0, 1], ticktext=["Negative", "Positive"]
            ),
            hovermode="closest",
            height=400,
        )

        return fig
