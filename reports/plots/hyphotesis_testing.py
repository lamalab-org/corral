import warnings

import dabest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


def create_sample_data_aggregate():
    n_runs = 15

    run_ids = [f"task_{i+1}" for i in range(n_runs)]

    control_aggregate_scores = np.array([0.67] * n_runs)
    treatment1_aggregate_scores = np.array([0.70] * n_runs)  # 1 tool treatment
    treatment5_aggregate_scores = np.array([0.73] * n_runs)  # 5 tools treatment
    chem1_aggregate_scores = np.array([0.71] * n_runs)  # 1 chemistry tool treatment
    chem5_aggregate_scores = np.array([0.74] * n_runs)  # 5 chemistry tools treatment

    control_total_calls = [4, 4, 5, 5, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 20]
    treatment1_total_calls = [5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 11, 12, 20, 20]
    treatment5_total_calls = [6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 13, 14, 17]
    chem1_total_calls = [4, 4, 4, 5, 5, 8, 8, 8, 8, 9, 9, 10, 10, 10, 20]
    chem5_total_calls = [4, 4, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 11, 12, 20]

    data = []

    # Use list comprehensions for better performance
    data.extend(
        [
            {
                "run_id": run_ids[i],
                "condition": "vanilla task",
                "aggregate_score": control_aggregate_scores[i],
                "total_tool_calls": int(control_total_calls[i]),
            }
            for i in range(n_runs)
        ]
    )

    data.extend(
        [
            {
                "run_id": run_ids[i],
                "condition": "1 extra general tool",
                "aggregate_score": treatment1_aggregate_scores[i],
                "total_tool_calls": int(treatment1_total_calls[i]),
            }
            for i in range(n_runs)
        ]
    )

    data.extend(
        [
            {
                "run_id": run_ids[i],
                "condition": "5 extra general tools",
                "aggregate_score": treatment5_aggregate_scores[i],
                "total_tool_calls": int(treatment5_total_calls[i]),
            }
            for i in range(n_runs)
        ]
    )

    data.extend(
        [
            {
                "run_id": run_ids[i],
                "condition": "1 chemistry tool",
                "aggregate_score": chem1_aggregate_scores[i],
                "total_tool_calls": int(chem1_total_calls[i]),
            }
            for i in range(n_runs)
        ]
    )

    data.extend(
        [
            {
                "run_id": run_ids[i],
                "condition": "5 chemistry tools",
                "aggregate_score": chem5_aggregate_scores[i],
                "total_tool_calls": int(chem5_total_calls[i]),
            }
            for i in range(n_runs)
        ]
    )

    return pd.DataFrame(data)


df_scores = create_sample_data_aggregate()

logger.info(df_scores.head(10))

dabest_tool_calls = dabest.load(
    data=df_scores,
    x="condition",
    y="total_tool_calls",
    idx=(
        "vanilla task",
        "1 extra general tool",
        "5 extra general tools",
        "1 chemistry tool",
        "5 chemistry tools",
    ),
)

dabest_results = dabest_tool_calls.mean_diff

fig_dabest = dabest_results.plot(fig_size=(14, 10))
fig_dabest.suptitle(
    "DABEST Estimation Plot: Effect of Extra Tools on Total Tool Usage",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

plt.savefig("tool_usage_analysis.png", dpi=300, bbox_inches="tight")
