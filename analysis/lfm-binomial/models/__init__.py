"""
Model registry (Binomial likelihood).

Maps model number to module. Each module exposes build_model(df) -> pm.Model.
"""

from . import (
    model1_baseline_tasks,
    model2_tasks_environment,
    model3_abilities_env,
    model4_scaffold_env,
    model5_scaffold_level,
    model6_env_level,
    model7_abilities_env_level,
    model8_abilities_env_envlevel_intercept,
)

MODELS = {
    1: model1_baseline_tasks,
    2: model2_tasks_environment,
    3: model3_abilities_env,
    4: model4_scaffold_env,
    5: model5_scaffold_level,
    6: model6_env_level,
    7: model7_abilities_env_level,
    8: model8_abilities_env_envlevel_intercept,
}

MODEL_NAMES = {
    1: "model1_baseline_tasks",
    2: "model2_tasks_environment",
    3: "model3_abilities_env",
    4: "model4_scaffold_env",
    5: "model5_scaffold_level",
    6: "model6_env_level",
    7: "model7_abilities_env_level",
    8: "model8_abilities_env_envlevel_intercept",
}
