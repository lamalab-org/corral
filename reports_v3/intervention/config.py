"""Shared configuration for intervention experiments."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_V2 = PROJECT_ROOT / "reports_v2" / "claude_sonnet_45"
INTERVENTION_ROOT = Path(__file__).resolve().parent

# Environment configs
# Each env has two ports: one for react agent, one for toolcalling agent.
# The server is stateful (per-task state), so parallel agents on the same
# server would clash. Two ports allow full parallelism.
ENVIRONMENTS = {
    "spectra": {
        "port": {"react": 8002, "toolcalling": 8012},
        "level": 2,
        "task_dir": "tasks",
        "max_iterations": 20,
        "report_v2_path": REPORTS_V2 / "spectra" / "level_2" / "tasks",
        "react_report": "claude_45_sonnet-react-spectra_lvl2_env-workflow_verbosity.json",
        "toolcalling_report": "claude_45_sonnet-tool_calling-spectra_lvl2_env-workflow_verbosity.json",
    },
    "wetlab": {
        "port": {"react": 8003, "toolcalling": 8013},
        "level": 2,
        "task_dir": "task",
        "max_iterations": 40,
        "report_v2_path": REPORTS_V2 / "wetlab" / "level_2" / "task",
        "react_report": "claude_sonnet_45-ReAct-WetLab_Level_2-workflow.json",
        "toolcalling_report": "claude_sonnet_45-Tool_Calling-WetLab_Level_2-workflow.json",
    },
    "resistor": {
        "port": {"react": 8001, "toolcalling": 8011},
        "level": 1,
        "task_dir": "tasks",
        "max_iterations": 20,
        "report_v2_path": REPORTS_V2 / "resistor" / "level_1" / "tasks",
        "react_report": "claude-react-resistor_network-workflow_verbosity_single.json",
        "toolcalling_report": "claude-toolcalling-resistor_network-workflow_verbosity_single.json",
    },
    "ml": {
        "port": {"react": 8004, "toolcalling": 8014},
        "level": 1,
        "task_dir": "tasks",
        "max_iterations": 20,
        "report_v2_path": REPORTS_V2 / "ml" / "level_1" / "tasks",
        "react_report": "claude-react-ml-workflow_verbosity_single.json",
        "toolcalling_report": "claude-toolcalling-ml-workflow_verbosity_single.json",
    },
    "catalyst": {
        "port": {"react": 8005, "toolcalling": 8015},
        "level": 1,
        "task_dir": "tasks",
        "max_iterations": 20,
        "report_v2_path": REPORTS_V2 / "catalyst" / "level_1" / "tasks",
        "react_report": "claude-react-catalyst-workflow_verbosity_single.json",
        "toolcalling_report": "claude-toolcalling-catalyst-workflow_verbosity_single.json",
    },
    "retrosynthesis": {
        "port": {"react": 8006, "toolcalling": 8016},
        "level": 2,
        "task_dir": "tasks",
        "max_iterations": 30,
        "report_v2_path": REPORTS_V2 / "retrosynthesis" / "level_2" / "tasks",
        "react_report": "claude_45_sonnet-react-retro_lvl2_env-workflow_verbosity.json",
        "toolcalling_report": "claude_45_sonnet-tool_calling-retro_lvl2_env-workflow_verbosity.json",
    },
    "md": {
        "port": {"react": 8007, "toolcalling": 8017},
        "level": 2,
        "task_dir": "tasks",
        "max_iterations": 30,
        # MD has multiple sub-envs; report_v2 paths are per-task (handled in select_traces)
        "report_v2_paths": {
            "silicon_melting_2": REPORTS_V2 / "md" / "melting" / "level_2" / "tasks",
            "aluminum_surface_energy_2": REPORTS_V2
            / "md"
            / "surface_energy"
            / "level_2"
            / "tasks",
            "na2sio3_quenching_1": REPORTS_V2
            / "md"
            / "quenching"
            / "level_1"
            / "tasks",
        },
        "report_v2_path": REPORTS_V2 / "md" / "melting" / "level_2" / "tasks",
        "react_report": "claude_45-react-melting-workflow-task-level_2_verbosity_try.json",
        "toolcalling_report": "claude_45-tool_calling-melting-workflow-task-level_2_verbosity_try.json",
    },
}

# Model config (Bedrock)
MODEL = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
AWS_PROFILE = "infra-dev/Bedrock-Invoke-Only-Lila"
AWS_REGION = "us-east-1"

# Agent log directory name patterns
AGENT_LOG_DIRS = {
    "react": "agent_logs-ReActAgent-claude-sonnet-4-5-20250929-workflow",
    "toolcalling": "agent_logs-ToolCallingAgent-claude-sonnet-4-5-20250929-workflow",
}

# Run parameters
TEMPERATURE = 0.7
TOOL_VERBOSITY = "workflow"
MAX_ITERATIONS = 20
TRIALS_PER_CONDITION = 15
K_VALUES = list(range(1, 16))

# Intervention parameters
NUM_STEPS_VALUES = [1, 2, -2, -1]
CONDITIONS = ["baseline", "success", "failed"]

# Selected tasks per environment
# Tasks are selected to have MIXED results (some success, some failure)
# in at least one agent type under workflow verbosity.
# The select_traces.py script handles per-agent trace availability.
SELECTED_TASKS = {
    "spectra": [
        "22_22222_orgsyn_222_2222",  # ReAct 60%, TC 20% - both MIXED
        "10_15227_orgsyn_084_0011_sub2",  # ReAct 80%, TC 20% - both MIXED
        "10_15227_orgsyn_084_0317m",  # ReAct 20%, TC 0%  - ReAct only
        "77_77777_orgsyn_777_7777",  # ReAct 40%, TC 0%  - ReAct only
        "10_15227_orgsyn_084_0077",  # ReAct 100%, TC 40% - TC only
    ],
    "wetlab": [
        "qualysis_lvl2_01",  # ReAct 40%, TC 60% - both MIXED
        "qualysis_lvl2_07",  # ReAct 20%, TC 20% - both MIXED
        "qualysis_lvl2_08",  # ReAct 60%, TC 60% - both MIXED
        "qualysis_lvl2_09",  # ReAct 60%, TC 80% - both MIXED
        "qualysis_lvl2_10",  # ReAct 60%, TC 60% - both MIXED
    ],
    "resistor": [
        "task_2",  # ReAct 100%, TC 60% - TC only
        "task_3",  # ReAct 80%, TC 80%  - both MIXED
        "task_4",  # ReAct 80%, TC 20%  - both MIXED
        "task_5",  # ReAct 100%, TC 20% - TC only
    ],
    "ml": [
        "ml_oxides",  # ReAct 80%, TC 80% - both MIXED
        "ml_nitrides",  # ReAct 100%, TC 80% - TC MIXED
        "ml_sulphides",  # ReAct 100%, TC 80% - TC MIXED
    ],
    "catalyst": [
        "si_workflow",  # ReAct 100%, TC 100% - hope for failures with sampling
        "tio2_workflow",  # ReAct 100%, TC 100% - hope for failures with sampling
        "cu2o_workflow",  # ReAct 100%, TC 100% - hope for failures with sampling
    ],
    "retrosynthesis": [
        "make_3_lvl2",  # ReAct 20%, TC 40% - both MIXED
        "make_4_lvl2",  # ReAct 80%, TC 100% - ReAct MIXED
        "make_5_lvl2",  # ReAct 40%, TC 0% - ReAct MIXED
        "make_7_lvl2",  # ReAct 80%, TC 80% - both MIXED
        "make_8_lvl2",  # ReAct 0%, TC 40% - TC MIXED
    ],
    "md": [
        "silicon_melting_2",  # ReAct 60%, TC 40% - both MIXED
        "aluminum_surface_energy_2",  # ReAct 100%, TC 100% - hope for failures with sampling
        "na2sio3_quenching_1",  # ReAct 0%, TC 100% - TC only
    ],
}

# WandB config
WANDB_PROJECT = "corral_intervention"
WANDB_GROUP = "intervention_experiment"
