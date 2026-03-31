"""
Name maps and colour palettes for Corral plots.
Models: claude-4.5, gpt-4o, gpt-oss-120b
Environments: afm, catalyst, md, ml, resistor, retro, spectra, wetlab
Agents: react, tool_calling
"""

# ---------- Name maps (id -> display name) ----------
MODEL_NAMES = {
    "claude-4.5": "Claude-4.5-Sonnet",
    "gpt-4o": "GPT-4o",
    "gpt-oss-120b": "GPT-OSS-120B",
}

ENVIRONMENT_NAMES = {
    "afm": "AFM Experiment\nExecution",
    "catalyst": "Adsorption Surface\nConstruction",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "resistor": "Circuit\nInference",
    "retro": "Retrosynthetic\nPlanning",
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
}

AGENT_NAMES = {
    "react": "ReAct",
    "tool_calling": "Tool calling",
}

# ---------- High-level environment grouping ----------
ENVIRONMENT_GROUPS = {
    "Hypothesis-driven inquiry": {
        "description": "Reason from observations to hidden structure",
        "environments": ["spectra", "wetlab", "resistor"],
    },
    "Strategic reasoning": {
        "description": "Navigate combinatorial spaces under constraints",
        "environments": ["retro", "afm"],
    },
    "Workflow construction": {
        "description": "Assemble and execute computational protocols",
        "environments": ["md", "catalyst", "ml"],
    },
}

# ---------- Environment max difficulty levels (S1, S2, ...) ----------
ENVIRONMENT_MAX_LEVELS = {
    "spectra": 2,
    "wetlab": 3,
    "resistor": 1,
    "retro": 3,
    "afm": 4,
    "md": 2,
    "catalyst": 1,
    "ml": 1,
}

# ---------- Default per-environment level selection ----------
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


# ---------- Colours per environment group ----------
GROUP_COLOURS = {
    "Hypothesis-driven inquiry": "#b000ff",
    "Strategic reasoning": "#fd00ff",
    "Workflow construction": "#0051ff",
}

# ---------- Primary / secondary colour lists ----------

GAP_COLORS = {
    "model_gap": "#7150e0",
    "agent_gap": "#e87584",
}


PRIMARY_COLOURS = [
    "#aae463",
    "#7150e0",
    "#e87584",
]

SECONDARY_COLOURS = [
    "#7c3aed",  # violet
    "#ea580c",  # orange
    "#0891b2",  # cyan
]

# ---------- Colours per category ----------
MODEL_COLOURS = {
    "claude-4.5": "#711c91",
    "gpt-4o": "#ea00d9",
    "gpt-oss-120b": "#7150e0",
}

AGENT_COLOURS = {
    "react": "#30292F",
    "tool_calling": "#5D737E",
}

ENVIRONMENT_COLOURS = {
    "spectra": "#7c3aed",
    "resistor": "#2563eb",
    "retro": "#0d9488",
    "catalyst": "#ca8a04",
    "afm": "#dc2626",
    "ml": "#16a34a",
    "md": "#0891b2",
}

# ---------- Font sizes ----------
FONT_SIZES = {
    "axis_label": 8,  # X and Y axis labels
    "tick_label": 8,  # X and Y tick labels
    "legend": 7,  # Legend text
    "title": 10,  # Plot title
}

# ---------- Single export dict (all of the above) ----------
PLOT_CONFIG = {
    "model_names": MODEL_NAMES,
    "environment_names": ENVIRONMENT_NAMES,
    "agent_names": AGENT_NAMES,
    "primary_colours": PRIMARY_COLOURS,
    "secondary_colours": SECONDARY_COLOURS,
    "model_colours": MODEL_COLOURS,
    "agent_colours": AGENT_COLOURS,
    "environment_colours": ENVIRONMENT_COLOURS,
    "environment_groups": ENVIRONMENT_GROUPS,
    "environment_max_levels": ENVIRONMENT_MAX_LEVELS,
    "default_env_level_map": DEFAULT_ENV_LEVEL_MAP,
    "font_sizes": FONT_SIZES,
}
