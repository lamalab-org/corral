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
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "Molecular\nDynamics",
    "ml": "Machine\nLearning",
    "resistor": "Resistor",
    "retro": "Retro-\nsynthesis",
    "spectra": "Spectra",
    "wetlab": "Wetlab",
}

AGENT_NAMES = {
    "react": "React",
    "tool_calling": "Tool calling",
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

# ---------- Colours per category (ordered to match logical order) ----------
MODEL_COLOURS = [
    "#711c91",  # Claude 4.5
    "#ea00d9",  # GPT-4o
    "#7150e0",  # GPT-OSS-120B
]

AGENT_COLOURS = [
    "#e11d48",  # React (rose)
    "#0d9488",  # Tool calling (teal)
]

ENVIRONMENT_COLOURS = [
    "#7c3aed",  # spectra
    "#2563eb",  # resistor
    "#0d9488",  # retro
    "#ca8a04",  # catalyst
    "#dc2626",  # afm
    "#16a34a",  # ml
    "#0891b2",  # md
]

# ---------- Optional: dict form for lookup by id ----------
MODEL_COLOUR_MAP = dict(zip(MODEL_NAMES.keys(), MODEL_COLOURS, strict=False))
AGENT_COLOUR_MAP = dict(zip(AGENT_NAMES.keys(), AGENT_COLOURS, strict=False))
ENVIRONMENT_COLOUR_MAP = dict(
    zip(ENVIRONMENT_NAMES.keys(), ENVIRONMENT_COLOURS, strict=False)
)

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
    "model_colour_map": MODEL_COLOUR_MAP,
    "agent_colour_map": AGENT_COLOUR_MAP,
    "environment_colour_map": ENVIRONMENT_COLOUR_MAP,
    "font_sizes": FONT_SIZES,
}
