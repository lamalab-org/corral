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
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retrosynthesis",
    "spectra": "Spectra",
    "wetlab": "Wetlab",
}

AGENT_NAMES = {
    "react": "React",
    "tool_calling": "Tool calling",
}

# ---------- Primary / secondary colour lists ----------
PRIMARY_COLOURS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
]

SECONDARY_COLOURS = [
    "#7c3aed",  # violet
    "#ea580c",  # orange
    "#0891b2",  # cyan
]

# ---------- Colours per category (ordered to match logical order) ----------
MODEL_COLOURS = [
    "#8D5F8C",  # Claude 4.5
    "#696FC7",  # GPT-4o
    "#4A90A4",  # GPT-OSS-120B
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
}
