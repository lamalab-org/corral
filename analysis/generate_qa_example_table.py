"""Generate LaTeX tables with example QA questions from jablonkagroup/corral-QAs.

Produces two tables (knowledge-based and reasoning-based), each with columns
'Environment' and 'Question'. For each environment the shortest question
(including concatenated answer options) is selected so the tables fit on one page.

Output: analysis/results/tables/qa_example_tables.tex
"""

import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

DATASET_ID = "jablonkagroup/corral-QAs"

CONFIGS = [
    "afm_qa",
    "afm_reasoning_qa",
    "catalyst_qa",
    "catalyst_reasoning_qa",
    "md_qa",
    "md_reasoning_qa",
    "ml_qa",
    "ml_reasoning_qa",
    "resistor_qa",
    "resistor_reasoning_qa",
    "retro_qa",
    "retro_reasoning_qa",
    "spectra_qa",
    "spectra_reasoning_qa",
    "wetlab_qa",
    "wetlab_reasoning_qa",
]

ENV_LABELS = {
    "afm": "AFM experiment execution",
    "catalyst": "Adsorption surface construction",
    "md": "Molecular simulation",
    "ml": "ML-based property prediction",
    "resistor": "Circuit inference",
    "retro": "Retrosynthetic planning",
    "spectra": "Spectroscopic structure elucidation",
    "wetlab": "Inorganic qualitative analysis",
}

QA_TYPE_LABELS = {
    "qa": "Knowledge-based questions",
    "reasoning_qa": "Reasoning-based questions",
}


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "tables"


def parse_config(config_name: str) -> tuple[str, str]:
    """Return (env, qa_type) from a config name like 'afm_qa' or 'md_reasoning_qa'."""
    if config_name.endswith("_reasoning_qa"):
        env = config_name[: -len("_reasoning_qa")]
        return env, "reasoning_qa"
    # ends with _qa
    env = config_name[: -len("_qa")]
    return env, "qa"


MAX_OPTION_CHARS = 280  # truncate individual answer options longer than this


def _truncate(text: str, limit: int) -> str:
    """Truncate *text* at the last space before *limit* and append '...'."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind(" ")
    if cut <= 0:
        cut = limit
    return text[:cut] + "..."


def build_question_text(example: dict) -> tuple[str, str]:
    """Concatenate question text with answer options labelled A., B., etc.

    Returns (full_text, correct_letter) where correct_letter is e.g. 'A'.
    """
    question = example["input"].strip()
    scores = json.loads(example["target_scores"])
    options = list(scores.keys())
    correct_letter = "?"
    for i, (_opt, score) in enumerate(scores.items()):
        if score == 1:
            correct_letter = chr(65 + i)
            break
    labelled = [
        f"{chr(65 + i)}. {_truncate(opt, MAX_OPTION_CHARS)}"
        for i, opt in enumerate(options)
    ]
    options_str = "  ".join(labelled)
    return f"{question} {options_str}", correct_letter


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Replace newlines with spaces
    return text.replace("\n", " ")


def main() -> None:
    # Collect all questions: {qa_type: {env: [(full_text, correct_letter), ...]}}
    questions: dict[str, dict[str, list[tuple[str, str]]]] = {
        "qa": {},
        "reasoning_qa": {},
    }

    for config_name in CONFIGS:
        env, qa_type = parse_config(config_name)
        ds = load_dataset(DATASET_ID, config_name, split="train")

        entries: list[tuple[str, str]] = []
        for row in ds:
            for example in row["examples"]:
                text, letter = build_question_text(example)
                entries.append((text, letter))

        questions[qa_type][env] = entries

    # For each qa_type and env, keep the shortest question
    shortest: dict[str, dict[str, tuple[str, str]]] = {"qa": {}, "reasoning_qa": {}}
    for qa_type in questions:
        for env, entries in questions[qa_type].items():
            shortest[qa_type][env] = min(entries, key=lambda e: len(e[0]))

    # Build LaTeX
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env_order = [
        "afm",
        "catalyst",
        "md",
        "ml",
        "resistor",
        "retro",
        "spectra",
        "wetlab",
    ]

    for qa_type in QA_TYPE_LABELS:
        lines: list[str] = []
        lines.append(r"\begin{tabularx}{\textwidth}{p{2.5cm} X c}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Environment} & \textbf{Question} & \textbf{Answer} \\")
        lines.append(r"\midrule")

        for env in env_order:
            if env not in shortest[qa_type]:
                continue
            env_label = escape_latex(ENV_LABELS.get(env, env))
            question_text, correct = shortest[qa_type][env]
            question = escape_latex(question_text)
            lines.append(f"{env_label} & {question} & {correct} \\\\")
            lines.append(r"\addlinespace")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabularx}")

        out_file = OUTPUT_DIR / f"qa_example_{qa_type}.tex"
        out_file.write_text("\n".join(lines))
        n = sum(1 for e in env_order if e in shortest[qa_type])
        total = sum(
            len(shortest[qa_type][e][0]) for e in env_order if e in shortest[qa_type]
        )
        logger.info(f"Written {out_file}  ({n} envs, ~{total} chars)")


if __name__ == "__main__":
    main()
