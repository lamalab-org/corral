"""Extract environment data from Corral task sources and output data.js for the landing page."""

import ast
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Same regex as VerbosityConfig._keyword_regex
SUPPORTED_KEYWORDS = [
    "BRIEF",
    "DETAILED",
    "PROCEDURAL",
    "WORKFLOW_INTEGRATION",
    "CONTEXTUAL",
    "SYNTACTICAL",
    "RAISES",
    "LIMITATIONS",
    "EXAMPLES",
    "ARGS_BRIEF",
    "ARGS_DETAILED",
    "ARGS_SYNTACTICAL",
    "ARGS_EXAMPLES",
    "RETURNS_BRIEF",
    "RETURNS_DETAILED",
    "RETURNS_EXAMPLES",
]
_keyword_pattern = "|".join(re.escape(kw) for kw in SUPPORTED_KEYWORDS)
TAG_RE = re.compile(rf"\[({_keyword_pattern})\](.*?)\[/\1\]", re.DOTALL | re.IGNORECASE)

SKIP_ENVS = {"samplemath"}

ENVIRONMENTS = {
    "Catalyst": {
        "dir": "catalyst",
        "tools_py": "tasks/catalyst/src/catalyst/tools.py",
        "score_py": "tasks/catalyst/src/catalyst/score.py",
        "description": "Design and evaluate catalyst structures for CO2 adsorption on crystal slabs using Materials Project data and surface chemistry tools.",
    },
    "MD": {
        "dir": "corral_md",
        "tools_py": "tasks/corral_md/src/corral_md/tools.py",
        "score_py": "tasks/corral_md/src/corral_md/score.py",
        "description": "Run molecular dynamics simulations with LAMMPS to compute physical properties like diffusion coefficients, glass transition temperatures, and surface energies.",
    },
    "ML": {
        "dir": "ml",
        "tools_py": "tasks/ml/src/ml/tools.py",
        "score_py": "tasks/ml/src/ml/score.py",
        "description": "Train and evaluate machine learning models (XGBoost) on materials science datasets from the Materials Project.",
    },
    "Resistor": {
        "dir": "resistor_network",
        "tools_py": "tasks/resistor_network/src/resistor_network/tools.py",
        "score_py": "tasks/resistor_network/src/resistor_network/score.py",
        "description": "Infer resistor circuit topologies and values from node-to-node resistance measurements using circuit analysis tools.",
    },
    "Spectra": {
        "dir": "spectra_elucidation",
        "tools_py": "tasks/spectra_elucidation/spectra_elucidation/tools.py",
        "score_py": "tasks/spectra_elucidation/spectra_elucidation/score.py",
        "description": "Identify organic molecules from spectroscopic data (NMR, IR, mass spectrometry) through systematic spectra analysis.",
    },
    "Retrosynthesis": {
        "dir": "retrosynthesis",
        "tools_py": "tasks/retrosynthesis/retrosynthesis/tools.py",
        "score_py": "tasks/retrosynthesis/retrosynthesis/score.py",
        "description": "Plan retrosynthetic routes for target molecules using reaction template catalogs and chemical verification tools.",
    },
    "AFM": {
        "dir": "afm",
        "tools_py": "tasks/afm/src/tools.py",
        "score_py": "tasks/afm/src/score.py",
        "description": "Operate an atomic force microscope to perform surface scans and measure roughness with optimized scanning parameters.",
    },
}


def extract_sections(docstring: str) -> dict[str, str]:
    """Extract all tagged sections from a docstring using the same regex as VerbosityConfig."""
    if not docstring:
        return {}
    sections = {}
    for match in TAG_RE.finditer(docstring):
        tag = match.group(1).upper()
        content = match.group(2).strip()
        if content:
            # Clean nested tags
            cleaned = re.sub(
                r"\[([A-Z_]+)\](.*?)\[/\1\]", r"\2", content, flags=re.DOTALL
            )
            sections[tag] = cleaned.strip()
    return sections


def extract_tools_from_file(filepath: Path) -> list[dict]:
    """Parse a tools.py file using AST and extract @tool decorated functions."""
    if not filepath.exists():
        return []

    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    tools = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Check for @tool decorator
        has_tool = any(
            (isinstance(d, ast.Name) and d.id == "tool")
            or (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "tool"
            )
            for d in node.decorator_list
        )
        if not has_tool:
            continue

        docstring = ast.get_docstring(node) or ""
        sections = extract_sections(docstring)

        # Extract function signature
        args_info = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            type_str = ""
            if arg.annotation:
                type_str = ast.unparse(arg.annotation)
            args_info.append(
                {
                    "name": arg.arg,
                    "type": type_str,
                }
            )

        # Extract return type
        return_type = ""
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Get code snippet: signature + body without docstring
        body_nodes = node.body
        if (
            body_nodes
            and isinstance(body_nodes[0], ast.Expr)
            and isinstance(body_nodes[0].value, ast.Constant)
        ):
            if len(body_nodes) > 1:
                body_start = body_nodes[1].lineno - 1
            else:
                body_start = body_nodes[0].end_lineno or node.lineno
        else:
            body_start = body_nodes[0].lineno - 1 if body_nodes else node.lineno

        def_line = lines[node.lineno - 1]
        end_line = node.end_lineno or body_start + 20
        body_lines = lines[body_start:end_line]
        code_snippet = def_line + "\n" + "\n".join(body_lines)

        tool_data = {
            "name": node.name,
            "sections": sections,
            "args": args_info,
            "returns": return_type,
            "code": code_snippet,
        }
        tools.append(tool_data)

    return tools


def extract_scoring_functions(filepath: Path) -> list[dict]:
    """Extract scoring function definitions from score.py."""
    if not filepath.exists():
        return []

    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue

        docstring = ast.get_docstring(node) or ""
        start_line = node.lineno - 1
        end_line = node.end_lineno or start_line + 50
        code_lines = lines[start_line:end_line]
        code_snippet = "\n".join(code_lines)

        functions.append(
            {
                "name": node.name,
                "docstring": docstring[:300] if docstring else "",
                "code": code_snippet,
            }
        )

    return functions


def load_entries_from_dir(directory: Path) -> list[dict]:
    """Load all JSON files from a directory and return flattened standardized entries."""
    entries = []
    if not directory.is_dir():
        return entries

    for fpath in sorted(directory.glob("*.json")):
        data = json.loads(fpath.read_text())
        if isinstance(data, list):
            entries.extend(data)
        else:
            entries.append(data)

    return entries


def load_tasks_and_subtasks(env_dir_name: str) -> dict:
    """Load all tasks and subtasks for an environment from the standardized directory structure.

    Returns:
        {
            "levels": {
                "level_1": {"tasks": [...], "subtasks": [...]},
                "level_2": {"tasks": [...], "subtasks": [...]},
                ...
            },
            "all_tasks": [...],
            "all_subtasks": [...],
            "task_count": int,
            "subtask_count": int,
        }
    """
    env_root = ROOT / "tasks" / env_dir_name / "environments"
    levels = {}
    all_tasks = []
    all_subtasks = []

    if not env_root.is_dir():
        return {
            "levels": {},
            "all_tasks": [],
            "all_subtasks": [],
            "task_count": 0,
            "subtask_count": 0,
        }

    for level_dir in sorted(env_root.iterdir()):
        if not level_dir.is_dir() or not re.match(r"^level_\d+$", level_dir.name):
            continue

        level_name = level_dir.name
        tasks_dir = level_dir / "tasks_json"
        subtasks_dir = level_dir / "subtasks_json"

        raw_tasks = load_entries_from_dir(tasks_dir)
        raw_subtasks = load_entries_from_dir(subtasks_dir)

        # Normalize entries — _uid ensures uniqueness even when id repeats across files
        tasks = [
            {
                "_uid": e.get("uuid", f"{level_name}-task-{i}"),
                "id": e.get("id", e.get("name", "")),
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "tools": e.get("tools", []),
                "scoring_function": e.get("scoring_function", ""),
                "submission_format": e.get("submission_format", ""),
                "level": level_name,
            }
            for i, e in enumerate(raw_tasks)
        ]
        subtasks = [
            {
                "_uid": e.get("uuid", f"{level_name}-subtask-{i}"),
                "id": e.get("id", e.get("name", "")),
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "tools": e.get("tools", []),
                "scoring_function": e.get("scoring_function", ""),
                "submission_format": e.get("submission_format", ""),
                "level": level_name,
            }
            for i, e in enumerate(raw_subtasks)
        ]

        levels[level_name] = {"tasks": tasks, "subtasks": subtasks}
        all_tasks.extend(tasks)
        all_subtasks.extend(subtasks)

    return {
        "levels": levels,
        "all_tasks": all_tasks,
        "all_subtasks": all_subtasks,
        "task_count": len(all_tasks),
        "subtask_count": len(all_subtasks),
    }


def build_data():
    """Build the complete data structure for all environments."""
    env_data = {}
    total_tasks = 0
    total_subtasks = 0
    total_tools = 0

    for env_name, config in ENVIRONMENTS.items():
        tools_path = ROOT / config["tools_py"]
        score_path = ROOT / config["score_py"]

        tools = extract_tools_from_file(tools_path)
        scoring_fns = extract_scoring_functions(score_path)
        task_data = load_tasks_and_subtasks(config["dir"])

        env_data[env_name] = {
            "description": config["description"],
            "tools": tools,
            "tasks": task_data["all_tasks"],
            "subtasks": task_data["all_subtasks"],
            "levels": task_data["levels"],
            "task_count": task_data["task_count"],
            "subtask_count": task_data["subtask_count"],
            "scoring_functions": scoring_fns,
        }

        total_tasks += task_data["task_count"]
        total_subtasks += task_data["subtask_count"]
        total_tools += len(tools)

        log.info(
            f"  {env_name}: {len(tools)} tools, "
            f"{task_data['task_count']} tasks, "
            f"{task_data['subtask_count']} subtasks, "
            f"{len(scoring_fns)} scoring functions, "
            f"{len(task_data['levels'])} levels"
        )

    log.info(
        f"\n  Totals: {total_tools} tools, {total_tasks} tasks, {total_subtasks} subtasks"
    )
    return env_data


def to_js(data: dict) -> str:
    """Convert data dict to a JS file with const CORRAL_DATA = {...}."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return f"const CORRAL_DATA = {json_str};\n"


def main():
    log.info("Extracting Corral environment data...")
    data = build_data()

    output_path = Path(__file__).resolve().parent / "data.js"
    output_path.write_text(to_js(data))
    log.info(f"\nWrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
