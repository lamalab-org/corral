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

ENVIRONMENTS = {
    "Catalyst": {
        "tools_py": "tasks/catalyst/src/catalyst/tools.py",
        "score_py": "tasks/catalyst/src/catalyst/score.py",
        "task_configs": [("single", "tasks/catalyst/config/single/single.json")],
        "description": "Design and evaluate catalyst structures for CO2 adsorption on crystal slabs using Materials Project data and surface chemistry tools.",
    },
    "MD": {
        "tools_py": "tasks/corral_md/src/corral_md/tools.py",
        "score_py": "tasks/corral_md/src/corral_md/score.py",
        "task_configs": [
            ("melting", "tasks/corral_md/environments/melting/level_1/tasks/si.json"),
            (
                "quenching",
                "tasks/corral_md/environments/quenching/level_1/tasks/na2sio3.json",
            ),
            (
                "surface_energy",
                "tasks/corral_md/environments/surface_energy/level_1/tasks/al.json",
            ),
        ],
        "description": "Run molecular dynamics simulations with LAMMPS to compute physical properties like diffusion coefficients, glass transition temperatures, and surface energies.",
    },
    "ML": {
        "tools_py": "tasks/ml/src/ml/tools.py",
        "score_py": "tasks/ml/src/ml/score.py",
        "task_configs": [("single", "tasks/ml/config/single/single.json")],
        "description": "Train and evaluate machine learning models (XGBoost) on materials science datasets from the Materials Project.",
    },
    "Resistor": {
        "tools_py": "tasks/resistor_network/src/resistor_network/tools.py",
        "score_py": "tasks/resistor_network/src/resistor_network/score.py",
        "task_configs": [
            ("single", "tasks/resistor_network/config/single/single.json")
        ],
        "description": "Infer resistor circuit topologies and values from node-to-node resistance measurements using circuit analysis tools.",
    },
    "Spectra": {
        "tools_py": "tasks/spectra_elucidation/spectra_elucidation/tools.py",
        "score_py": "tasks/spectra_elucidation/spectra_elucidation/score.py",
        "task_configs": [
            (
                "level_1",
                "tasks/spectra_elucidation/environments/level_1/tasks_json/task_1.json",
            ),
            (
                "level_2",
                "tasks/spectra_elucidation/environments/level_2/tasks_json/task_1.json",
            ),
        ],
        "description": "Identify organic molecules from spectroscopic data (NMR, IR, mass spectrometry) through systematic spectra analysis.",
    },
    "Retrosynthesis": {
        "tools_py": "tasks/retrosynthesis/retrosynthesis/tools.py",
        "score_py": "tasks/retrosynthesis/retrosynthesis/score.py",
        "task_configs": [
            ("level_1", "tasks/retrosynthesis/environments/level_1/tasks/make_1.json"),
            ("level_2", "tasks/retrosynthesis/environments/level_2/tasks/make_1.json"),
        ],
        "description": "Plan retrosynthetic routes for target molecules using reaction template catalogs and chemical verification tools.",
    },
    "AFM": {
        "tools_py": "tasks/afm/src/tools.py",
        "score_py": "tasks/afm/src/score.py",
        "task_configs": [
            ("level_1", "tasks/afm/src/enviroment/tasks_1.json"),
            ("level_2", "tasks/afm/src/enviroment/tasks_2.json"),
        ],
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
        # Find where the body starts (after docstring)
        body_nodes = node.body
        if (
            body_nodes
            and isinstance(body_nodes[0], ast.Expr)
            and isinstance(body_nodes[0].value, ast.Constant)
        ):
            # First body node is docstring, skip it
            if len(body_nodes) > 1:
                body_start = body_nodes[1].lineno - 1
            else:
                body_start = body_nodes[0].end_lineno or node.lineno
        else:
            body_start = body_nodes[0].lineno - 1 if body_nodes else node.lineno

        # Build code: decorator + def line + body (no docstring)
        def_line = lines[node.lineno - 1]
        end_line = node.end_lineno or body_start + 20
        body_lines = lines[body_start:end_line]
        # Limit body to ~20 lines
        if len(body_lines) > 20:
            body_lines = body_lines[:20]
            body_lines.append("    ...")
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
        # Skip private/helper functions
        if node.name.startswith("_"):
            continue

        docstring = ast.get_docstring(node) or ""
        start_line = node.lineno - 1
        end_line = min(node.end_lineno or start_line + 25, start_line + 25)
        code_lines = lines[start_line:end_line]
        code_snippet = "\n".join(code_lines)
        if end_line < (node.end_lineno or 0):
            code_snippet += "\n    ..."

        functions.append(
            {
                "name": node.name,
                "docstring": docstring[:300] if docstring else "",
                "code": code_snippet,
            }
        )

    return functions


def load_task_config(filepath: Path) -> list[dict]:
    """Load and parse a task JSON config file."""
    if not filepath.exists():
        return []

    data = json.loads(filepath.read_text())

    tasks = []
    # Handle both dict-of-tasks and list-of-tasks formats
    if isinstance(data, dict):
        for key, val in data.items():
            tasks.append(
                {
                    "id": key,
                    "name": val.get("name", key),
                    "description": val.get("description", ""),
                    "tools": val.get("tools", []),
                    "scoring_function": val.get(
                        "scoring_function", val.get("scoring_fn", "")
                    ),
                    "submission_format": val.get("submission_format", ""),
                }
            )
    elif isinstance(data, list):
        tasks.extend(
            {
                "id": item.get("id", item.get("name", "")),
                "name": item.get("name", ""),
                "description": item.get("input", {}).get("prompt", ""),
                "tools": item.get("tools", []),
                "scoring_function": item.get(
                    "scoring_fn", item.get("scoring_function", "")
                ),
                "submission_format": item.get("submission_format", ""),
            }
            for item in data
        )

    return tasks


def build_data():
    """Build the complete data structure for all environments."""
    env_data = {}

    for env_name, config in ENVIRONMENTS.items():
        tools_path = ROOT / config["tools_py"]
        score_path = ROOT / config["score_py"]

        tools = extract_tools_from_file(tools_path)
        scoring_fns = extract_scoring_functions(score_path)

        all_tasks = []
        for level_name, config_path in config["task_configs"]:
            tasks = load_task_config(ROOT / config_path)
            for t in tasks:
                t["level"] = level_name
            all_tasks.extend(tasks)

        env_data[env_name] = {
            "description": config["description"],
            "tools": tools,
            "tasks": all_tasks,
            "scoring_functions": scoring_fns,
        }

        log.info(
            f"  {env_name}: {len(tools)} tools, {len(all_tasks)} tasks, {len(scoring_fns)} scoring functions"
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
