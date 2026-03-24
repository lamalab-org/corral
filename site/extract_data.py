"""Extract environment data from Corral task sources and output data.js for the landing page.

Auto-discovers environments from tasks/*/environments/ directory structure.
Display names and descriptions are loaded from site/env_meta.json.
"""

import ast
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = Path(__file__).resolve().parent
TASKS_ROOT = ROOT / "tasks"
SKIP_ENVS = {"samplemath"}

# Verbosity tag regex (same as VerbosityConfig._keyword_regex)
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


def load_env_meta() -> dict[str, dict]:
    """Load display names and descriptions from env_meta.json."""
    meta_path = SITE_DIR / "env_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def discover_environments() -> dict[str, dict]:
    """Auto-discover environments from tasks/*/environments/ structure.

    Returns dict keyed by display_name with paths to tools.py, score.py, and dir name.
    """
    meta = load_env_meta()
    envs = {}

    for env_dir in sorted(TASKS_ROOT.iterdir()):
        if not env_dir.is_dir() or env_dir.name in SKIP_ENVS:
            continue
        if not (env_dir / "environments").is_dir():
            continue

        dir_name = env_dir.name
        env_meta = meta.get(dir_name, {})
        display_name = env_meta.get("display_name", dir_name.replace("_", " ").title())
        description = env_meta.get("description", "")

        # Auto-discover tools.py and score.py (exclude .venv and hidden dirs)
        tools_files = [
            p
            for p in env_dir.rglob("tools.py")
            if ".venv" not in p.parts
            and not any(part.startswith(".") for part in p.relative_to(env_dir).parts)
        ]
        score_files = [
            p
            for p in env_dir.rglob("score.py")
            if ".venv" not in p.parts
            and not any(part.startswith(".") for part in p.relative_to(env_dir).parts)
        ]

        envs[display_name] = {
            "dir": dir_name,
            "tools_py": tools_files[0] if tools_files else None,
            "score_py": score_files[0] if score_files else None,
            "description": description,
        }

    return envs


# ── AST extraction ──────────────────────────────────────────────────────


def extract_sections(docstring: str) -> dict[str, str]:
    """Extract all tagged sections from a docstring."""
    if not docstring:
        return {}
    sections = {}
    for match in TAG_RE.finditer(docstring):
        tag = match.group(1).upper()
        content = match.group(2).strip()
        if content:
            cleaned = re.sub(
                r"\[([A-Z_]+)\](.*?)\[/\1\]", r"\2", content, flags=re.DOTALL
            )
            sections[tag] = cleaned.strip()
    return sections


def extract_tools_from_file(filepath: Path | None) -> list[dict]:
    """Parse a tools.py file and extract @tool decorated functions."""
    if not filepath or not filepath.exists():
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

        args_info = []
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            type_str = ast.unparse(arg.annotation) if arg.annotation else ""
            args_info.append({"name": arg.arg, "type": type_str})

        return_type = ast.unparse(node.returns) if node.returns else ""

        # Code snippet: full function body (no docstring)
        body_nodes = node.body
        if (
            body_nodes
            and isinstance(body_nodes[0], ast.Expr)
            and isinstance(body_nodes[0].value, ast.Constant)
        ):
            body_start = (
                body_nodes[1].lineno - 1
                if len(body_nodes) > 1
                else (body_nodes[0].end_lineno or node.lineno)
            )
        else:
            body_start = body_nodes[0].lineno - 1 if body_nodes else node.lineno

        def_line = lines[node.lineno - 1]
        end_line = node.end_lineno or body_start + 20
        body_lines = lines[body_start:end_line]
        code_snippet = def_line + "\n" + "\n".join(body_lines)

        tools.append(
            {
                "name": node.name,
                "sections": sections,
                "args": args_info,
                "returns": return_type,
                "code": code_snippet,
            }
        )

    return tools


def extract_scoring_functions(filepath: Path | None) -> list[dict]:
    """Extract scoring function definitions from score.py."""
    if not filepath or not filepath.exists():
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


# ── Task loading ────────────────────────────────────────────────────────


def load_entries_from_dir(directory: Path) -> list[dict]:
    """Load all JSON files from a directory and return flattened entries."""
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
    """Load all tasks and subtasks for an environment."""
    env_root = TASKS_ROOT / env_dir_name / "environments"
    levels: dict[str, dict] = {}
    all_tasks: list[dict] = []
    all_subtasks: list[dict] = []

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
        raw_tasks = load_entries_from_dir(level_dir / "tasks_json")
        raw_subtasks = load_entries_from_dir(level_dir / "subtasks_json")

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


# ── Main build ──────────────────────────────────────────────────────────


def build_data():
    """Build the complete data structure for all environments."""
    environments = discover_environments()
    env_data = {}
    total_tasks = 0
    total_subtasks = 0
    total_tools = 0

    for display_name, config in environments.items():
        tools = extract_tools_from_file(config["tools_py"])
        scoring_fns = extract_scoring_functions(config["score_py"])
        task_data = load_tasks_and_subtasks(config["dir"])

        env_data[display_name] = {
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
            f"  {display_name}: {len(tools)} tools, "
            f"{task_data['task_count']} tasks, "
            f"{task_data['subtask_count']} subtasks, "
            f"{len(scoring_fns)} scoring fns, "
            f"{len(task_data['levels'])} levels"
        )

    log.info(
        f"\n  Totals: {total_tools} tools, {total_tasks} tasks, "
        f"{total_subtasks} subtasks"
    )
    return env_data


def to_js(data: dict) -> str:
    """Convert data dict to a JS file with const CORRAL_DATA = {...}."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return f"const CORRAL_DATA = {json_str};\n"


def main():
    log.info("Extracting Corral environment data...")
    data = build_data()

    output_path = SITE_DIR / "data.js"
    output_path.write_text(to_js(data))
    log.info(f"\nWrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
