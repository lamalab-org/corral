#!/usr/bin/env python3
"""Validate that all @tool-decorated functions have properly structured docstrings.

This script checks that every @tool function in the tasks/ directory follows
the tagged docstring format required by the verbosity system.

Can be run standalone or as a pre-commit hook.
Usage:
    python scripts/validate_tool_docstrings.py [files...]

If no files are given, scans all tasks/*/src/**/tools.py files.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tag definitions
# ---------------------------------------------------------------------------

MAIN_TAGS = [
    "BRIEF",
    "DETAILED",
    "PROCEDURAL",
    "WORKFLOW_INTEGRATION",
    "CONTEXTUAL",
    "SYNTACTICAL",
    "RAISES",
    "LIMITATIONS",
]

WORKFLOW_NESTED_TAGS = [
    "PREREQUISITE",
    "CURRENT",
    "FOLLOW_UP",
]

RAISES_NESTED_TAGS = [
    "ERROR_WHEN",
    "ERROR_DETAILS",
    "ERROR_RECOVERY",
]

ARGS_TAGS = [
    "ARGS_BRIEF",
    "ARGS_DETAILED",
    "ARGS_SYNTACTICAL",
    "ARGS_EXAMPLES",
]

RETURNS_TAGS = [
    "RETURNS_BRIEF",
    "RETURNS_DETAILED",
    "RETURNS_EXAMPLES",
]

ALL_KNOWN_TAGS = (
    MAIN_TAGS + WORKFLOW_NESTED_TAGS + RAISES_NESTED_TAGS + ARGS_TAGS + RETURNS_TAGS
)

# Common misspellings / variant spellings that should be flagged
COMMON_MISSPELLINGS: dict[str, str] = {
    "ARGS_SYNTACTIC": "ARGS_SYNTACTICAL",
    "RETURNS_SYNTACTIC": "RETURNS_SYNTACTICAL",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_tag(docstring: str, tag: str) -> bool:
    """Return True if [TAG]...[/TAG] is present (case-insensitive)."""
    pattern = rf"\[{re.escape(tag)}\].*?\[/{re.escape(tag)}\]"
    return bool(re.search(pattern, docstring, re.DOTALL | re.IGNORECASE))


def _find_unclosed_tags(docstring: str) -> list[str]:
    """Return list of tags that are opened but not closed."""
    unclosed = []
    tags_to_check = ALL_KNOWN_TAGS + list(COMMON_MISSPELLINGS.keys())
    for tag in tags_to_check:
        opening = len(re.findall(rf"\[{re.escape(tag)}\]", docstring, re.IGNORECASE))
        closing = len(re.findall(rf"\[/{re.escape(tag)}\]", docstring, re.IGNORECASE))
        if opening > closing:
            unclosed.append(tag)
    return unclosed


def _has_args_section(docstring: str) -> bool:
    return bool(re.search(r"^\s*Args:", docstring, re.MULTILINE))


def _has_returns_section(docstring: str) -> bool:
    return bool(re.search(r"^\s*Returns:", docstring, re.MULTILINE))


def _extract_section(docstring: str, header: str) -> str | None:
    """Extract the content of an Args: or Returns: block."""
    pattern = rf"{header}:(.*?)(?=\n\s*(?:Args|Returns|Raises|Examples):|$)"
    m = re.search(pattern, docstring, re.DOTALL)
    return m.group(1) if m else None


def _count_documented_args(docstring: str) -> list[str]:
    """Return list of argument names documented in the Args: block."""
    section = _extract_section(docstring, "Args")
    if not section:
        return []
    arg_pattern = re.compile(r"^\s+([a-zA-Z_]\w*)\s*(?:\([^)]*\))?\s*:", re.MULTILINE)
    return arg_pattern.findall(section)


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


class Violation:
    def __init__(self, file: str, func: str, line: int, message: str):
        self.file = file
        self.func = func
        self.line = line
        self.message = message

    def __str__(self) -> str:
        return f"{self.file}:{self.line} ({self.func}): {self.message}"


def _get_tool_functions(filepath: Path) -> list[tuple[str, int, str, ast.FunctionDef]]:
    """Parse a file and return (name, lineno, docstring, node) for @tool functions."""
    source = filepath.read_text()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Check if decorated with @tool (plain or with args)
        for dec in node.decorator_list:
            is_tool = False
            if isinstance(dec, ast.Name) and dec.id == "tool":
                is_tool = True
            elif isinstance(dec, ast.Call):
                func_node = dec.func
                if isinstance(func_node, ast.Name) and func_node.id == "tool":
                    is_tool = True
            if is_tool:
                docstring = ast.get_docstring(node, clean=False) or ""
                results.append((node.name, node.lineno, docstring, node))
                break
    return results


def _get_hidden_args(node: ast.FunctionDef) -> set[str]:
    """Extract hidden_args from @tool(hidden_args=[...]) if present."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg == "hidden_args" and isinstance(kw.value, ast.List):
                    return {
                        elt.value
                        for elt in kw.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    return set()


def _get_func_params(node: ast.FunctionDef) -> list[str]:
    """Return the parameter names (excluding self)."""
    return [arg.arg for arg in node.args.args if arg.arg != "self"]


def validate_tool_docstring(
    filepath: str,
    func_name: str,
    lineno: int,
    docstring: str,
    node: ast.FunctionDef,
) -> list[Violation]:
    """Validate a single @tool function's docstring."""
    violations: list[Violation] = []

    def v(msg: str) -> None:
        violations.append(Violation(filepath, func_name, lineno, msg))

    if not docstring.strip():
        v("Missing docstring")
        return violations

    # 0. Check common misspellings
    for wrong, correct in COMMON_MISSPELLINGS.items():
        if _has_tag(docstring, wrong):
            v(f"Misspelled tag [{wrong}] — should be [{correct}]")

    # 1. Check unclosed tags
    unclosed = _find_unclosed_tags(docstring)
    if unclosed:
        v(f"Unclosed tags: {', '.join(unclosed)}")

    # 2. Check main tags
    for tag in MAIN_TAGS:
        if not _has_tag(docstring, tag):
            v(f"Missing main tag: [{tag}]")

    # 3. Check WORKFLOW_INTEGRATION nested tags
    if _has_tag(docstring, "WORKFLOW_INTEGRATION"):
        wi_content = re.search(
            r"\[WORKFLOW_INTEGRATION\](.*?)\[/WORKFLOW_INTEGRATION\]",
            docstring,
            re.DOTALL | re.IGNORECASE,
        )
        if wi_content:
            wi_text = wi_content.group(1)
            for tag in WORKFLOW_NESTED_TAGS:
                if not _has_tag(wi_text, tag):
                    v(f"Missing nested tag inside [WORKFLOW_INTEGRATION]: [{tag}]")

    # 4. Check RAISES nested tags
    if _has_tag(docstring, "RAISES"):
        raises_content = re.search(
            r"\[RAISES\](.*?)\[/RAISES\]",
            docstring,
            re.DOTALL | re.IGNORECASE,
        )
        if raises_content:
            raises_text = raises_content.group(1)
            for tag in RAISES_NESTED_TAGS:
                if not _has_tag(raises_text, tag):
                    v(f"Missing nested tag inside [RAISES]: [{tag}]")

    # 5. Check Args section
    hidden_args = _get_hidden_args(node)
    func_params = _get_func_params(node)
    visible_params = [p for p in func_params if p not in hidden_args]

    if visible_params:
        if not _has_args_section(docstring):
            v("Missing Args: section (function has visible parameters)")
        else:
            args_section = _extract_section(docstring, "Args")
            if args_section:
                documented = _count_documented_args(docstring)
                for param in visible_params:
                    if param not in documented:
                        v(f"Parameter '{param}' not documented in Args section")

                # Check ARGS_* tags for each documented argument
                for arg_name in documented:
                    # Extract the block for this argument
                    # Find content from "arg_name ...:" to next arg or end of section
                    arg_block_pattern = rf"({re.escape(arg_name)}\s*(?:\([^)]*\))?\s*:.*?)(?=\n\s+[a-zA-Z_]\w*\s*(?:\([^)]*\))?\s*:|$)"
                    arg_block_match = re.search(
                        arg_block_pattern, args_section, re.DOTALL
                    )
                    if arg_block_match:
                        arg_text = arg_block_match.group(1)
                        for tag in ARGS_TAGS:
                            if not _has_tag(arg_text, tag):
                                v(f"Argument '{arg_name}' missing tag: [{tag}]")

    # 6. Check Returns section
    if _has_returns_section(docstring):
        returns_section = _extract_section(docstring, "Returns")
        if returns_section:
            for tag in RETURNS_TAGS:
                if not _has_tag(returns_section, tag):
                    v(f"Missing tag in Returns section: [{tag}]")

    return violations


# ---------------------------------------------------------------------------
# File discovery & main
# ---------------------------------------------------------------------------


def find_tool_files(repo_root: Path) -> list[Path]:
    """Find all tools.py files under tasks/ and src/corral/utils/."""
    files: list[Path] = []

    tasks_dir = repo_root / "tasks"
    if tasks_dir.exists():
        files.extend(tasks_dir.rglob("**/tools.py"))

    utils_dir = repo_root / "src" / "corral" / "utils"
    if utils_dir.exists():
        files.extend(utils_dir.glob("*_tools.py"))

    return sorted(files)


def validate_file(filepath: Path) -> list[Violation]:
    """Validate all @tool functions in a single file."""
    all_violations: list[Violation] = []
    for func_name, lineno, docstring, node in _get_tool_functions(filepath):
        all_violations.extend(
            validate_tool_docstring(str(filepath), func_name, lineno, docstring, node)
        )
    return all_violations


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    # Determine repo root (script lives in scripts/)
    repo_root = Path(__file__).resolve().parent.parent

    if args:
        # Pre-commit mode: only check supplied files
        files = [Path(f) for f in args if f.endswith("tools.py")]
    else:
        # Standalone mode: scan all task tool files
        files = find_tool_files(repo_root)

    if not files:
        return 0

    all_violations: list[Violation] = []
    for f in files:
        if not f.exists():
            continue
        all_violations.extend(validate_file(f))

    if all_violations:
        lines: list[str] = []
        lines.append(f"\n{'='*70}")
        lines.append(
            f"  Tool Docstring Validation: {len(all_violations)} issue(s) found"
        )
        lines.append(f"{'='*70}\n")

        # Group by file
        by_file: dict[str, list[Violation]] = {}
        for v in all_violations:
            by_file.setdefault(v.file, []).append(v)

        for filepath, violations in by_file.items():
            try:
                rel = str(Path(filepath).resolve().relative_to(repo_root))
            except ValueError:
                rel = filepath
            lines.append(f"  {rel}")
            # Group by function
            by_func: dict[str, list[Violation]] = {}
            for v in violations:
                by_func.setdefault(v.func, []).append(v)
            for func_name, func_violations in by_func.items():
                lines.append(f"    {func_name} (line {func_violations[0].line}):")
                lines.extend(f"      - {v.message}" for v in func_violations)
            lines.append("")

        lines.append(f"{'='*70}")
        lines.append(
            "  Fix the above issues to ensure proper verbosity-level filtering."
        )
        lines.append(f"{'='*70}\n")
        sys.stdout.write("\n".join(lines) + "\n")
        return 1

    sys.stdout.write("All @tool docstrings are valid.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
