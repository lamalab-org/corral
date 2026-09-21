from __future__ import annotations

import ast
from pathlib import Path


def test_lammps_modal_app_exports_simulation_workers() -> None:
    task_root = Path(__file__).resolve().parents[1]
    app_tree = ast.parse((task_root / "modal_app/lammps_app.py").read_text())

    app_functions = {
        node.name
        for node in app_tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert {"prepare_workspace", "run_lammps", "run_python_gpu"} <= app_functions


def test_evaluation_workers_cap_parallel_modal_calls() -> None:
    task_root = Path(__file__).resolve().parents[1]
    app_tree = ast.parse((task_root / "modal_app/lammps_app.py").read_text())
    limit = next(
        ast.literal_eval(node.value)
        for node in app_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MAX_EVALUATION_CONTAINERS"
            for target in node.targets
        )
    )

    limits = {}
    for node in app_tree.body:
        if not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ) or node.name not in {
            "verify_calculations",
            "verify_md_provenance",
        }:
            continue
        decorator = next(
            item
            for item in node.decorator_list
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "function"
        )
        value = next(
            keyword.value
            for keyword in decorator.keywords
            if keyword.arg == "max_containers"
        )
        assert isinstance(value, ast.Name)
        assert value.id == "MAX_EVALUATION_CONTAINERS"
        limits[node.name] = limit

    assert limits == {
        "verify_calculations": 25,
        "verify_md_provenance": 25,
    }


def test_md_tools_use_the_structured_docstring_format() -> None:
    task_root = Path(__file__).resolve().parents[1]
    tools_tree = ast.parse((task_root / "src/corral_md/tools.py").read_text())
    required_sections = {
        "[BRIEF]",
        "[DETAILED]",
        "[PROCEDURAL]",
        "[WORKFLOW_INTEGRATION]",
        "[CONTEXTUAL]",
        "[SYNTACTICAL]",
        "Args:",
        "Returns:",
        "[RAISES]",
        "[LIMITATIONS]",
    }
    tool_functions = [
        node
        for node in ast.walk(tools_tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            (isinstance(decorator, ast.Name) and decorator.id == "tool")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "tool"
            )
            for decorator in node.decorator_list
        )
    ]

    assert tool_functions
    for function in tool_functions:
        docstring = ast.get_docstring(function) or ""
        missing = required_sections - {
            section for section in required_sections if section in docstring
        }
        assert not missing, f"{function.name} is missing {sorted(missing)}"
