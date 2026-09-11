from __future__ import annotations

import ast
from pathlib import Path


def test_lammps_modal_app_exports_only_run_lammps() -> None:
    task_root = Path(__file__).resolve().parents[1]
    app_tree = ast.parse((task_root / "modal_app/lammps_app.py").read_text())

    app_functions = {
        node.name
        for node in app_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert app_functions == {"run_lammps"}
