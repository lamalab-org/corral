"""The bundled retrieval index must not depend on the launch directory."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tasks/afm/src/tool_utils.py"


@pytest.mark.parametrize("launch_dir", [ROOT, SOURCE.parent])
def test_retriever_opens_bundled_database(launch_dir, monkeypatch):
    monkeypatch.chdir(launch_dir)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    initializer = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "db_new" for t in node.targets)
    )
    calls = []
    namespace = {
        "__file__": str(SOURCE),
        "Path": Path,
        "embeddings": object(),
        "Chroma": lambda **kwargs: calls.append(kwargs),
    }
    exec(
        compile(ast.Module(body=[initializer], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    directory = Path(calls[0]["persist_directory"])
    assert directory == SOURCE.parent / "aila_db"
    assert (directory / "chroma.sqlite3").is_file()
