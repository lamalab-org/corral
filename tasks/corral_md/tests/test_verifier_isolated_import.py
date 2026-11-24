"""Exercise raw worker calculations under the deployed Python isolation."""

import os
import subprocess
import sys
from pathlib import Path


def test_isolated_worker_returns_raw_hessian_without_importing_workspace_code(tmp_path):
    app_dir = Path(__file__).resolve().parents[1] / "modal_app"
    (tmp_path / "ase.py").write_text(
        "raise RuntimeError('must not import submitted code')\n"
    )
    script = """
import json, runpy, sys
from pathlib import Path
import numpy as np
trusted = Path(sys.argv[1])
worker = runpy.run_path(str(trusted / 'verification_worker.py'))
calculate = worker['calculate']
raw = np.eye(6) * 10 + np.arange(36).reshape(6, 6)
class Calculator:
    def get_hessian(self, *, atoms):
        return raw
calculate.__globals__['mace_calculator'] = lambda *args, **kwargs: Calculator()
job = {
    'operation': 'mace', 'parameters': {}, 'properties': ['hessian'],
    'frames': [{'numbers': [14, 14], 'positions': [[0, 0, 0], [1, 1, 1]],
                'cell': np.eye(3).tolist(), 'pbc': [False] * 3}],
}
result = calculate(job, {}, Path('/unused'), {})
value = np.asarray(result['frames'][0]['hessian'])
assert np.array_equal(value, raw)
assert not Path(sys.modules['ase'].__file__).is_relative_to(Path.cwd())
assert str(Path.cwd()) not in sys.path
print(json.dumps({'raw_hessian_preserved': True}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(app_dir)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert '"raw_hessian_preserved": true' in result.stdout
