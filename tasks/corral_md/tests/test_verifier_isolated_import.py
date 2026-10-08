"""Exercise the worker's sibling imports under its deployed Python isolation."""

import os
import subprocess
import sys
from pathlib import Path


def test_isolated_hessian_postprocessing_uses_only_trusted_sibling_code(tmp_path):
    app_dir = Path(__file__).resolve().parents[1] / "modal_app"
    (tmp_path / "ground_truth.py").write_text(
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
    'postprocess': [{'symmetrization': 'average_transpose',
                     'acoustic_sum_rule': 'projection'}],
}
result = calculate(job, {}, Path('/unused'), {})
value = np.asarray(result['frames'][0]['hessian'])
assert np.allclose(value, value.T)
assert np.allclose(value @ np.tile(np.eye(3), (2, 1)), 0)
assert not np.allclose(value, raw)
assert Path(sys.modules['ground_truth'].__file__).parent == trusted
assert str(Path.cwd()) not in sys.path
print(json.dumps({'hessian_postprocessed': True}))
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
    assert '"hessian_postprocessed": true' in result.stdout
