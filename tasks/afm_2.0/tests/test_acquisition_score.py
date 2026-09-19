"""Hardware-independent tests for the acquisition scorer (requires NumPy)."""
import ast
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Load only the new pure scoring functions without importing Windows hardware SDKs.
module = ast.parse((ROOT / 'src/score.py').read_text())
functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and
             node.name in {'_close_record', '_ratio', '_acquisition_metrics', 'check_acquisition_function'}]
ns = dict(np=np, math=math, json=json, Path=Path,
          logger=SimpleNamespace(warning=lambda message: None))
exec(compile(ast.Module(body=functions, type_ignores=[]), 'score.py', 'exec'), ns)


class AcquisitionTests(unittest.TestCase):
    def test_all_environments_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            for file in sorted((ROOT / 'environments').glob('*/*.json')):
                task = json.loads(file.read_text())[0]
                config = task['scoring_params']
                sequence = config['acquisition_sequence']
                self.assertEqual(len(sequence), 1 if config['level'] == 1 else 3)
                report = json.loads(task['submission_format'])
                arrays = {}
                for i, (image, expected) in enumerate(zip(report['images'], sequence)):
                    path = Path(directory) / f'{i}.nid'
                    path.write_bytes(f'raw-{i}'.encode())
                    image['path'] = str(path)
                    shape = (expected['params']['lines_per_frame'], expected['params']['points_per_line'])
                    y, x = np.indices(shape)
                    height = ((x+y) % 2 * 2 - 1) * (i+1) * 1e-9
                    lateral = np.full(shape, float(i+1))
                    image['channels'] = dict(topography='Z-Axis', lateral_trace='Lateral', lateral_retrace='Lateral')
                    arrays[str(path)] = SimpleNamespace(data={'Image': {
                        'Forward': {'Z-Axis': height, 'Lateral': lateral},
                        'Backward': {'Lateral': -lateral}}})
                ns['read'] = lambda path: arrays[path]
                measured = []
                for image, expected in zip(report['images'], sequence):
                    metrics = ns['_acquisition_metrics'](image, expected, config['question'])
                    if 'ra_nm' in metrics:
                        self.assertAlmostEqual(metrics['ra_nm'], len(measured)+1, places=6)
                        self.assertAlmostEqual(metrics['rq_nm'], len(measured)+1, places=6)
                    if 'friction_mean_v' in metrics:
                        self.assertAlmostEqual(metrics['friction_mean_v'], len(measured)+1)
                        self.assertEqual(metrics['friction_std_v'], 0)
                    image.update(metrics)
                    measured.append(metrics)
                keys = [key for key in measured[0] if key != 'friction_std_v']
                if 'summary' in report:
                    for key in keys:
                        values = np.array([m[key] for m in measured])
                        mean, std = float(values.mean()), float(values.std(ddof=1))
                        report['summary'][key] = dict(mean=mean, sample_std=std)
                        if config['question'] != 10:
                            report['summary'][key]['cv_percent'] = 100*std/mean
                if 'percent_change' in report['images'][0]:
                    for image, metrics in zip(report['images'], measured):
                        image['percent_change'] = {key: 100*(metrics[key]-measured[1][key])/measured[1][key] for key in keys}
                if 'fit' in report:
                    x = np.array([5,10,20]); y = np.array([1,2,3])
                    slope, intercept = np.polyfit(x,y,1)
                    report['fit'] = dict(slope=float(slope), intercept=float(intercept), r_squared=float(1-np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)))
                score = ns['check_acquisition_function'](**config)
                with self.subTest(file=file):
                    self.assertEqual(score(json.dumps(report)), 1)
                    bad = copy.deepcopy(report); bad['images'][0]['physical_settings'] = {}
                    self.assertEqual(score(bad), 0)
                    bad = copy.deepcopy(report); bad['images'].append(bad['images'][0])
                    self.assertEqual(score(bad), 0)
                    bad = copy.deepcopy(report); bad['images'][0]['recorded_params']['times_per_line'] = 2
                    self.assertEqual(score(bad), 0)
                    if keys:
                        bad = copy.deepcopy(report); bad['images'][0][keys[0]] = float('nan')
                        self.assertEqual(score(bad), 0)
                    if len(sequence) == 3:
                        bad = copy.deepcopy(report); bad['images'].reverse()
                        self.assertEqual(score(bad), 0)
                        bad = copy.deepcopy(report); bad['images'][1]['path'] = bad['images'][0]['path']
                        self.assertEqual(score(bad), 0)

    def test_numeric_validation(self):
        close = ns['_close_record']
        for invalid in (True, '1', float('inf'), float('nan'), None):
            self.assertFalse(close(invalid, 1, .05))
        self.assertTrue(close(-10, -10, .05))
        self.assertTrue(close(None, None, .05))
        self.assertIsNone(ns['_ratio'](1, 0))


if __name__ == '__main__':
    unittest.main()
