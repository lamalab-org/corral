"""Regression cases captured immediately before the reported solver failures."""
import json
from pathlib import Path

import numpy as np
import pytest

from wetlab.engine import ChemicalSystemSpec, WetlabEngine

CASES = json.loads((Path(__file__).parent / 'fixtures/solver_nonconvergence.json').read_text())


@pytest.mark.parametrize('case', CASES)
def test_stalled_equilibria_recover_and_conserve_elements(case):
    engine = WetlabEngine(ChemicalSystemSpec(elements=case['chemical_system']['elements']))
    solution = engine.solution({}, volume=case['volume'])
    solution.state.assign(engine.deserialize_chemical_state(case['initial_state']))
    initial_elements = np.asarray(solution.state.elementAmounts(), dtype=float).copy()
    solution.equilibrate()
    assert solution.is_equilibrated
    assert solution._eq_error < 4e-8
    np.testing.assert_allclose(np.asarray(solution.state.elementAmounts(), dtype=float), initial_elements, atol=1e-12, rtol=1e-12)
