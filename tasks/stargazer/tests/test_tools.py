from __future__ import annotations

import json

import numpy as np
import pytest
from stargazer.tools import create_analysis_session, create_tools


@pytest.fixture
def analysis_session(simple_task):
    session = create_analysis_session(
        **vars(simple_task.observations),
        star_mass_sun=simple_task.star_mass_sun,
    )
    yield session
    session.close()


def test_original_tool_schemas_are_preserved(protocol_reference):
    actual = list(create_tools().values())
    assert [tool.name for tool in actual] == ["PythonREPL", "submit_action"]
    formats = [tool.get_openai_tool_format() for tool in actual]
    # The REPL description now advertises unrestricted Python and no deadline;
    # its input schema and the entire submission tool remain unchanged.
    expected_repl = protocol_reference["tools"][0]
    assert formats[0] == {
        **expected_repl,
        "function": {
            **expected_repl["function"],
            "description": actual[0].description,
        },
    }
    assert formats[1] == protocol_reference["tools"][1]
    assert all(not tool.hidden_args for tool in actual)


def test_repl_and_checkpoints_match_original_reference(
    analysis_session, protocol_reference
):
    for step in protocol_reference["repl"]:
        actual = analysis_session.execute(step["code"])
        expected = step["output"]
        if isinstance(expected, float):
            # Numerical fits can differ in the last digits across platforms.
            actual = float(actual)
            expected = pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert actual == expected, step["code"]
        analysis_session.restore(analysis_session.snapshot())
    assert analysis_session.protocol_acknowledged()


def test_helper_clamps_and_converts_inside_repl(analysis_session):
    result = analysis_session.execute(
        "import json\nprint(json.dumps(stargazer_planet_from_fit(17.25, -4, e=2, inc_rad=-2, Omega_rad=8, M0_rad=7)))"
    )
    planet = json.loads(result)
    assert planet["m_sin_i_mjup"] == 0.001
    assert planet["e"] == 0.8
    assert planet["inc_rad"] == 0
    assert planet["Omega_rad"] == pytest.approx(8 % (2 * np.pi))
    assert planet["l_rad"] == pytest.approx(15 % (2 * np.pi))


def test_namespace_matches_upstream_public_surface(analysis_session):
    assert analysis_session.execute("history").strip() == "[]"
    for name in (
        "benchmark_task",
        "StargazerTask",
        "instruments",
        "scipy",
        "optimize",
        "signal",
    ):
        assert "NameError" in analysis_session.execute(name)


def test_analysis_checkpoint_preserves_random_state(analysis_session):
    analysis_session.execute("np.random.seed(123)\nrng = np.random.default_rng(456)")
    checkpoint = analysis_session.snapshot()
    code = "print((np.random.random(3).tolist(), rng.random(3).tolist()))"
    expected = analysis_session.execute(code)
    analysis_session.restore(checkpoint)
    assert analysis_session.execute(code) == expected


def test_analysis_checkpoint_preserves_shared_recursive_closures(analysis_session):
    analysis_session.execute("""def counter():
    count = 0
    def advance():
        nonlocal count
        count += 1
        return count
    def read():
        return count
    return advance, read
advance, read = counter()
def factorial_factory():
    def factorial(n):
        return n * factorial(n - 1) if n else 1
    return factorial
fact_callable = factorial_factory()""")
    analysis_session.restore(analysis_session.snapshot())
    assert (
        analysis_session.execute("print((advance(), read(), fact_callable(5)))").strip()
        == "(1, 1, 120)"
    )


def test_repl_supports_normal_python_and_checkpointed_functions(analysis_session):
    result = analysis_session.execute("""import pathlib
import sys
import types
import baselines
from collections import Counter
from scipy.optimize import minimize
class FitResult:
    def __init__(self, value):
        self.value = value
fit = FitResult(42)
def describe():
    return type(fit).__name__, getattr(fit, "value"), pathlib.Path(".").is_dir()
print((isinstance(np, types.ModuleType), callable(minimize), describe()))""")
    assert result.strip() == "(True, True, ('FitResult', 42, True))"
    analysis_session.restore(analysis_session.snapshot())
    assert analysis_session.execute("print(describe())").strip() == (
        "('FitResult', 42, True)"
    )
    assert analysis_session.execute("print(np.__name__)").strip() == "numpy"
    assert analysis_session.execute("print(baselines.__name__)").strip() == "baselines"


def test_repl_supports_files_and_subprocesses(analysis_session):
    result = analysis_session.execute("""from pathlib import Path
import subprocess
import sys
np.save("fit.npy", np.arange(3))
np.savetxt("fit.txt", np.arange(3))
with open("note.txt", "w") as note:
    note.write("fit complete")
child = subprocess.run([sys.executable, "-c", "print(6 * 7)"], capture_output=True, text=True, check=True)
print((np.load("fit.npy").tolist(), np.loadtxt("fit.txt").tolist(), Path("note.txt").read_text(), child.stdout.strip()))""")
    assert result.strip() == ("([0, 1, 2], [0.0, 1.0, 2.0], 'fit complete', '42')")


def test_long_analysis_keeps_namespace(analysis_session):
    # Exceed the former ten-second deadline, then checkpoint and resume.
    assert (
        analysis_session.execute(
            "import time\nretained = 42\ntime.sleep(10.2)\nprint(retained)"
        ).strip()
        == "42"
    )
    analysis_session.restore(analysis_session.snapshot())
    assert analysis_session.execute("retained").strip() == "42"


def test_worker_exit_resets_session(analysis_session):
    analysis_session.execute("retained = 42")
    with pytest.raises(RuntimeError, match="worker stopped unexpectedly"):
        analysis_session.execute("import os\nos._exit(1)")
    assert analysis_session.snapshot() is None
    assert not analysis_session.protocol_acknowledged()
    assert "NameError" in analysis_session.execute("retained")


def test_last_line_statements_run_instead_of_being_wrapped(analysis_session):
    # A bare expression is still echoed, which is what the wrapper is for.
    assert analysis_session.execute("candidate = 3.5\ncandidate") == "3.5\n"

    # An unspaced assignment used to become `print(baseline_rms=candidate)`,
    # which raised TypeError and silently dropped the assignment.
    analysis_session.execute("baseline_rms=candidate")
    assert analysis_session.execute("print(baseline_rms)") == "3.5\n"

    # An augmented assignment used to become a SyntaxError, discarding every
    # earlier line in the same cell.
    analysis_session.execute("periods = [10.0, 20.0]\nn_peaks = len(periods)\nn_peaks+=1")
    assert analysis_session.execute("print(periods, n_peaks)") == "[10.0, 20.0] 3\n"

    # A variable named after a print keyword argument is an assignment too.
    analysis_session.execute("end=times_days[-1]")
    assert analysis_session.execute("print(end == times_days[-1])") == "True\n"


def test_last_line_inside_a_block_keeps_its_indentation(analysis_session):
    # Dedenting the wrapped line used to raise SyntaxError, discarding the cell.
    assert analysis_session.execute("for t in range(3):\n    t") == "0\n1\n2\n"

    # Dedenting also used to hoist print(...) out of the block it belonged to,
    # so a branch that must not run would run anyway.
    assert analysis_session.execute("if False:\n    hidden = 1\n    hidden") == (
        "No output. You likely forgot to print the result. "
        "Please use `print(...)` to see any output."
    )


def test_submission_guide_uses_real_newlines(analysis_session):
    from stargazer.tools import STARGAZER_SUBMISSION_GUIDE

    # The guide is the prompt's mandatory step 0, so a literal "\n" is the
    # first thing every agent reads.
    assert "\\n" not in STARGAZER_SUBMISSION_GUIDE
    assert len(STARGAZER_SUBMISSION_GUIDE.splitlines()) == 6
    assert analysis_session.execute("print(STARGAZER_SUBMISSION_GUIDE)").count("\n") == 7
