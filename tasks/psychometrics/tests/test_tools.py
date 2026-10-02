from tempfile import TemporaryDirectory

from corral_psychometrics.tools import validate_model_syntax, workspace_tools


def test_validate_model_syntax_only_parses() -> None:
    valid = validate_model_syntax.execute(syntax="F1 =~ x1 + x2")
    invalid = validate_model_syntax.execute(syntax="F1 =~")

    assert "'valid': True" in valid
    assert "'error': None" in valid
    assert "'valid': False" in invalid
    assert "SyntaxError" in invalid


def test_validate_model_syntax_is_in_workspace_toolset() -> None:
    with TemporaryDirectory() as workspace:
        assert "validate_model_syntax" in workspace_tools(workspace)


def test_repl_fits_ordinal_semopy_models() -> None:
    code = """
import pandas as pd
from semopy import Model
from semopy.polycorr import polychoric_corr

rng = np.random.default_rng(0)
f = rng.normal(size=600)
df = pd.DataFrame(
    {f"x{i}": np.digitize(0.8 * f + 0.6 * rng.normal(size=600), [-1, 0, 1]) for i in range(4)}
)
model = Model("F =~ x0 + x1 + x2 + x3\\nDEFINE(ordinal) x0 x1 x2 x3")
model.fit(df, obj="DWLS")
print("polychoric", round(polychoric_corr(df.x0, df.x1), 2))
"""
    with TemporaryDirectory() as workspace:
        repl = workspace_tools(workspace)["PythonREPL"]
        with repl.create_session() as session:
            output = session.execute(code)

    assert "Traceback" not in output
    # Latent correlation between two items with 0.8 loadings is 0.64.
    assert abs(float(output.split()[-1]) - 0.64) < 0.08


def test_repl_call_past_its_time_limit_is_stopped_and_session_kept() -> None:
    import time

    import pytest
    from corral.runtime.python_repl import PythonREPLTimeoutError

    with TemporaryDirectory() as workspace:
        repl = workspace_tools(workspace, repl_timeout_minutes=2 / 60)["PythonREPL"]
        with repl.create_session() as session:
            session.execute("kept = 41")
            checkpoint = session.snapshot()
        with repl.create_session() as session:
            session.restore(checkpoint)
            started = time.monotonic()
            # sleep blocks in C, as a long compiled model fit does.
            with pytest.raises(PythonREPLTimeoutError):
                session.execute("import time\ntime.sleep(60)\nkept = 0")
            assert time.monotonic() - started < 15
        with repl.create_session() as session:
            session.restore(checkpoint)
            assert session.execute("print(kept + 1)").strip() == "42"


def test_repl_time_limit_is_configurable_and_described() -> None:
    from corral_psychometrics.env import create_environments

    with TemporaryDirectory() as workspace:
        assert workspace_tools(workspace)["PythonREPL"].timeout_seconds == 15 * 60
        unlimited = workspace_tools(workspace, repl_timeout_minutes=None)["PythonREPL"]
        assert unlimited.timeout_seconds is None
        assert "minutes" not in unlimited.description
        assert "at most 15 minutes" in workspace_tools(workspace)["PythonREPL"].description

    environment = next(iter(create_environments(level=1, repl_timeout_minutes=7).values()))
    tools = environment.get_available_tools()
    repl = next(tool for tool in tools if tool["function"]["name"] == "PythonREPL")
    assert "at most 7 minutes" in repl["function"]["description"]
