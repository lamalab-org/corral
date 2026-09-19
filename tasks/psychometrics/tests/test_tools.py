from tempfile import TemporaryDirectory

from psychometrics.tools import validate_model_syntax, workspace_tools


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
