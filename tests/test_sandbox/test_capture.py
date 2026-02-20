from corral.sandbox._capture import generate_output_capture_code, parse_execution_output


class TestParseExecutionOutput:
    def test_empty_stdout(self):
        result, lines = parse_execution_output("")
        assert result == {}
        assert lines == []

    def test_only_marker(self):
        stdout = 'EXECUTION_RESULT: {"result": 42}'
        result, lines = parse_execution_output(stdout)
        assert result == {"result": 42}
        assert lines == []

    def test_marker_with_output(self):
        stdout = 'hello world\nsome debug\nEXECUTION_RESULT: {"x": 1}'
        result, lines = parse_execution_output(stdout)
        assert result == {"x": 1}
        assert lines == ["hello world", "some debug"]

    def test_no_marker(self):
        stdout = "just output\nno marker here"
        result, lines = parse_execution_output(stdout)
        assert result == {}
        assert lines == ["just output", "no marker here"]

    def test_invalid_json_marker(self):
        stdout = "EXECUTION_RESULT: not-json"
        result, lines = parse_execution_output(stdout)
        assert result == {}
        assert lines == []


class TestGenerateOutputCaptureCode:
    def test_returns_string(self):
        code = generate_output_capture_code()
        assert isinstance(code, str)
        assert "EXECUTION_RESULT:" in code

    def test_code_is_valid_python(self):
        code = generate_output_capture_code()
        # Should compile without syntax errors
        compile(code, "<capture>", "exec")
