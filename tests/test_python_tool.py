import json

from corral.utils import execute_python_code


class TestPythonExecutionTools:
    """Test tools that execute Python code."""

    def test_execute_python_code_simple(self):
        """Test simple code execution."""
        code = "result = 10 * 5"

        result_json = execute_python_code.execute(python_code=code)
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["execution_result"]["result"] == 50

    def test_execute_python_code_with_input_data(self):
        """Test code execution with input data."""
        code = """
# Process the input data
total = sum(input_data)
average = total / len(input_data)
output = {
    'total': total,
    'average': average,
    'count': len(input_data)
}
"""
        input_data = json.dumps([1, 2, 3, 4, 5])

        result_json = execute_python_code.execute(
            python_code=code, input_data=input_data
        )

        result = json.loads(result_json)
        assert result["success"] is True

        output = result["execution_result"]["output"]
        assert output["total"] == 15
        assert output["average"] == 3.0
        assert output["count"] == 5

    def test_execute_python_code_syntax_error(self):
        """Test error handling for syntax errors."""
        bad_code = "this is not valid python syntax {"

        result_json = execute_python_code.execute(python_code=bad_code)
        result = json.loads(result_json)

        assert result["success"] is False
        assert "error" in result

    def test_execute_python_code_runtime_error(self):
        """Test error handling for runtime errors."""
        code = "result = undefined_variable * 2"

        result_json = execute_python_code.execute(python_code=code)
        result = json.loads(result_json)

        assert result["success"] is False
        assert "error" in result

    def test_execute_python_code_with_timeout(self):
        """Test code execution with timeout."""
        # Code that should complete quickly
        fast_code = "result = sum(range(100))"

        result_json = execute_python_code.execute(
            python_code=fast_code,
            timeout=5,  # 5 second timeout
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["execution_result"]["result"] == sum(range(100))


class TestPythonExecution:
    """Test Python code execution functions."""

    def test_execute_python_code_simple(self):
        """Test simple Python code execution."""
        code = "result = 2 + 2"
        result_json = execute_python_code.execute(python_code=code)

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["execution_result"]["result"] == 4

    def test_execute_python_code_with_input(self):
        """Test Python code execution with input data."""
        code = "output = sum(input_data)"
        input_data = json.dumps([1, 2, 3, 4, 5])

        result_json = execute_python_code.execute(
            python_code=code, input_data=input_data
        )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["execution_result"]["output"] == 15

    def test_execute_python_code_syntax_error(self):
        """Test Python code execution with syntax error."""
        code = "this is not valid python code {"
        result_json = execute_python_code.execute(python_code=code)

        result = json.loads(result_json)
        assert result["success"] is False
        assert "error" in result

    def test_execute_python_code_runtime_error(self):
        """Test Python code execution with runtime error."""
        code = "result = undefined_variable + 1"
        result_json = execute_python_code.execute(python_code=code)

        result = json.loads(result_json)
        assert result["success"] is False
        assert "error" in result
