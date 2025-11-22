"""Simple test script for the simplified terminal tools.

This script tests the basic functionality of run_in_terminal and list_terminal_sessions.
"""

import json

from corral.utils.terminal_tools import list_terminal_sessions, run_in_terminal


def test_basic_command():
    """Test a simple command execution."""
    result = run_in_terminal.execute(command="echo 'Hello from terminal'", timeout=10)

    result_dict = json.loads(result)
    assert result_dict["success"] is True, "Command should execute successfully"
    assert result_dict["exit_code"] == 0, "Exit code should be 0 for successful command"
    assert (
        "Hello from terminal" in result_dict["stdout"]
    ), "Output should contain the echoed text"
    assert "session_id" in result_dict, "Result should include a session ID"
    assert isinstance(result_dict["session_id"], str), "Session ID should be a string"


def test_session_persistence():
    """Test that sessions persist working directory."""
    # First, create a session by running a command
    result_init = run_in_terminal.execute(
        command="echo 'Initializing session'", timeout=10
    )
    result_init_dict = json.loads(result_init)
    session_id = result_init_dict["session_id"]

    assert result_init_dict["success"] is True, "Initial command should succeed"
    assert session_id, "Session ID should be created"

    # First command - check current directory
    result1 = run_in_terminal.execute(command="pwd", timeout=10, session_id=session_id)
    result1_dict = json.loads(result1)

    assert result1_dict["success"] is True, "pwd command should succeed"
    assert result1_dict["session_id"] == session_id, "Session ID should be preserved"
    assert len(result1_dict["stdout"].strip()) > 0, "Should return current directory"

    # Second command - create temp directory
    result2 = run_in_terminal.execute(
        command="mkdir -p /tmp/test_terminal_tools", timeout=10, session_id=session_id
    )
    result2_dict = json.loads(result2)

    assert result2_dict["success"] is True, "Directory creation should succeed"
    assert result2_dict["session_id"] == session_id, "Session ID should remain the same"


def test_list_sessions():
    """Test listing all sessions."""
    result = list_terminal_sessions.execute()
    result_dict = json.loads(result)

    assert "session_count" in result_dict, "Result should include session_count"
    assert "sessions" in result_dict, "Result should include sessions list"
    assert isinstance(
        result_dict["session_count"], int
    ), "session_count should be an integer"
    assert isinstance(result_dict["sessions"], list), "sessions should be a list"
    assert result_dict["session_count"] >= 0, "session_count should be non-negative"

    # Verify structure of each session
    for session in result_dict["sessions"]:
        assert "session_id" in session, "Each session should have a session_id"
        assert (
            "cwd" in session
        ), "Each session should have a cwd (current working directory)"
        assert "command_count" in session, "Each session should have a command_count"
        assert isinstance(
            session["command_count"], int
        ), "command_count should be an integer"


def test_error_handling():
    """Test error handling for invalid commands."""
    result = run_in_terminal.execute(command="nonexistent_command_xyz", timeout=5)
    result_dict = json.loads(result)

    assert result_dict["success"] is False, "Invalid command should fail"
    assert (
        result_dict["exit_code"] != 0
    ), "Exit code should be non-zero for failed command"
    assert len(result_dict["stderr"]) > 0, "stderr should contain error message"


def test_timeout():
    """Test timeout functionality."""
    result = run_in_terminal.execute(command="sleep 10", timeout=3)
    result_dict = json.loads(result)

    assert result_dict["success"] is False, "Timeout command should fail"
    assert "error" in result_dict, "Result should include error field"
    assert (
        result_dict.get("error") == "TimeoutExpired"
    ), "Error should be TimeoutExpired"
    assert "timeout" in str(result_dict).lower(), "Error message should mention timeout"
