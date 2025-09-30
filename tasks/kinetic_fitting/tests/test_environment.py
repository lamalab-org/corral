"""Test environment creation and tool integration."""

from pathlib import Path

import pytest

from kinetic_fitting.env import create_environments


def test_environment_creation():
    """Test that environments can be created from task JSON."""
    task_json_path = Path(__file__).parent.parent / "tasks" / "kinetic_fitting_tasks.json"
    
    if not task_json_path.exists():
        pytest.skip(f"Task JSON not found: {task_json_path}")
    
    envs = create_environments(task_json_path)
    
    assert len(envs) == 1, "Should create exactly one environment"
    assert 'kinetic_fitting' in envs, "Should have kinetic_fitting environment"
    
    env = envs['kinetic_fitting']
    assert hasattr(env, 'tools'), "Environment should have tools"
    assert len(env.tools) > 0, "Environment should have at least one tool"


def test_environment_tools():
    """Test that environment has expected tools."""
    task_json_path = Path(__file__).parent.parent / "tasks" / "kinetic_fitting_tasks.json"
    
    if not task_json_path.exists():
        pytest.skip(f"Task JSON not found: {task_json_path}")
    
    envs = create_environments(task_json_path)
    env = envs['kinetic_fitting']
    
    expected_tools = [
        'describe_experimental_data',
        'get_current_network', 
        'fit_single_experiment',
        'fit_all_experiments',
        'evaluate_phenomenological_trends',
        'modify_reaction_network',
        'analyze_fit_with_vision'
    ]
    
    tool_names = list(env.tools.keys())
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Missing expected tool: {expected_tool}"


def test_tool_execution():
    """Test basic tool execution."""
    task_json_path = Path(__file__).parent.parent / "tasks" / "kinetic_fitting_tasks.json"
    data_path = Path(__file__).parent.parent / "data" / "experimental_data.h5"
    
    if not task_json_path.exists():
        pytest.skip(f"Task JSON not found: {task_json_path}")
    
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    
    envs = create_environments(task_json_path)
    env = envs['kinetic_fitting']
    
    # Test describe_experimental_data tool
    result = env.call_tool('describe_experimental_data', {'data_path': str(data_path)})
    
    assert hasattr(result, 'result'), "Tool call should return result"
    assert isinstance(result.result, str), "Result should be a string"
    assert len(result.result) > 0, "Result should not be empty"
    assert 'MRG-' in result.result, "Result should contain experiment names"


def test_environment_workspace():
    """Test that environment has proper workspace setup."""
    task_json_path = Path(__file__).parent.parent / "tasks" / "kinetic_fitting_tasks.json"
    
    if not task_json_path.exists():
        pytest.skip(f"Task JSON not found: {task_json_path}")
    
    envs = create_environments(task_json_path)
    env = envs['kinetic_fitting']
    
    assert hasattr(env, 'current_work_dir'), "Environment should have working directory"
    assert hasattr(env, 'fs_manager'), "Environment should have file system manager"
    
    work_dir = env.get_current_work_dir()
    assert work_dir is not None, "Working directory should not be None"
    assert Path(work_dir).exists(), "Working directory should exist"