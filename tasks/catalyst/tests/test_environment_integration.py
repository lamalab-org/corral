"""
Integration tests for the TaskGroupEnvironment and task execution system.

Tests the full workflow including path resolution, scoring, and task dependencies.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from catalyst.env import TaskGroupEnvironment, create_environments, load_tasks_from_json
from catalyst.score import check_mp_structure, check_slabs_json, check_valid_json_file
from hypothesis import given
from hypothesis import strategies as st

from corral.task import TaskDefinition, TaskGroup

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_files" / "temp")
TEMP_DIR = Path(os.environ["CORRAL_WORK_DIR"])


class TestTaskGroupEnvironment:
    """Integration tests for TaskGroupEnvironment."""

    @pytest.fixture()
    def temp_workspace(self):
        """Create a temporary workspace with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files that scoring functions expect
            test_files = {
                "bulk_structure.cif": """# CIF file
# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1
""",
                "slabs.json": json.dumps(
                    {
                        "slab_1": """# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1

""",
                        "slab_2": """# Another slab CIF
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1

""",
                    }
                ),
                "valid_data.json": json.dumps({"test": "data", "value": 123}),
                "invalid.txt": "This is not JSON",
                "adsorption_sites.json": json.dumps(
                    {
                        "ontop": [[0.0, 0.0, 3.5], [0.5, 0.5, 3.5]],
                        "bridge": [[0.25, 0.25, 3.2]],
                        "hollow": [[0.33, 0.67, 3.0]],
                    }
                ),
            }

            for filename, content in test_files.items():
                file_path = temp_path / filename
                file_path.write_text(content)

            # Create subdirectory with additional files
            subdir = temp_path / "results"
            subdir.mkdir()
            (subdir / "output.cif").write_text(test_files["bulk_structure.cif"])

            yield temp_path

    @pytest.fixture()
    def sample_tasks(self):
        """Create sample task definitions for testing."""
        return {
            "task1": TaskDefinition(
                name="Test Structure Retrieval",
                description="Test retrieving a structure file",
                tools=["mock_tool"],
                scoring_fn=check_mp_structure,
                submission_format="/path/to/structure.cif",
                initial_input={"mp_id": "mp-149"},
                input_from_tasks=[],
            ),
            "task2": TaskDefinition(
                name="Test Slab Enumeration",
                description="Test enumerating slabs",
                tools=["mock_tool"],
                scoring_fn=check_slabs_json,
                submission_format="/path/to/slabs.json",
                initial_input={"miller_index": [1, 1, 1]},
                input_from_tasks=["task1"],
            ),
            "task3": TaskDefinition(
                name="Test JSON Validation",
                description="Test validating JSON files",
                tools=["mock_tool"],
                scoring_fn=check_valid_json_file,
                submission_format="/path/to/file.json",
                initial_input={},
                input_from_tasks=[],
            ),
        }

    @pytest.fixture()
    def task_group(self, sample_tasks):
        """Create a TaskGroup with sample tasks."""
        return TaskGroup("test_group", sample_tasks)

    @pytest.fixture()
    def mock_tools(self):
        """Create mock tools for testing."""
        mock_tool = Mock()
        mock_tool.name = "mock_tool"
        return {"mock_tool": mock_tool}

    def test_environment_creation(
        self, sample_tasks, task_group, mock_tools, temp_workspace
    ):
        """Test basic environment creation."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            assert env.task_id == "task1"
            assert env.current_task == sample_tasks["task1"]
            assert env.task_group == task_group
            assert "mock_tool" in env.tools

    def test_file_tools_setup(self, task_group, mock_tools, temp_workspace):
        """Test that file tools are properly set up."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Check that file tools are available
            expected_file_tools = [
                "list_files",
                "read_file",
                "write_file",
                "file_info",
                "cat_files",
                "copy_file",
            ]
            for tool_name in expected_file_tools:
                assert tool_name in env.tools

    def test_task_prompt_generation(self, task_group, mock_tools, temp_workspace):
        """Test task prompt generation."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            prompt = env.get_task_prompt()
            assert "Test Structure Retrieval" in prompt
            assert "Test retrieving a structure file" in prompt
            assert "/path/to/structure.cif" in prompt
            assert "mp_id: mp-149" in prompt

    def test_task_prompt_with_dependencies(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test task prompt generation with dependencies."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            # First complete task1
            task_group.store_result("task1", {"answer": "bulk_structure.cif"}, 1.0)

            env = TaskGroupEnvironment(
                task_id="task2",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            prompt = env.get_task_prompt()
            assert "Test Slab Enumeration" in prompt
            assert "Input from task1: bulk_structure.cif" in prompt
            assert "task1 (available)" in prompt

    def test_scoring_with_valid_submission(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring with a valid file submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Simulate submitting a valid CIF file path
            env.state.submitted_answer = "bulk_structure.cif"
            score = env.score()

            assert score == 1.0
            assert "task1" in task_group.results
            assert task_group.results["task1"]["answer"] == str(
                temp_workspace / "bulk_structure.cif"
            )

    def test_scoring_with_markdown_formatted_submission(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring with markdown-formatted submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Submit with markdown formatting
            env.state.submitted_answer = "The structure file is `bulk_structure.cif`."
            score = env.score()

            assert score == 1.0
            assert task_group.results["task1"]["answer"] == str(
                temp_workspace / "bulk_structure.cif"
            )

    def test_scoring_with_quoted_submission(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring with quoted submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task3",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Submit with quotes
            env.state.submitted_answer = 'The JSON file is "valid_data.json".'
            score = env.score()

            assert score == 1.0
            assert task_group.results["task3"]["answer"] == str(
                temp_workspace / "valid_data.json"
            )

    def test_scoring_with_absolute_path_submission(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring with absolute path submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Submit absolute path
            abs_path = str(temp_workspace / "bulk_structure.cif")
            env.state.submitted_answer = abs_path
            score = env.score()

            assert score == 1.0
            assert task_group.results["task1"]["answer"] == abs_path

    def test_scoring_with_nonexistent_file(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring with nonexistent file."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            env.state.submitted_answer = "nonexistent_file.cif"
            score = env.score()

            assert score == 0.0

    def test_scoring_with_wrong_file_type(self, task_group, mock_tools, temp_workspace):
        """Test scoring with wrong file type."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task3",  # Expects JSON
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            env.state.submitted_answer = "invalid.txt"  # Not JSON
            score = env.score()

            assert score == 0.0

    def test_scoring_with_subdirectory_file(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring when file is in subdirectory."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Submit filename that exists in subdirectory
            env.state.submitted_answer = "output.cif"
            score = env.score()

            assert score == 1.0
            # Should resolve to the file in the subdirectory
            assert "results/output.cif" in task_group.results["task1"]["answer"]

    def test_scoring_with_no_submission(self, task_group, mock_tools, temp_workspace):
        """Test scoring when no submission is made."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # No submission made
            score = env.score()

            assert score == 0.0

    def test_scoring_with_empty_submission(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test scoring with empty submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            env.state.submitted_answer = ""
            score = env.score()

            assert score == 0.0

    def test_scoring_error_handling(
        self, sample_tasks, task_group, mock_tools, temp_workspace
    ):
        """Test scoring error handling."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            # Create a task with a scoring function that raises an exception
            def failing_scorer():
                raise ValueError("Test error")

            sample_tasks["task1"].scoring_fn = failing_scorer

            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            env.state.submitted_answer = "bulk_structure.cif"
            score = env.score()

            assert score == 0.0

    def test_reset_state_updates_file_tools(
        self, task_group, mock_tools, temp_workspace
    ):
        """Test that reset_state properly updates file tools for new workspace."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = TaskGroupEnvironment(
                task_id="task1",
                task_group=task_group,
                subtask_specific_tools=mock_tools,
                base_work_dir=str(temp_workspace),
            )

            # Get initial workspace
            initial_workspace = env.current_work_dir

            # Reset state (creates new workspace)
            trial_id = env.reset_state()

            # Check that workspace changed
            assert env.current_work_dir != initial_workspace
            assert isinstance(trial_id, str)

            # Check that file tools still work
            assert "list_files" in env.tools
            assert env.tools["list_files"] is not None


class TestTaskSystemIntegration:
    """Integration tests for the complete task system."""

    @pytest.fixture()
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture()
    def sample_task_json(self, temp_workspace):
        """Create a sample task JSON file."""
        task_def = {
            "retrieve_structure": {
                "name": "Retrieve Bulk Structure",
                "description": "Retrieve structure and save as CIF",
                "tools": ["mock_tool"],
                "scoring_function": "mp_structure",
                "submission_format": "/path/to/structure.cif",
                "initial_input": {"mp_id": "mp-149"},
            },
            "enumerate_slabs": {
                "name": "Enumerate Slabs",
                "description": "Enumerate slabs from bulk structure",
                "tools": ["mock_tool"],
                "scoring_function": "slabs_json",
                "submission_format": "/path/to/slabs.json",
                "input_from_tasks": ["retrieve_structure"],
                "initial_input": {"miller_index": [1, 1, 1]},
            },
            "validate_json": {
                "name": "Validate JSON",
                "description": "Validate a JSON file",
                "tools": ["mock_tool"],
                "scoring_function": "file_exists",
                "submission_format": "/path/to/file.json",
                "initial_input": {},
            },
        }

        json_file = temp_workspace / "test_tasks.json"
        with json_file.open("w") as f:
            json.dump(task_def, f, indent=2)

        return json_file

    def test_load_tasks_from_json(self, sample_task_json, temp_workspace):
        """Test loading tasks from JSON file."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            tasks = load_tasks_from_json(sample_task_json, str(temp_workspace))

            assert len(tasks) == 3
            assert "retrieve_structure" in tasks
            assert "enumerate_slabs" in tasks
            assert "validate_json" in tasks

            # Check task properties
            task1 = tasks["retrieve_structure"]
            assert task1.name == "Retrieve Bulk Structure"
            assert task1.initial_input["mp_id"] == "mp-149"
            assert task1.initial_input["work_dir"] == str(temp_workspace)

            task2 = tasks["enumerate_slabs"]
            assert "retrieve_structure" in task2.input_from_tasks

    def test_create_environments(self, sample_task_json, temp_workspace):
        """Test creating environments from JSON file."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            environments = create_environments(
                sample_task_json, work_dir=str(temp_workspace)
            )

            assert len(environments) == 3
            assert "retrieve_structure" in environments
            assert "enumerate_slabs" in environments
            assert "validate_json" in environments

            # Check environment properties
            env1 = environments["retrieve_structure"]
            assert isinstance(env1, TaskGroupEnvironment)
            assert env1.task_id == "retrieve_structure"

    def test_task_dependency_workflow(self, sample_task_json, temp_workspace):
        """Test complete workflow with task dependencies."""
        # Create test files
        cif_content = """# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1

"""
        slabs_content = json.dumps({"slab_1": cif_content, "slab_2": cif_content})

        (temp_workspace / "test_structure.cif").write_text(cif_content)
        (temp_workspace / "test_slabs.json").write_text(slabs_content)
        (temp_workspace / "test_data.json").write_text('{"valid": "json"}')

        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            environments = create_environments(
                sample_task_json, work_dir=str(temp_workspace)
            )

            # Complete task 1
            env1 = environments["retrieve_structure"]
            env1.state.submitted_answer = "test_structure.cif"
            score1 = env1.score()
            assert score1 == 1.0

            # Complete task 2 (depends on task 1)
            env2 = environments["enumerate_slabs"]
            prompt2 = env2.get_task_prompt()
            assert "Input from retrieve_structure:" in prompt2

            env2.state.submitted_answer = "test_slabs.json"
            score2 = env2.score()
            assert score2 == 1.0

            # Complete independent task 3
            env3 = environments["validate_json"]
            env3.state.submitted_answer = "test_data.json"
            score3 = env3.score()
            assert score3 == 1.0

    def test_invalid_task_json(self, temp_workspace):
        """Test handling of invalid task JSON."""
        invalid_json = temp_workspace / "invalid.json"
        invalid_json.write_text("{ invalid json")

        with pytest.raises(json.JSONDecodeError):
            load_tasks_from_json(invalid_json, str(temp_workspace))

    def test_missing_scoring_function(self, temp_workspace):
        """Test handling of missing scoring function."""
        task_def = {
            "test_task": {
                "name": "Test Task",
                "description": "Test description",
                "scoring_function": "nonexistent_function",
                "submission_format": "/path/to/file",
            }
        }

        json_file = temp_workspace / "bad_tasks.json"
        with json_file.open("w") as f:
            json.dump(task_def, f)

        with pytest.raises(
            ValueError, match="Scoring function 'nonexistent_function' not found"
        ):
            load_tasks_from_json(json_file, str(temp_workspace))

    @given(st.text(min_size=1, max_size=100))
    def test_scoring_robustness(self, submission_text):
        """Property test: scoring should handle arbitrary submission text gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a valid test file
            test_file = temp_path / "test.cif"
            test_file.write_text("# CIF content")

            with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_path)}):
                task = TaskDefinition(
                    name="Test",
                    description="Test",
                    tools=[],
                    scoring_fn=check_mp_structure,
                    submission_format="/path/to/file",
                    initial_input={},
                )

                task_group = TaskGroup("test", {"test": task})
                env = TaskGroupEnvironment(
                    task_id="test",
                    task_group=task_group,
                    subtask_specific_tools={},
                    base_work_dir=str(temp_path),
                )

                env.state.submitted_answer = submission_text
                score = env.score()

                # Score should always be a float between 0 and 1
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0


class TestPathResolutionInScoring:
    """Specific tests for path resolution behavior in scoring."""

    @pytest.fixture()
    def scoring_environment(self):
        """Create environment for testing scoring path resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create various test files
            (temp_path / "structure.cif").write_text("""# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1
""")
            (temp_path / "data.json").write_text('{"test": "data"}')

            subdir = temp_path / "results"
            subdir.mkdir()
            (subdir / "output.cif").write_text("""# Output CIF generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.84927840
_cell_length_b   3.84927941
_cell_length_c   3.84927800
_cell_angle_alpha   60.00001213
_cell_angle_beta   60.00000347
_cell_angle_gamma   60.00001098
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si2
_cell_volume   40.32952685
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  1  0.87500000  0.87500000  0.87500000  1
  Si  Si1  1  0.12500000  0.12500000  0.12500000  1
""")
            (subdir / "analysis.json").write_text('{"analysis": "complete"}')

            with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_path)}):
                yield temp_path

    def test_scoring_resolves_relative_paths(self, scoring_environment):
        """Test that scoring correctly resolves relative paths."""
        task = TaskDefinition(
            name="Test",
            description="Test",
            tools=[],
            scoring_fn=check_mp_structure,
            submission_format="/path/to/file",
            initial_input={},
        )

        task_group = TaskGroup("test", {"test": task})
        env = TaskGroupEnvironment(
            task_id="test",
            task_group=task_group,
            subtask_specific_tools={},
            base_work_dir=str(scoring_environment),
        )

        # Test relative path
        env.state.submitted_answer = "structure.cif"
        score = env.score()
        assert score == 1.0

        # Check that the resolved path is absolute
        resolved_path = task_group.results["test"]["answer"]
        assert Path(resolved_path).is_absolute()
        assert Path(resolved_path).exists()

    def test_scoring_resolves_paths_from_subdirectories(self, scoring_environment):
        """Test that scoring finds files in subdirectories."""
        task = TaskDefinition(
            name="Test",
            description="Test",
            tools=[],
            scoring_fn=check_mp_structure,
            submission_format="/path/to/file",
            initial_input={},
        )

        task_group = TaskGroup("test", {"test": task})
        env = TaskGroupEnvironment(
            task_id="test",
            task_group=task_group,
            subtask_specific_tools={},
            base_work_dir=str(scoring_environment),
        )

        # Submit filename that exists in subdirectory
        env.state.submitted_answer = "output.cif"
        score = env.score()
        assert score == 1.0

        # Should resolve to the subdirectory file
        resolved_path = task_group.results["test"]["answer"]
        assert "results" in resolved_path
        assert "output.cif" in resolved_path

    def test_scoring_handles_formatted_answers(self, scoring_environment):
        """Test scoring with various answer formats."""
        task = TaskDefinition(
            name="Test",
            description="Test",
            tools=[],
            scoring_fn=check_valid_json_file,
            submission_format="/path/to/file",
            initial_input={},
        )

        task_group = TaskGroup("test", {"test": task})
        env = TaskGroupEnvironment(
            task_id="test",
            task_group=task_group,
            subtask_specific_tools={},
            base_work_dir=str(scoring_environment),
        )

        # Test various formatted answers
        test_cases = [
            "data.json",
            "`data.json`",
            '"data.json"',
            "'data.json'",
            "The file is data.json",
            "Answer: data.json",
            "Final answer: `data.json`",
            f"The path is {scoring_environment / 'data.json'}",
        ]

        for answer_format in test_cases:
            # Reset the task group for each test
            task_group.results.clear()
            env.state.submitted_answer = answer_format
            score = env.score()
            assert score == 1.0, f"Failed for format: {answer_format}"

    def test_scoring_handles_nonexistent_files_gracefully(self, scoring_environment):
        """Test that scoring handles nonexistent files without crashing."""
        task = TaskDefinition(
            name="Test",
            description="Test",
            tools=[],
            scoring_fn=check_mp_structure,
            submission_format="/path/to/file",
            initial_input={},
        )

        task_group = TaskGroup("test", {"test": task})
        env = TaskGroupEnvironment(
            task_id="test",
            task_group=task_group,
            subtask_specific_tools={},
            base_work_dir=str(scoring_environment),
        )

        # Test various nonexistent file formats
        test_cases = [
            "nonexistent.cif",
            "`missing_file.cif`",
            "/fake/path/to/file.cif",
            "The file is definitely_not_there.cif",
        ]

        for answer_format in test_cases:
            task_group.results.clear()
            env.state.submitted_answer = answer_format
            score = env.score()
            assert score == 0.0, f"Should fail for nonexistent file: {answer_format}"

    def test_task_dependency_path_resolution(self, scoring_environment):
        """Test path resolution in task dependencies."""
        # Create tasks with dependencies
        task1 = TaskDefinition(
            name="Task 1",
            description="First task",
            tools=[],
            scoring_fn=check_mp_structure,
            submission_format="/path/to/file",
            initial_input={},
            input_from_tasks=[],
        )

        task2 = TaskDefinition(
            name="Task 2",
            description="Second task",
            tools=[],
            scoring_fn=check_valid_json_file,
            submission_format="/path/to/file",
            initial_input={},
            input_from_tasks=["task1"],
        )

        tasks = {"task1": task1, "task2": task2}
        task_group = TaskGroup("test", tasks)

        # Complete task1 with a relative path
        env1 = TaskGroupEnvironment(
            task_id="task1",
            task_group=task_group,
            subtask_specific_tools={},
            base_work_dir=str(scoring_environment),
        )

        env1.state.submitted_answer = "structure.cif"
        score1 = env1.score()
        assert score1 == 1.0

        # Create task2 environment
        env2 = TaskGroupEnvironment(
            task_id="task2",
            task_group=task_group,
            subtask_specific_tools={},
            base_work_dir=str(scoring_environment),
        )

        # Check that task2 can see the resolved path from task1
        prompt = env2.get_task_prompt()
        assert "Input from task1:" in prompt
        # The resolved path should be absolute
        assert str(scoring_environment) in prompt
