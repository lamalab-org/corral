"""
Integration tests for the TaskEnvironment and task execution system.

Tests the full workflow including path resolution, scoring, and task dependencies.
"""

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_files" / "temp")

import pytest
from catalyst.env import create_environments, entries_to_task_definitions
from corral.backend.env import Environment, Toolset
from corral.utils.task_loader import load_task_entries
from catalyst.score import check_mp_structure, check_slabs_json, check_valid_json_file
from hypothesis import given
from hypothesis import strategies as st

from corral.backend.state import TaskRunState
from corral.backend.task import InputRef, TaskDefinition

TEMP_DIR = Path(os.environ["CORRAL_WORK_DIR"])


def _use_files_workspace(env, work_dir):
    """Point a freshly built environment's trial workspace at ``work_dir``.

    On construction an ``Environment`` creates an isolated ``{task}_trial_N``
    subdirectory and scopes submitted-answer/path resolution to it. These tests
    stage their files directly in the provided work dir, so we redirect the
    trial workspace there so scoring resolves against the staged files.
    """
    env.current_work_dir = str(work_dir)
    return env


class TestTaskEnvironment:
    """Integration tests for TaskEnvironment."""

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
            ),
            "task2": TaskDefinition(
                name="Test Slab Enumeration",
                description="Test enumerating slabs",
                tools=["mock_tool"],
                scoring_fn=check_slabs_json,
                submission_format="/path/to/slabs.json",
                initial_input={"miller_index": [1, 1, 1]},
                input_map={"task1": InputRef("task1")},
            ),
            "task3": TaskDefinition(
                name="Test JSON Validation",
                description="Test validating JSON files",
                tools=["mock_tool"],
                scoring_fn=check_valid_json_file,
                submission_format="/path/to/file.json",
                initial_input={},
            ),
        }

    @pytest.fixture()
    def shared_task_runs(self):
        """Run store shared between linked environments."""
        return {}

    @pytest.fixture()
    def mock_tools(self):
        """Create mock tools for testing."""
        mock_tool = Mock()
        mock_tool.name = "mock_tool"
        # Not a background-capable tool: a bare Mock would otherwise auto-vivify
        # this attribute to a truthy Mock, so the environment would try to build
        # a `start_<tool>` background variant and fail on the Mock's metadata.
        mock_tool.background_capable = False
        return {"mock_tool": mock_tool}

    def test_environment_creation(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test basic environment creation."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            assert env.task_id == "task1"
            assert env.current_task == sample_tasks["task1"]
            assert env.group_tasks == sample_tasks
            assert "mock_tool" in env.tools

    def test_file_tools_setup(self, sample_tasks, shared_task_runs, mock_tools, temp_workspace):
        """Test that file tools are properly set up."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
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

    def test_task_prompt_generation(self, sample_tasks, shared_task_runs, mock_tools, temp_workspace):
        """Test task prompt generation."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            prompt = env.get_task_prompt()
            assert "Test Structure Retrieval" in prompt
            assert "Test retrieving a structure file" in prompt
            assert "/path/to/structure.cif" in prompt
            assert "mp_id: mp-149" in prompt

    def test_task_prompt_with_dependencies(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test task prompt generation with dependencies."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            # First complete task1
            shared_task_runs["task1"] = TaskRunState(
                task_id="task1",
                output={"answer": "bulk_structure.cif"},
                score=1.0,
            )

            env = Environment(
                task_id="task2",
                task=sample_tasks["task2"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            prompt = env.get_task_prompt()
            assert "Test Slab Enumeration" in prompt
            assert "task1 (from task1): bulk_structure.cif" in prompt
            # The dependency is resolved strictly and rendered inline; the old
            # trailing "task1 (available)" status line has been removed.

    def test_scoring_with_valid_submission(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring with a valid file submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            _use_files_workspace(env, temp_workspace)

            # Simulate submitting a valid CIF file path
            env.state.submitted_answer = "bulk_structure.cif"
            score = env.score()

            assert score == 1.0
            assert env.state.is_completed("task1")
            assert env.state.get_output("task1") == str(
                temp_workspace / "bulk_structure.cif"
            )

    def test_scoring_with_markdown_formatted_submission(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring with markdown-formatted submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            _use_files_workspace(env, temp_workspace)

            # Submit with markdown formatting
            env.state.submitted_answer = "The structure file is `bulk_structure.cif`."
            score = env.score()

            assert score == 1.0
            assert env.state.get_output("task1") == str(
                temp_workspace / "bulk_structure.cif"
            )

    def test_scoring_with_quoted_submission(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring with quoted submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task3",
                task=sample_tasks["task3"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            _use_files_workspace(env, temp_workspace)

            # Submit with quotes
            env.state.submitted_answer = 'The JSON file is "valid_data.json".'
            score = env.score()

            assert score == 1.0
            assert env.state.get_output("task3") == str(
                temp_workspace / "valid_data.json"
            )

    def test_scoring_with_absolute_path_submission(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring with absolute path submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            # Submit absolute path
            abs_path = str(temp_workspace / "bulk_structure.cif")
            env.state.submitted_answer = abs_path
            score = env.score()

            assert score == 1.0
            assert env.state.get_output("task1") == abs_path

    def test_scoring_with_nonexistent_file(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring with nonexistent file."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            env.state.submitted_answer = "nonexistent_file.cif"
            score = env.score()

            assert score == 0.0

    def test_scoring_with_wrong_file_type(self, sample_tasks, shared_task_runs, mock_tools, temp_workspace):
        """Test scoring with wrong file type."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task3",  # Expects JSON
                task=sample_tasks["task3"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            env.state.submitted_answer = "invalid.txt"  # Not JSON
            score = env.score()

            assert score == 0.0

    def test_scoring_with_subdirectory_file(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring when file is in subdirectory."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            _use_files_workspace(env, temp_workspace)

            # Submit filename that exists in subdirectory
            env.state.submitted_answer = "output.cif"
            score = env.score()

            assert score == 1.0
            # Should resolve to the file in the subdirectory
            assert "results/output.cif" in env.state.get_output("task1")

    def test_scoring_with_no_submission(self, sample_tasks, shared_task_runs, mock_tools, temp_workspace):
        """Test scoring when no submission is made."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            # No submission made
            score = env.score()

            assert score == 0.0

    def test_scoring_with_empty_submission(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring with empty submission."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            env.state.submitted_answer = ""
            score = env.score()

            assert score == 0.0

    def test_scoring_error_handling(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test scoring error handling."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            # Create a task with a scoring function that raises an exception
            def failing_scorer():
                raise ValueError("Test error")

            sample_tasks["task1"] = replace(
                sample_tasks["task1"], scoring_fn=failing_scorer
            )

            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
            )

            env.state.submitted_answer = "bulk_structure.cif"
            score = env.score()

            assert score == 0.0

    def test_reset_state_updates_file_tools(
        self, sample_tasks, shared_task_runs, mock_tools, temp_workspace
    ):
        """Test that reset_state properly updates file tools for new workspace."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            env = Environment(
                task_id="task1",
                task=sample_tasks["task1"],
                toolset=Toolset(pool=mock_tools),
                base_work_dir=str(temp_workspace),
                group_tasks=sample_tasks,
                shared_task_runs=shared_task_runs,
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
    def sample_task_dir(self, temp_workspace):
        """Create a directory with standardized task JSON files."""
        task_entries = [
            {
                "uuid": "test-uuid-001",
                "id": "retrieve_structure",
                "name": "Retrieve Bulk Structure",
                "description": "Retrieve structure and save as CIF",
                "tools": ["mock_tool"],
                "scoring_function": "mp_structure",
                "submission_format": "/path/to/structure.cif",
                "initial_input": {"mp_id": "mp-149"},
            },
            {
                "uuid": "test-uuid-002",
                "id": "enumerate_slabs",
                "name": "Enumerate Slabs",
                "description": "Enumerate slabs from bulk structure",
                "tools": ["mock_tool"],
                "scoring_function": "slabs_json",
                "submission_format": "/path/to/slabs.json",
                "input_from_tasks": ["retrieve_structure"],
                "initial_input": {"miller_index": [1, 1, 1]},
            },
            {
                "uuid": "test-uuid-003",
                "id": "validate_json",
                "name": "Validate JSON",
                "description": "Validate a JSON file",
                "tools": ["mock_tool"],
                "scoring_function": "file_exists",
                "submission_format": "/path/to/file.json",
                "initial_input": {},
            },
        ]

        task_dir = temp_workspace / "tasks_json"
        task_dir.mkdir()
        json_file = task_dir / "tasks.json"
        with json_file.open("w") as f:
            json.dump(task_entries, f, indent=2)

        return task_dir

    def test_load_and_convert_tasks(self, sample_task_dir, temp_workspace):
        """Test loading task entries and converting to TaskDefinitions."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            entries = load_task_entries(local_dir=sample_task_dir)
            assert len(entries) == 3

            tasks = entries_to_task_definitions(entries, str(temp_workspace))

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
            assert "retrieve_structure" in task2.dependencies()

    def test_create_environments(self, sample_task_dir, temp_workspace):
        """Test creating environments from task directory."""
        with patch.dict(os.environ, {"CORRAL_WORK_DIR": str(temp_workspace)}):
            environments = create_environments(
                local_dir=sample_task_dir, work_dir=str(temp_workspace)
            )

            assert len(environments) == 3
            assert "retrieve_structure" in environments
            assert "enumerate_slabs" in environments
            assert "validate_json" in environments

            # Check environment properties
            env1 = environments["retrieve_structure"]
            assert isinstance(env1, Environment)
            assert env1.task_id == "retrieve_structure"

    def test_task_dependency_workflow(self, sample_task_dir, temp_workspace):
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
                local_dir=sample_task_dir, work_dir=str(temp_workspace)
            )

            # Complete task 1
            env1 = environments["retrieve_structure"]
            _use_files_workspace(env1, temp_workspace)
            env1.state.submitted_answer = "test_structure.cif"
            score1 = env1.score()
            assert score1 == 1.0

            # Complete task 2 (depends on task 1)
            env2 = environments["enumerate_slabs"]
            _use_files_workspace(env2, temp_workspace)
            prompt2 = env2.get_task_prompt()
            assert "(from retrieve_structure):" in prompt2

            env2.state.submitted_answer = "test_slabs.json"
            score2 = env2.score()
            assert score2 == 1.0

            # Complete independent task 3
            env3 = environments["validate_json"]
            _use_files_workspace(env3, temp_workspace)
            env3.state.submitted_answer = "test_data.json"
            score3 = env3.score()
            assert score3 == 1.0

    def test_invalid_task_json(self, temp_workspace):
        """Test handling of invalid task JSON."""
        invalid_dir = temp_workspace / "invalid_tasks"
        invalid_dir.mkdir()
        (invalid_dir / "bad.json").write_text("{ invalid json")

        with pytest.raises(json.JSONDecodeError):
            load_task_entries(local_dir=invalid_dir)

    def test_missing_scoring_function(self, temp_workspace):
        """Test handling of missing scoring function."""
        entries = [
            {
                "uuid": "test-uuid-bad",
                "id": "test_task",
                "name": "Test Task",
                "description": "Test description",
                "scoring_function": "nonexistent_function",
                "submission_format": "/path/to/file",
            }
        ]

        with pytest.raises(
            ValueError, match="Scoring function 'nonexistent_function' not found"
        ):
            entries_to_task_definitions(entries, str(temp_workspace))

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

                env = Environment(
                    task_id="test",
                    task=task,
                    toolset=Toolset(pool={}),
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

        env = Environment(
            task_id="test",
            task=task,
            toolset=Toolset(pool={}),
            base_work_dir=str(scoring_environment),
        )
        _use_files_workspace(env, scoring_environment)

        # Test relative path
        env.state.submitted_answer = "structure.cif"
        score = env.score()
        assert score == 1.0

        # Check that the resolved path is absolute
        resolved_path = env.state.get_output("test")
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

        env = Environment(
            task_id="test",
            task=task,
            toolset=Toolset(pool={}),
            base_work_dir=str(scoring_environment),
        )
        _use_files_workspace(env, scoring_environment)

        # Submit filename that exists in subdirectory
        env.state.submitted_answer = "output.cif"
        score = env.score()
        assert score == 1.0

        # Should resolve to the subdirectory file
        resolved_path = env.state.get_output("test")
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

        env = Environment(
            task_id="test",
            task=task,
            toolset=Toolset(pool={}),
            base_work_dir=str(scoring_environment),
        )
        _use_files_workspace(env, scoring_environment)

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
            # Reset the shared results for each test
            env.state.task_runs.clear()
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

        env = Environment(
            task_id="test",
            task=task,
            toolset=Toolset(pool={}),
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
            env.state.task_runs.clear()
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
        )

        task2 = TaskDefinition(
            name="Task 2",
            description="Second task",
            tools=[],
            scoring_fn=check_valid_json_file,
            submission_format="/path/to/file",
            initial_input={},
            input_map={"task1": InputRef("task1")},
        )

        tasks = {"task1": task1, "task2": task2}

        # Linked tasks share this store through their state
        shared_task_runs: dict = {}

        # Complete task1 with a relative path
        env1 = Environment(
            task_id="task1",
            task=tasks["task1"],
            toolset=Toolset(pool={}),
            base_work_dir=str(scoring_environment),
            group_tasks=tasks,
            shared_task_runs=shared_task_runs,
        )
        _use_files_workspace(env1, scoring_environment)

        env1.state.submitted_answer = "structure.cif"
        score1 = env1.score()
        assert score1 == 1.0

        # Create task2 environment
        env2 = Environment(
            task_id="task2",
            task=tasks["task2"],
            toolset=Toolset(pool={}),
            base_work_dir=str(scoring_environment),
            group_tasks=tasks,
            shared_task_runs=shared_task_runs,
        )

        # Check that task2 can see the resolved path from task1
        prompt = env2.get_task_prompt()
        assert "task1 (from task1):" in prompt
        # The resolved path should be absolute
        assert str(scoring_environment) in prompt
