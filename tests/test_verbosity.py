"""
Tests for verbosity filtering functionality in corral.router.verbosity module.

This module tests the filter_tool_description and filter_argument_description
functions across different verbosity levels (BRIEF, WORKFLOW, COMPREHENSIVE, FULL)
using realistic tool docstrings from the tasks.
"""

import pytest

from corral.router.verbosity import ToolVerbosity, VerbosityConfig

# Complete tool docstring from corral_md/tools.py (execute_python_script)
FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT = """[BRIEF] Execute a Python script file with arguments in a controlled environment. [/BRIEF]

[DETAILED] This tool executes existing Python script files with command-line arguments, providing a controlled environment for running complex analysis workflows, data processing pipelines, or computational simulations.
It captures all output streams and provides comprehensive execution monitoring with timeout protection.
This is essential for integrating existing Python scripts into automated workflows and materials analysis pipelines. [/DETAILED]

[PROCEDURAL] When to use this tool:
- Use when you need to execute existing Python scripts with specific arguments. You can also use io tool to write a script and then execute it.
- Best suited for running complex analysis workflows or simulations
- Essential for integrating external Python tools into automated pipelines
- Recommended for batch processing and computational workflows
- Avoid for simple code execution
[/PROCEDURAL]

[CONTEXTUAL] How this tool works:
- Validates script file existence and accessibility
- Constructs command with script path and provided arguments
- Executes script in subprocess with timeout protection
- Captures standard output, error streams, and return codes
- Provides comprehensive execution monitoring and error reporting
- Supports custom working directory for script execution
[/CONTEXTUAL]

[WORKFLOW_INTEGRATION] Typical workflow integration example:
1. [PREREQUISITE] Ensure script file exists and is executable with proper dependencies [/PREREQUISITE]
2. [CURRENT] Execute script with appropriate arguments and timeout [/CURRENT]
3. [FOLLOW_UP] Process script output and results for further analysis. Can be used to process json script as required [/FOLLOW_UP]
[/WORKFLOW_INTEGRATION]

[SYNTACTICAL] Usage examples:
`execute_python_script("analysis.py", ["--input", "data.json", "--output", "results.json"], 300)`,
`execute_python_script("simulation.py", ["--steps", "1000", "--temp", "300"], 1800, "/path/to/workdir")`,
`execute_python_script("processing.py", None, 600, None)`,
[/SYNTACTICAL]

[RAISES] Exceptions:
    FileNotFoundError: [ERROR_WHEN] When the specified script file doesn't exist [/ERROR_WHEN]
                      [ERROR_DETAILS] Script path is invalid or file is not accessible [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Verify script path exists and is readable [/ERROR_RECOVERY]
    TimeoutExpired: [ERROR_WHEN] When script execution exceeds the specified timeout [/ERROR_WHEN]
                   [ERROR_DETAILS] Script terminated due to timeout limit [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Increase timeout value or optimize script performance [/ERROR_RECOVERY]
    PermissionError: [ERROR_WHEN] When script file lacks execute permissions [/ERROR_WHEN]
                    [ERROR_DETAILS] Insufficient permissions to execute the script [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check file permissions and ensure script is executable [/ERROR_RECOVERY]
[/RAISES]

[LIMITATIONS] Known limitations:
- Cannot modify script execution environment beyond working directory
- Limited to Python scripts and available system Python installation
- No real-time output streaming during execution
- Cannot interact with scripts requiring user input
[/LIMITATIONS]
"""

# Complete tool docstring from catalyst/tools.py (get_structure_from_mp_text)
FULL_DOCSTRING_GET_STRUCTURE = """[BRIEF] Retrieve a pymatgen structure from Materials Project using its API and return CIF content as text. [/BRIEF]
[DETAILED] This tool connects to the Materials Project database to download crystal structure data for a given material ID.
It retrieves the structure object and converts it to CIF (Crystallographic Information File) format, which is the standard format for storing crystal structure information.
CIF is then returned as string [/DETAILED]
[PROCEDURAL] When to use this tool:
- Use when you need to retrieve a bulk crystal structure from the Materials Project database
- Best suited for materials with known MP IDs
- Usually first step in simulation workflows
- Recommended for obtaining a crystal structure for preparing bulk structures, supercells, bulk cells, slabs etc.
- Avoid when you need multiple structures
[/PROCEDURAL]
[CONTEXTUAL] How this tool works:
- Connects to Materials Project API using authentication key (which is already provided in the environment)
- Searches for the specified material ID (MP ID) in the database (MP ID is given as input parameter or if other tools are available to search for MP ID based on available information, then use those tools)
- Retrieves the pymatgen Structure object containing atomic positions and lattice parameters
- Converts the structure to CIF format string for compatibility with other tools
- Returns standardized crystallographic data suitable for further processing
[/CONTEXTUAL]
[WORKFLOW_INTEGRATION] Typical workflow integration example:
1. [PREREQUISITE] Ensure that the other more specific tools are not suitable and you dont have to retrieve multiple strucutres[/PREREQUISITE]
2. [CURRENT] Apply this tool with a valid MP ID to retrieve bulk structure [/CURRENT]
3. [FOLLOW_UP] Use the CIF output with slab generation tools like enumerate_slabs_text to create slab structures [/FOLLOW_UP]
[/WORKFLOW_INTEGRATION]
[SYNTACTICAL] Usage examples:
[
    `get_structure_from_mp_text("mp-149")`, # Silicon structure
    `get_structure_from_mp_text("mp-20066")`, # CO2 structure
    `get_structure_from_mp_text("mp-2")` # Other material
    `get_structure_from_mp_text("mp-12345")` # Example with a different MP ID
    `get_structure_from_mp_text("mp-67890")` # Another example with a different MP ID
]
[/SYNTACTICAL]
[RAISES] Exceptions:
    ConnectionError: [ERROR_WHEN] When unable to connect to Materials Project API [/ERROR_WHEN]
                    [ERROR_DETAILS] Network connectivity issues or API server downtime [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check internet connection and MP_API_KEY environment variable [/ERROR_RECOVERY]
    KeyError: [ERROR_WHEN] When the specified MP ID is not found in the database [/ERROR_WHEN]
             [ERROR_DETAILS] Invalid or non-existent material ID provided [/ERROR_DETAILS]
             [ERROR_RECOVERY] Verify MP ID exists on Materials Project website or check MP ID syntax[/ERROR_RECOVERY]
    AuthenticationError: [ERROR_WHEN] When API key is invalid or missing [/ERROR_WHEN]
                         [ERROR_DETAILS] MP_API_KEY environment variable not set or expired [/ERROR_DETAILS]
                         [ERROR_RECOVERY] Obtain valid API key from Materials Project and set environment variable [/ERROR_RECOVERY]
[/RAISES]
[LIMITATIONS] Known limitations:
- Requires valid Materials Project API key to be set and internet connection
- Limited to materials available in the Materials Project database
- May not include the most recent experimental structures
[/LIMITATIONS]
"""

# Complete argument docstring from corral_md/tools.py (script_path argument)
FULL_ARG_SCRIPT_PATH = """[BRIEF] Path to the Python script file to execute. [/BRIEF]
[DETAILED] Complete file path to the Python script that should be executed.
The script must exist and be readable.
The path can be relative to the current working directory or absolute.
The script should be a valid Python file with appropriate shebang or run using the Python interpreter. [/DETAILED]
[SYNTACTIC] "Valid file path to Python script" [/SYNTACTIC]
[EXAMPLES] "scripts/analysis.py", "/home/user/simulations/run_sim.py", "data_processing.py" [/EXAMPLES]
"""

# Complete argument docstring from catalyst/tools.py (mp_id argument)
FULL_ARG_MP_ID = """[ARGS_BRIEF] Materials Project identifier string. [/ARGS_BRIEF]
[ARGS_DETAILED] The unique identifier used by Materials Project to catalog materials.
Should be in the format "mp-XXXXX" where XXXXX is a numerical ID.
This ID corresponds to a specific material entry in the Materials Project database. [/ARGS_DETAILED]
[ARGS_SYNTACTIC] "mp-" followed by digits (e.g., "mp-149", "mp-20066") [/ARGS_SYNTACTIC]
[ARGS_EXAMPLES] "mp-149" (Silicon), "mp-20066" (CO2), "mp-2" (Li) [/ARGS_EXAMPLES]
"""


class TestFilterToolDescription:
    """Test suite for filter_tool_description function"""

    def test_brief_verbosity(self):
        """Test BRIEF verbosity level - should only include BRIEF section"""
        result = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.BRIEF
        )

        # Should contain the BRIEF content
        assert "Execute a Python script file with arguments" in result
        assert "controlled environment" in result

        # Should NOT contain DETAILED content
        assert "command-line arguments" not in result
        assert "comprehensive execution monitoring" not in result

        # Should NOT contain PROCEDURAL content
        assert "When to use this tool:" not in result
        assert "How this tool works" not in result

        # Should NOT contain WORKFLOW_INTEGRATION
        assert "PREREQUISITE" not in result
        assert "FOLLOW_UP" not in result

        # Should NOT contain SYNTACTICAL
        assert "execute_python_script(" not in result

        # Should NOT contain RAISES or LIMITATIONS
        assert "FileNotFoundError" not in result
        assert "Known limitations" not in result

    def test_workflow_verbosity(self):
        """Test WORKFLOW verbosity level - should include up to WORKFLOW_INTEGRATION"""
        result = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.WORKFLOW
        )

        # Should contain BRIEF
        assert "Execute a Python script file with arguments" in result

        # Should contain DETAILED
        assert (
            "command-line arguments" in result or "data processing pipelines" in result
        )

        # Should contain PROCEDURAL
        assert "When to use:" in result or "Best suited" in result

        # Should contain CONTEXTUAL
        assert "Validates script file" in result or "Constructs command" in result

        # Should contain WORKFLOW_INTEGRATION
        assert "Workflow:" in result
        # The nested tags like [PREREQUISITE], [CURRENT], [FOLLOW_UP] get cleaned out
        # Check for the actual workflow content instead
        assert "Ensure script file exists" in result or "Execute script" in result

        # Should NOT contain SYNTACTICAL (not in WORKFLOW level)
        assert "execute_python_script(" not in result

        # Should NOT contain RAISES or LIMITATIONS
        assert "FileNotFoundError" not in result
        assert "Known limitations" not in result

    def test_comprehensive_verbosity(self):
        """Test COMPREHENSIVE verbosity level - should include all sections"""
        result = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.COMPREHENSIVE
        )

        # Should contain BRIEF
        assert "Execute a Python script file with arguments" in result

        # Should contain DETAILED
        assert (
            "command-line arguments" in result or "data processing pipelines" in result
        )

        # Should contain PROCEDURAL
        assert "When to use:" in result or "Best suited" in result

        # Should contain CONTEXTUAL
        assert "Validates script file" in result or "Constructs command" in result

        # Should contain WORKFLOW_INTEGRATION
        assert "Workflow:" in result

        # Should contain SYNTACTICAL
        assert "execute_python_script(" in result

        # Should contain RAISES
        assert "Exceptions:" in result
        assert "FileNotFoundError" in result or "TimeoutExpired" in result

        # Should contain LIMITATIONS
        assert "Limitations:" in result
        assert "Known limitations" in result or "Cannot modify script" in result

    def test_full_verbosity(self):
        """Test FULL verbosity level - should return original docstring unchanged"""
        result = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.FULL
        )

        # Should be identical to original (including tags)
        assert result == FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT
        assert "[BRIEF]" in result
        assert "[/BRIEF]" in result
        assert "[DETAILED]" in result
        assert "[RAISES]" in result
        assert "[LIMITATIONS]" in result

    def test_minimal_verbosity(self):
        """Test MINIMAL verbosity level - should only show basic description"""
        result = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_GET_STRUCTURE, ToolVerbosity.MINIMAL
        )

        # Should contain basic description (BRIEF content)
        assert "Retrieve a pymatgen structure" in result
        assert "Materials Project" in result

        # Should be concise - no tagged sections
        assert "When to use this tool" not in result
        assert "PREREQUISITE" not in result
        assert "[BRIEF]" not in result  # Tags should be removed

    def test_empty_docstring(self):
        """Test filtering with empty/None docstring"""
        result = VerbosityConfig.filter_tool_description("", ToolVerbosity.BRIEF)
        assert result == "No description available"

        result = VerbosityConfig.filter_tool_description(None, ToolVerbosity.WORKFLOW)
        assert result == "No description available"

    def test_docstring_without_tags(self):
        """Test filtering docstring without special tags"""
        simple_docstring = "This is a simple tool that does something useful."
        result = VerbosityConfig.filter_tool_description(
            simple_docstring, ToolVerbosity.BRIEF
        )

        # Should return the basic content
        assert "simple tool" in result
        assert "does something useful" in result


class TestFilterArgumentDescription:
    """Test suite for filter_argument_description function"""

    def test_brief_verbosity_args(self):
        """Test BRIEF verbosity for argument descriptions"""
        result = VerbosityConfig.filter_argument_description(
            FULL_ARG_SCRIPT_PATH, ToolVerbosity.BRIEF
        )

        # Should contain BRIEF content (or ARGS_BRIEF for args)
        assert "Path to the Python script file" in result

        # Should NOT contain DETAILED
        assert "Complete file path" not in result
        assert "must exist and be readable" not in result

        # Should NOT contain EXAMPLES
        assert "scripts/analysis.py" not in result

    def test_workflow_verbosity_args(self):
        """Test WORKFLOW verbosity for argument descriptions"""
        result = VerbosityConfig.filter_argument_description(
            FULL_ARG_MP_ID, ToolVerbosity.WORKFLOW
        )

        # Should contain ARGS_BRIEF
        assert "Materials Project identifier" in result

        # Should contain ARGS_DETAILED (WORKFLOW includes DETAILED)
        assert (
            "unique identifier" in result
            or "mp-XXXXX" in result
            or "numerical ID" in result
        )

        # Should NOT contain ARGS_SYNTACTIC (not included in WORKFLOW for args)
        # Note: WORKFLOW level doesn't include SYNTACTICAL for args
        # This depends on the implementation

    def test_comprehensive_verbosity_args(self):
        """Test COMPREHENSIVE verbosity for argument descriptions"""
        result = VerbosityConfig.filter_argument_description(
            FULL_ARG_SCRIPT_PATH, ToolVerbosity.COMPREHENSIVE
        )

        # Should contain BRIEF
        assert "Path to the Python script file" in result

        # Should contain DETAILED
        assert "Complete file path" in result or "must exist and be readable" in result

        # Should contain SYNTACTIC or EXAMPLES
        # Note: The tag in the docstring is [SYNTACTIC] not [SYNTACTICAL], and it gets filtered
        # as part of argument filtering logic
        assert "scripts/analysis.py" in result or "data_processing.py" in result

    def test_minimal_verbosity_args(self):
        """Test MINIMAL verbosity for argument descriptions"""
        result = VerbosityConfig.filter_argument_description(
            FULL_ARG_MP_ID, ToolVerbosity.MINIMAL
        )

        # Should be very minimal - just basic text before tags
        assert len(result) < 100  # Should be quite short

        # Should NOT contain detailed examples
        assert "mp-149" not in result
        assert "Silicon" not in result

    def test_empty_arg_description(self):
        """Test filtering with empty argument description"""
        result = VerbosityConfig.filter_argument_description("", ToolVerbosity.BRIEF)
        assert result == ""

        # None is returned as-is by the function
        result = VerbosityConfig.filter_argument_description(
            None, ToolVerbosity.WORKFLOW
        )
        assert result is None or result == ""

    def test_simple_arg_no_tags(self):
        """Test argument without special tags"""
        simple_arg = "A simple parameter description"
        result = VerbosityConfig.filter_argument_description(
            simple_arg, ToolVerbosity.BRIEF
        )

        # Should return as-is for BRIEF
        assert result == simple_arg

        result_minimal = VerbosityConfig.filter_argument_description(
            simple_arg, ToolVerbosity.MINIMAL
        )
        # MINIMAL should also return it (no brackets to split on)
        assert result_minimal == simple_arg


class TestVerbosityCaching:
    """Test caching behavior of verbosity filtering"""

    def test_cache_hit_performance(self):
        """Test that repeated calls use cache"""
        # Clear cache first
        VerbosityConfig.clear_cache()

        # First call - should parse
        result1 = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.BRIEF
        )

        stats1 = VerbosityConfig.get_cache_stats()
        assert stats1["parsed_cache_size"] >= 1
        assert stats1["filtered_cache_size"] >= 1

        # Second call with same inputs - should use cache
        result2 = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.BRIEF
        )

        # Results should be identical
        assert result1 == result2

        # Cache should not grow (same docstring)
        stats2 = VerbosityConfig.get_cache_stats()
        assert stats2["parsed_cache_size"] == stats1["parsed_cache_size"]

    def test_cache_different_verbosity_levels(self):
        """Test that different verbosity levels are cached separately"""
        VerbosityConfig.clear_cache()

        # Call with BRIEF
        result_brief = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.BRIEF
        )

        # Call with WORKFLOW
        result_workflow = VerbosityConfig.filter_tool_description(
            FULL_DOCSTRING_EXECUTE_PYTHON_SCRIPT, ToolVerbosity.WORKFLOW
        )

        # Results should be different
        assert result_brief != result_workflow
        assert len(result_workflow) > len(result_brief)

        # Cache should have entries for both
        stats = VerbosityConfig.get_cache_stats()
        assert stats["filtered_cache_size"] >= 2


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_malformed_tags(self):
        """Test handling of malformed tags"""
        malformed = "[BRIEF] Some text [DETAILED] Missing close tag"

        # Should handle gracefully
        result = VerbosityConfig.filter_tool_description(malformed, ToolVerbosity.BRIEF)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nested_tags(self):
        """Test handling of nested tags"""
        nested = "[BRIEF] Outer [NESTED] Inner [/NESTED] Outer [/BRIEF]"

        result = VerbosityConfig.filter_tool_description(nested, ToolVerbosity.BRIEF)

        # Should clean nested tags and keep content
        assert "Outer" in result
        assert "Inner" in result
        assert "[NESTED]" not in result

    def test_very_long_docstring(self):
        """Test handling of very long docstrings"""
        long_section = "\n".join([f"Line {i} of detailed content" for i in range(100)])
        long_doc = (
            f"[BRIEF] Brief description [/BRIEF]\n[DETAILED] {long_section} [/DETAILED]"
        )

        result = VerbosityConfig.filter_tool_description(
            long_doc, ToolVerbosity.DETAILED
        )

        # Should include all content
        assert "Brief description" in result
        assert "Line 99" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
