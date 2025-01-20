from __future__ import annotations

import tempfile

import uvicorn
from tools import (
    ase_lammps,
    brave_search,
    pickle_to_lammps,
    read_lammps_to_string,
    wikipedia_search,
)

from corral.base import Environment
from corral.server import create_benchmark_server

LAMMPS_TOOLS = [
    read_lammps_to_string,
    ase_lammps,
    pickle_to_lammps,
    brave_search,
    wikipedia_search,
]


class LammpsEnvironment(Environment):
    def __init__(self, task_id: str, question: str):
        self.question = question
        self._tempdir = tempfile.TemporaryDirectory()
        self._path = self._tempdir.name
        super().__init__(task_id)

        # Add multiple tools
        for tool in LAMMPS_TOOLS:
            self.add_tool(tool)

    def get_task_prompt(self) -> str:
        return f"Solve this problem you can use the path {self._path} to read and write if you want: {self.question}"

    def _validate_lammps_data(self, data_string: str) -> tuple[bool, str]:
        """
        Validate if string contains valid LAMMPS data file format.

        Args:
            data_string: String containing LAMMPS data file content

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            from io import StringIO

            from ase.io import read

            # Try parsing with ASE
            atoms = read(StringIO(data_string), format="lammps-data")

            # Basic validation checks
            if atoms is None:
                return False, "Failed to parse atoms object"
            if len(atoms) == 0:
                return False, "No atoms found in data"
            if not all(atoms.get_cell().any()):
                return False, "Invalid or missing cell parameters"

            return True, "Valid LAMMPS data format"

        except Exception as e:
            return False, f"Invalid format: {e!s}"

    def score(self) -> float:
        """Score based on submitted answer"""
        if self.state.submitted_answer is None:
            return 0.0
        try:
            submitted_result = self.state.submitted_answer
            # return 1.0 if type(submitted_result) == str else 0.0
            # validate with ase if the string is valid lammps file
            is_valid, message = self._validate_lammps_data(submitted_result)
            return 1.0 if is_valid else 0.0

        except ValueError:
            return 0.0


if __name__ == "__main__":
    # Create environments for different tasks
    environments = {
        "lammps_1": LammpsEnvironment(
            "input_1",
            "Write input file for LAMMPS. I want to simulate the FCC gold unitcell. Give me the lammps input file as string",
        ),
    }

    # Create and run server
    app = create_benchmark_server(environments)

    uvicorn.run(app, host="0.0.0.0", port=8000)
