from __future__ import annotations

import os
import subprocess
from pathlib import Path

import fsspec
import modal
from envs_tools.lammps import _run_lammps, lammps_image
from modal import App

simagent_name = os.getenv("SIMAGENT_NAME", "")
if simagent_name and not simagent_name.startswith("-"):
    simagent_name = f"-{simagent_name}"

# Create the app
app = App(f"simagent{simagent_name}")
volume_potential = modal.Volume.from_name("potentials", create_if_missing=True)
volume_sim = modal.Volume.from_name("simulations", create_if_missing=True)
volume_struct = modal.Volume.from_name("structures", create_if_missing=True)
volume_test_files = modal.Volume.from_name("test_files", create_if_missing=True)

CPUS = 1

# with volume_potential.batch_upload() as batch:
#     batch.put_directory("./potentials/", "/")

# with volume_struct.batch_upload() as batch:
#     batch.put_directory("./structures/", "/")

# with volume_test_files.batch_upload() as batch:
#     batch.put_directory("./test_files/", "/")


@app.function(
    image=lammps_image,
    cpu=CPUS,
    timeout=3600,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
        "/test_files": volume_test_files,
    },
)
def run_lammps(input_file: str, log_file: str) -> None:
    """
    Run a LAMMPS simulation.

    Args:
        input_file (str): Path to the LAMMPS input script file.
        log_file (str): Path where the log file output will be stored.


    Raises:
        ValueError: If the simulation fails.
    """
    volume_sim.reload()
    input_path = Path(input_file)
    directory_path = input_path.parent
    input_file_ = input_path.name
    try:
        _run_lammps(input_file_, log_file, str(directory_path), CPUS)
        volume_sim.commit()

    except Exception as e:
        raise ValueError(f"LAMMPS simulation failed: {e!s}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def convert_structure_to_lammps_data(
    structure: str, output_file: str, atom_style: str = "charge"
) -> None:
    """
    Convert a CIF-format structure into a LAMMPS data file. This function takes a CIF string as input (assumed to be the output from the previous tool),
    converts it to a pymatgen Structure object, and then generates the corresponding LAMMPS data
    file.

    Args:
        structure: The CIF-format structure as a string.
        output_file: The path to the output LAMMPS data file to be written.
        atom_style: The atom style for LAMMPS. Common values include:
            - "atomic": Basic atomic positions and masses.
            - "charge": Atoms with charge data (default).

    Returns:
        None. Writes the LAMMPS data to the specified output file.
    """
    from pymatgen.core import Structure
    from pymatgen.io.lammps.data import LammpsData

    # Convert CIF string to pymatgen Structure object
    volume_sim.reload()
    try:
        # Load structure from CIF file
        structure_obj = Structure.from_file(structure)

        # Check if the structure contains ONLY Si
        elements = {str(el) for el in structure_obj.composition.elements}
        is_silicon_only = elements == {"Si"} or elements == {"Cu"} or elements == {"Al"}

        # Convert to LAMMPS data
        lammps_data = LammpsData.from_structure(structure_obj, atom_style=atom_style)
        lammps_data.write_file(output_file)

        # Only modify the file if it's pure silicon
        if is_silicon_only:
            with Path(output_file).open() as f:
                lines = f.readlines()

            # Insert tilt line after zlo zhi
            for i, line in enumerate(lines):
                if "zlo zhi" in line:
                    lines.insert(i + 1, "0.0 0.0 0.0 xy xz yz\n")
                    break

            # Write final file
            with Path(output_file).open("w") as f:
                f.writelines(lines)

        volume_sim.commit()

    except Exception as e:
        raise ValueError(
            f"An unexpected error occurred while converting to Lammps data format: {e!s}"
        ) from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def run_bash_command(
    command: str, args: list[str] | None = None, shell: bool = False
) -> str:
    """Execute a bash command with optional arguments.

    Args:
        command: The bash command to execute (e.g., 'mkdir', 'ls').
        args: A list of arguments for the bash command.
    """
    # volume_potential.reload()

    volume_sim.reload()
    if args is None:
        args = []
    try:
        # Construct the full command with arguments
        full_command = [command, *args]

        # Run the command
        result = subprocess.run(
            full_command,
            shell=shell,  # Avoid shell injection risks
            check=True,  # Raise CalledProcessError on failure
            capture_output=True,
            text=True,
        )

        # Return the standard output
        volume_sim.commit()
        return result.stdout

    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e!s}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def list_files(path: str, recursive: bool = False) -> list[str]:
    """List files in a directory"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        return fs.ls(path, detail=False, recursive=recursive)
    except Exception as e:
        raise RuntimeError(f"Error listing files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def read_file(path: str) -> str:
    """Read contents of a file"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "r") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def read_large_file(path: str, start: int, length: int, encoding: str) -> str:
    """Read specific portion of a large file"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(length)
        return chunk.decode(encoding, errors="replace")
    except Exception as e:
        f"[ERROR] Could not read file chunk: {e}"


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def write_file(path: str, content: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "w") as f:
            f.write(content)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error writing files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def file_info(path: str) -> dict:
    """Get file information"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        return fs.info(path)
    except Exception as e:
        raise RuntimeError(f"Error getting file info: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def copy_file(source: str, destination: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        fs.copy(source, destination)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error copying from {source} to {destination}: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def move_file(source: str, destination: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        # fsspec does not always provide a move method; if not, copy then remove.
        fs.mv(source, destination)
        volume_sim.commit()
    except Exception:
        copy_file(source, destination)
        fs.rm(source)


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def mkdir(path: str, create_parents: bool = False) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        if create_parents and hasattr(fs, "mkdirs"):
            # Many fsspec implementations support mkdirs.
            # If not available, fall back to calling mkdir for each missing part.
            fs.mkdirs(path, exist_ok=True)
        else:
            fs.mkdir(path)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error creating directory {path}: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
    },
)
def cat_files(paths: list[str], separator: str = "\n") -> str:
    contents = []
    for path in paths:
        try:
            contents.append(read_file(path))
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {e}") from e
    return separator.join(contents)
