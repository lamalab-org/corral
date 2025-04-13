from __future__ import annotations

import os

import shutil

from modal import App
import modal
import subprocess
import fsspec
import numpy as np
import matplotlib.pyplot as plt


from general_tools.lammps import (
    _run_lammps,
    lammps_image
)

simagent_name = os.getenv("SIMAGENT_NAME", "")
if simagent_name and not simagent_name.startswith("-"):
    simagent_name = f"-{simagent_name}"

# Create the app
app = App(f"simagent{simagent_name}")
volume_potential = modal.Volume.from_name("potentials", create_if_missing=True)
volume_sim = modal.Volume.from_name("simulations", create_if_missing=True)
volume_struct = modal.Volume.from_name("structures", create_if_missing=True)

# with volume_potential.batch_upload() as batch:
#     batch.put_directory("/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/modal_app/potentials/", "/")

# with volume_struct.batch_upload() as batch:
#     batch.put_directory("/Users/chandan21gupta/Desktop/iit_delhi/agent_llms_3/mat-agent-bench/modal_app/structures/", "/")

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})
def plot_stress_strain(data_file: str, output_image: str) -> str:
    """
    Generate a stress-strain curve plot from a text file and save it as an image.
    Plots stress in x, y, and z directions.

    Args:
        data_file: Path to the text file containing strain and stress values.
        output_image: Path where the image will be saved (e.g., "plot.png").
    """

    try:
        # Load comma or whitespace-separated data
        try:
            data = np.loadtxt(data_file, delimiter=",", skiprows=1)
        except:
            data = np.loadtxt(data_file, skiprows=1)

        strain = data[:, 0]
        stress_x = data[:, 1]
        stress_y = data[:, 2]
        stress_z = data[:, 3]

        plt.figure(figsize=(6, 4))
        plt.plot(strain, stress_x, label="Stress XX", color="steelblue")
        plt.plot(strain, stress_y, label="Stress YY", color="seagreen")
        plt.plot(strain, stress_z, label="Stress ZZ", color="darkorange")

        plt.xlabel("Strain")
        plt.ylabel("Stress (GPa)")
        plt.title("Uniaxial Stress-Strain Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_image)
        plt.close()

        return f"Plot saved successfully to {os.path.abspath(output_image)}"

    except FileNotFoundError:
        raise ValueError(f"Data file not found at {data_file}")
    except Exception as e:
        raise Exception(f"Failed to generate plot: {str(e)}")


@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})
def run_lammps(input_file: str, log_file: str) -> None:
    """
    Run a LAMMPS simulation.

    Args:
        input_file (str): Path to the LAMMPS input script file.
        log_file (str): Path where the log file output will be stored.


    Raises:
        ValueError: If the simulation fails.
    """
    # volume_potential.reload()
    volume_sim.reload()
    try:
        # if save_path:
        #     os.chdir(save_path)
        # print("changed directory", os.getcwd())
        _run_lammps(input_file, log_file)
        volume_sim.commit()
        # return output_dict
        # if save_path:
        #     os.makedirs(save_path, exist_ok=True)
        #     for file_name, content in output_dict.items():
        #         file_path = os.path.join(save_path, file_name)
        #         with open(file_path, "w") as file:
        #             file.write(content)
    except Exception as e:
        raise ValueError(f"LAMMPS simulation failed: {str(e)}")
    finally:
        # volume_potential.commit()
        # os.chdir(original_cwd)
        # volume_sim.commit()
        pass

    # try:
    #     os.chdir(path)
    # finally:
    #     os.chdir(original_cwd)

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})
def list_files(path: str, recursive: bool = False) -> list[str]:
    """List files in a directory"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        return fs.ls(path, detail=False, recursive=recursive)
    except Exception as e:
        raise RuntimeError(f"Error listing files: {e}") from e
    
@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})
def read_file(path: str) -> str:
    """Read contents of a file"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "r") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading files: {e}") from e

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})
def write_file(path: str, content: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "w") as f:
            f.write(content)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error writing files: {e}") from e
   
@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})    
def file_info(path: str) -> dict:
    """Get file information"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        return fs.info(path)
    except Exception as e:
        raise RuntimeError(f"Error getting file info: {e}") from e

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})    
def copy_file(source: str, destination: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        fs.copy(source, destination)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(
            f"Error copying from {source} to {destination}: {e}"
        ) from e

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})    
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

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})    
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

@app.function(image=lammps_image, cpu=1.0, memory=5120, volumes = {'/potentials' : volume_potential, '/results' : volume_sim, '/structures' : volume_struct})    
def cat_files(paths: list[str], separator: str = "\n") -> str:
    contents = []
    for path in paths:
        try:
            contents.append(read_file(path))
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {e}") from e
    return separator.join(contents)



    
    
