def get_structure_from_mp_text(mp_id: str, file_path: str) -> str:
    try:
        from mp_api.client import MPRester
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        import fsspec


        with MPRester("3pKn435e6fN6hfcOEfpYX4OqbnfV1MKB") as mpr:
            docs = mpr.materials.summary.search(
                material_ids=[str(mp_id)], fields=["structure"]
            )
            structure = docs[0].structure

        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()
        structure_cif = structure.to(fmt="cif")
        fs = fsspec.filesystem("file")
        with fs.open(file_path, "w") as f:
            f.write(structure_cif)

        return f"Structure saved successfully at {file_path}"

    except Exception as e:
        return f"Failed to retrieve or save structure: {e!s}"

def convert_structure_to_lammps_data(
    structure_path: str, output_file: str, atom_style: str = "charge"
) -> str:
    from pymatgen.core import Structure
    from pymatgen.io.lammps.data import LammpsData
    try:
        structure_obj = Structure.from_file(structure_path)

        # Convert the Structure to LAMMPS data format
        lammps_data = LammpsData.from_structure(structure_obj, atom_style=atom_style)
        lammps_data.write_file(output_file)
        return f"LAMMPS data file successfully written to: {output_file}"
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while converting structure to LAMMPS data: {e!s}"
        ) from e


get_structure_from_mp_text("mp-153", "./ground_truth/melting/Magnesium/Mg.cif")
convert_structure_to_lammps_data("./ground_truth/melting/Magnesium/Mg.cif", "./ground_truth/structures/Mg.data", "atomic")
