from pymatgen.core import Structure


def load_structure(
    bulk_structure_path_or_string: str, from_path: bool = False
) -> Structure:
    """Loads pymatgen structure from CIF string."""
    from pymatgen.core import Structure

    try:
        if from_path:
            return Structure.from_file(bulk_structure_path_or_string)
        else:
            from pymatgen.core import Structure

            return Structure.from_str(bulk_structure_path_or_string, fmt="cif")
    except Exception as e:
        raise ValueError(f"Failed to parse structure CIF: {e}") from e
