from pymatgen.core import Structure


def check_cif_structure(cif_text: str) -> float:
    """
    Verify that a CIF string represents a valid structure.
    Returns 1.0 if valid (with at least one site), else 0.0.
    """
    try:
        structure = Structure.from_str(cif_text, fmt="cif")
        return 1.0 if structure and len(structure) > 0 else 0.0
    except Exception:
        return 0.0


def check_slabs_json(slabs_json: str) -> float:
    """
    Check that the slabs JSON contains at least one valid slab by trying to parse
    the CIF string for one of the slabs.
    """
    import json

    from pymatgen.core import Structure

    try:
        slabs = json.loads(slabs_json)
        if not slabs:
            return 0.0
        # Choose one slab and validate
        for cif in slabs.values():
            try:
                struct = Structure.from_str(cif, fmt="cif")
                if struct and len(struct) > 0:
                    return 1.0
            except Exception:
                continue
        return 0.0
    except Exception:
        return 0.0


def check_adsorption_sites(sites_json: str) -> float:
    """
    Check that the adsorption sites output from get_adsorption_sites is valid.
    We simply ensure it is parsed as a dict and has at least one key.
    """
    import json

    try:
        sites = json.loads(sites_json)
        return 1.0 if isinstance(sites, dict) and sites else 0.0
    except Exception:
        return 0.0


def check_combined_structure(cif_text: str) -> float:
    """
    Check that the structure (after adding an adsorbate) is valid.
    """
    return check_cif_structure(cif_text)
