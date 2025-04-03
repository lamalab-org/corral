import ast
import os
from pathlib import Path

import numpy as np
from mp_api.client import MPRester

API_KEY = os.environ.get("MP_API_KEY")


def retrieve_material_id(material: str) -> list:
    """
    Retrieves the material ID for a given material formula.

    Args:
        material: Material formula, i.e. "NaCl"

    Returns:
        list: Material IDs
    """
    with MPRester(API_KEY) as mpr:
        search_results = mpr.materials.search(
            formula=material, fields=["material_id", "formula_pretty"]
        )

    return [result.material_id for result in search_results]


def retrieve_material_property(material: str, property_type: str) -> list:
    """
    Retrieves specified property for a given material formula.

    Args:
        material: Material formula, i.e. "NaCl"
        property_type: Type of property to retrieve ('elastic_tensor', 'bandgap', etc.)

    Returns:
        list: List of tuples containing (material_id, property_value) for the material
    """
    material_ids = retrieve_material_id(material)
    property_values = []

    for material_id in material_ids:
        with MPRester(API_KEY) as mpr:
            if property_type == "elastic_tensor":
                elasticity_data = mpr.elasticity.get_data_by_id(material_id)
                if elasticity_data:
                    property_values.append(
                        (material_id, str(elasticity_data.elastic_tensor).strip())
                    )

            elif property_type == "bandgap":
                docs = mpr.materials.summary.search(
                    material_ids=[material_id], fields=["material_id", "band_gap"]
                )
                if docs and docs[0].band_gap is not None:
                    property_values.append((material_id, docs[0].band_gap))

            elif property_type == "formation_energy":
                docs = mpr.materials.summary.search(
                    material_ids=[material_id],
                    fields=["material_id", "formation_energy_per_atom"],
                )
                if docs and docs[0].formation_energy_per_atom is not None:
                    property_values.append(
                        (material_id, docs[0].formation_energy_per_atom)
                    )

            elif property_type == "nsites":
                docs = mpr.materials.summary.search(
                    material_ids=[material_id], fields=["material_id", "nsites"]
                )
                if docs and docs[0].nsites is not None:
                    property_values.append((material_id, docs[0].nsites))

            elif property_type == "bulk_modulus":
                elasticity_data = mpr.elasticity.get_data_by_id(material_id)
                if elasticity_data and hasattr(elasticity_data, "bulk_modulus"):
                    # Use the VRH (Voigt-Reuss-Hill) average as it's generally preferred
                    property_values.append(
                        (material_id, elasticity_data.bulk_modulus.vrh)
                    )

            else:
                raise ValueError(f"Property type {property_type} not supported")

    return str(property_values)


def retrieve_elastic_tensors(material: str) -> list[str]:
    """
    Retrieves the elastic tensors for a given material.

    Args:
        material: Material name, i.e. "NaCl"

    Returns:
        list[str]: List of elastic tensors
    """
    results = retrieve_material_property(material, "elastic_tensor")
    return [tensor for _, tensor in results]


def retrieve_bandgap_values(material: str) -> list[float]:
    """
    Retrieves the bandgap values for a given material.

    Args:
        material: Material name, i.e. "NaCl"

    Returns:
        list[float]: List of bandgap values
    """
    results = retrieve_material_property(material, "bandgap")
    return [bandgap for _, bandgap in results]


def retrieve_formation_energy(material: str) -> list[float]:
    """
    Retrieves the formation energy values for a given material.

    Args:
        material: Material name, i.e. "NaCl"

    Returns:
        list[float]: List of formation energy values
    """
    results = retrieve_material_property(material, "formation_energy")
    return [energy for _, energy in results]


def check_elastic_tensor(elastic_tensor: str, material: str) -> float:
    """
    Validates if the given elastic tensor matches any reference elastic tensor.

    Performs three types of checks:
    1. Exact string matching
    2. IEEE format matching
    3. Numerical comparison with tolerance

    Args:
        elastic_tensor: String representation of the elastic tensor to check
        material: Material name, i.e. "NaCl"

    Returns:
        float: 1.0 if the tensor matches any reference tensor, 0.0 otherwise
    """
    elastic_tensors = retrieve_elastic_tensors(material)

    # Direct string match
    if elastic_tensor in elastic_tensors:
        return float(True)

    try:
        input_ieee = None
        if "ieee_format=" in elastic_tensor:
            parts = elastic_tensor.split("ieee_format=")
            if len(parts) > 1:
                input_ieee = parts[1].strip()

        # IEEE format comparison
        if input_ieee:
            for tensor in elastic_tensors:
                if "ieee_format=" in tensor:
                    tensor_parts = tensor.split("ieee_format=")
                    if len(tensor_parts) > 1:
                        tensor_ieee = tensor_parts[1].strip()
                        if input_ieee == tensor_ieee:
                            return float(True)

        input_tensor = None
        if "ieee_format=" in elastic_tensor:
            tensor_str = elastic_tensor.split("ieee_format=")[0].strip()
            if tensor_str:
                input_tensor = ast.literal_eval(tensor_str)
        else:
            input_tensor = ast.literal_eval(elastic_tensor)

        if input_tensor is None:
            return float(False)

        input_array = np.array(input_tensor, dtype=float)

        # Validate tensor shape
        if input_array.shape != (6, 6):
            return float(False)

        # Numerical comparison with reference tensors
        for tensor in elastic_tensors:
            ref_tensor = None
            if "ieee_format=" in tensor:
                tensor_str = tensor.split("ieee_format=")[0].strip()
                if tensor_str:
                    ref_tensor = ast.literal_eval(tensor_str)
            else:
                ref_tensor = ast.literal_eval(tensor)

            if ref_tensor is None:
                continue

            ref_array = np.array(ref_tensor, dtype=float)

            # Compare tensors with tolerance
            if np.allclose(input_array, ref_array, rtol=1e-5, atol=1e-8):
                return float(True)

    except (ValueError, SyntaxError, TypeError, IndexError):
        return float(False)

    return float(False)


def check_elastic_tensor_file(file_path: str, material: str) -> float:
    """
    Validates if the elastic tensor in the given file matches any reference elastic tensor.

    Args:
        file_path: Path to the file containing the elastic tensor

    Returns:
        float: 1.0 if the tensor matches any reference tensor, 0.0 otherwise
    """
    try:
        with Path.open(file_path) as f:
            elastic_tensor = f.read()
            return check_elastic_tensor(elastic_tensor, material)
    except (OSError, FileNotFoundError):
        return float(False)


def check_simple_bandgap(bandgap: float, material: str) -> float:
    """
    Validates if the given bandgap is within the expected range.

    Args:
        bandgap: Bandgap value to check
        material: Material name, i.e. "NaCl"

    Returns:
        float: 1.0 if the bandgap is within the expected range, 0.0 otherwise
    """
    if not isinstance(bandgap, float):
        return float(False)
    bandgap_values = retrieve_bandgap_values(material)
    for bandgap_value in bandgap_values:
        if bandgap_value == bandgap:
            return float(True)
    return float(False)


def check_simple_formation_energy(formation_energy: float, material: str) -> float:
    """
    Validates if the given formation energy is within the expected range.

    Args:
        formation_energy: Formation energy value to check
        material: Material name, i.e. "NaCl"

    Returns:
        float: 1.0 if the formation energy is within the expected range, 0.0 otherwise
    """
    if not isinstance(formation_energy, float):
        return float(False)

    formation_energy_values = retrieve_formation_energy(material)
    for formation_energy_value in formation_energy_values:
        if formation_energy_value == formation_energy:
            return float(True)
    return float(False)


def check_bandgap_formation_energy(
    material: str,
    bandgap_low: float,
    bandgap_high: float,
    formation_energy_low: float,
    formation_energy_high: float,
    max_atoms: int,
) -> float:
    """
    Validates if the given material meets the criteria of having a bandgap between the specified range, a formation energy between the specified range, and a maximum number of atoms in the unit cell.

    Args:
        material: Material name, i.e. "NaCl"

    Returns:
        float: 1.0 if the material meets the criteria, 0.0 otherwise
    """
    nsites_data = retrieve_material_property(material, "nsites")
    bandgap_data = retrieve_material_property(material, "bandgap")
    formation_energy_data = retrieve_material_property(material, "formation_energy")

    nsites_dict = dict(nsites_data)
    bandgap_dict = dict(bandgap_data)
    formation_energy_dict = dict(formation_energy_data)

    for material_id in nsites_dict:
        if material_id not in bandgap_dict or material_id not in formation_energy_dict:
            continue

        nsites = nsites_dict[material_id]
        bandgap = bandgap_dict[material_id]
        formation_energy = formation_energy_dict[material_id]

        if (
            nsites <= max_atoms
            and bandgap_low <= bandgap <= bandgap_high
            and formation_energy_low <= formation_energy <= formation_energy_high
        ):
            return float(True)

    return float(False)


def check_bandgap_bulk_modulus(
    material: str,
    bandgap_low: float,
    bandgap_high: float,
    bulk_modulus_low: float,
    bulk_modulus_high: float,
) -> float:
    """
    Validates if the given material meets the criteria of having a bandgap between the specified range and a bulk modulus between the specified range.

    Args:
        material: Material name, i.e. "NaCl"
        bandgap_low: Lower bound of the bandgap range
        bandgap_high: Upper bound of the bandgap range
        bulk_modulus_low: Lower bound of the bulk modulus range
        bulk_modulus_high: Upper bound of the bulk modulus range

    Returns:
        float: 1.0 if the material meets the criteria, 0.0 otherwise
    """
    bandgap_data = retrieve_material_property(material, "bandgap")
    bulk_modulus_data = retrieve_material_property(material, "bulk_modulus")

    bandgap_dict = dict(bandgap_data)
    bulk_modulus_dict = dict(bulk_modulus_data)

    for material_id in bandgap_dict:
        if material_id not in bulk_modulus_dict:
            continue

        bandgap = bandgap_dict[material_id]
        bulk_modulus = bulk_modulus_dict[material_id]

        if (
            bandgap_low <= bandgap <= bandgap_high
            and bulk_modulus_low <= bulk_modulus <= bulk_modulus_high
        ):
            return float(True)

    return float(False)


def check_cif_material(cif_path: str) -> float:
    """
    Validates if the path contains a CIF file for a material that meets the criteria of having a bandgap between the specified range, a formation energy between the specified range, and a maximum number of atoms in the unit cell.

    Args:
        cif_path: Path to the CIF file

    Returns:
        float: 1.0 if the CIF file meets the criteria, 0.0 otherwise
    """
    try:
        with Path.open(cif_path) as f:
            cif_content = f.read()
            if not cif_content:
                return float(False)
            else:
                return float(True)
    except (OSError, FileNotFoundError):
        return float(False)


def check_bandgap_formation_bulk_modulus(
    material: str,
    bandgap_low: float,
    bandgap_high: float,
    formation_energy_low: float,
    formation_energy_high: float,
    bulk_modulus_low: float,
    bulk_modulus_high: float,
) -> float:
    """
    Validates if the given material meets the criteria of having a bandgap between the specified range, a formation energy between the specified range, and a bulk modulus between the specified range.

    Args:
        material: Material name, i.e. "NaCl"
        bandgap_low: Lower bound of the bandgap range
        bandgap_high: Upper bound of the bandgap range
        formation_energy_low: Lower bound of the formation energy range
        formation_energy_high: Upper bound of the formation energy range
        bulk_modulus_low: Lower bound of the bulk modulus range
        bulk_modulus_high: Upper bound of the bulk modulus range

    Returns:
        float: 1.0 if the material meets the criteria, 0.0 otherwise
    """
    bandgap_data = retrieve_material_property(material, "bandgap")
    formation_energy_data = retrieve_material_property(material, "formation_energy")
    bulk_modulus_data = retrieve_material_property(material, "bulk_modulus")

    bandgap_dict = dict(bandgap_data)
    formation_energy_dict = dict(formation_energy_data)
    bulk_modulus_dict = dict(bulk_modulus_data)

    for material_id in bandgap_dict:
        if (
            material_id not in formation_energy_dict
            or material_id not in bulk_modulus_dict
        ):
            continue

        bandgap = bandgap_dict[material_id]
        formation_energy = formation_energy_dict[material_id]
        bulk_modulus = bulk_modulus_dict[material_id]

        if (
            bandgap_low <= bandgap <= bandgap_high
            and formation_energy_low <= formation_energy <= formation_energy_high
            and bulk_modulus_low <= bulk_modulus <= bulk_modulus_high
        ):
            return float(True)

    return float(False)
