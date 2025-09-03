import modal
from loguru import logger
from rdkit import Chem

get_raw_proton_spectrum = modal.Function.lookup("chemenv", "get_raw_h_nmr_prediction")
get_raw_carbon_spectrum = modal.Function.lookup("chemenv", "get_raw_c_nmr_prediction")


def parse_molfile_to_mol(molfile_string: str) -> Chem.Mol:
    """Convert molfile string to RDKit molecule object."""
    mol = Chem.MolFromMolBlock(molfile_string)
    if mol is None:
        raise ValueError("Invalid molfile provided")
    return mol


def normalize_molecule_charges(mol: Chem.Mol) -> Chem.Mol:
    """Normalize molecule by removing formal charges for substructure matching."""
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetFormalCharge(0)
    return mol_copy


def handle_ionic_fragment(fragment_smiles: str) -> tuple[str, dict]:
    """Handle ionic fragments by providing alternative SMILES representations."""
    ion_info = {"is_ion": False, "charge": 0, "original_smiles": fragment_smiles}

    try:
        mol = Chem.MolFromSmiles(fragment_smiles)
        if mol is not None:
            total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if total_charge != 0:
                ion_info["is_ion"] = True
                ion_info["charge"] = total_charge

                # Create neutral version for matching
                neutral_mol = Chem.Mol(mol)
                for atom in neutral_mol.GetAtoms():
                    atom.SetFormalCharge(0)

                neutral_smiles = Chem.MolToSmiles(neutral_mol)
                return neutral_smiles, ion_info
    except Exception:
        pass

    return fragment_smiles, ion_info


def create_atom_mapping(
    original_mol: Chem.Mol, fragment_mol: Chem.Mol
) -> dict[int, int]:
    """Create a mapping between fragment atom indices and original molecule atom indices."""

    # Strategy 1: Direct matching
    matches = original_mol.GetSubstructMatches(fragment_mol)
    if matches:
        match = matches[0]
        return dict(enumerate(match))

    # Strategy 2: Charge-normalized matching
    try:
        original_neutral = normalize_molecule_charges(original_mol)
        fragment_neutral = normalize_molecule_charges(fragment_mol)
        matches = original_neutral.GetSubstructMatches(fragment_neutral)
        if matches:
            match = matches[0]
            return dict(enumerate(match))
    except Exception:
        pass

    # Strategy 3: Element-based matching for simple cases
    try:
        original_elements = [atom.GetSymbol() for atom in original_mol.GetAtoms()]
        fragment_elements = [atom.GetSymbol() for atom in fragment_mol.GetAtoms()]

        # Single atom fragment
        if len(fragment_elements) == 1:
            target_element = fragment_elements[0]
            for i, element in enumerate(original_elements):
                if element == target_element:
                    return {0: i}

        # Multi-atom fragments
        if len(fragment_elements) <= len(original_elements):
            for start_idx in range(len(original_elements) - len(fragment_elements) + 1):
                if (
                    original_elements[start_idx : start_idx + len(fragment_elements)]
                    == fragment_elements
                ):
                    atom_mapping = {}
                    for i, fragment_idx in enumerate(range(len(fragment_elements))):
                        atom_mapping[fragment_idx] = start_idx + i
                    return atom_mapping
    except Exception:
        pass

    return {}


def analyze_fragment_mapping(
    fragment_smiles: str, molfile_string: str, ground_spectra: dict
) -> dict:
    """
    Clean and comprehensive analysis of fragment mapping to NMR spectra.

    Args:
        fragment_smiles: SMILES string of the fragment (can be ionic)
        molfile_string: Molfile string of the original molecule
        ground_spectra: Dictionary containing NMR spectra data

    Returns:
        Clean dictionary with fragment mapping and signal analysis
    """
    try:
        # Parse original molecule
        original_mol = parse_molfile_to_mol(molfile_string)

        # Handle ionic fragments
        processed_fragment_smiles, ion_info = handle_ionic_fragment(fragment_smiles)
        fragment_mol = Chem.MolFromSmiles(processed_fragment_smiles)

        if fragment_mol is None:
            fragment_mol = Chem.MolFromSmiles(fragment_smiles)
            if fragment_mol is None:
                return {
                    "success": False,
                    "error": "Invalid fragment SMILES",
                    "fragment_info": None,
                    "spectra_analysis": {},
                }

        # Create atom mapping
        atom_mapping = create_atom_mapping(original_mol, fragment_mol)

        if not atom_mapping:
            return {
                "success": False,
                "error": "Fragment not found in original molecule",
                "fragment_info": {
                    "smiles": fragment_smiles,
                    "is_ion": ion_info["is_ion"],
                    "charge": ion_info["charge"],
                },
                "spectra_analysis": {},
            }

        # Get fragment atoms in original molecule
        fragment_atoms_in_original = set(atom_mapping.values())

        # Create clean fragment info
        fragment_atom_details = []
        for fragment_idx, original_idx in atom_mapping.items():
            original_atom = original_mol.GetAtomWithIdx(original_idx)
            fragment_atom_details.append(
                {
                    "fragment_position": fragment_idx,
                    "original_position": original_idx,
                    "element": original_atom.GetSymbol(),
                    "formal_charge": original_atom.GetFormalCharge(),
                }
            )

        fragment_info = {
            "smiles": fragment_smiles,
            "is_ion": ion_info["is_ion"],
            "charge": ion_info["charge"],
            "atoms_in_original": sorted(fragment_atoms_in_original),
            "atom_details": fragment_atom_details,
        }

        # Analyze each spectra type
        spectra_analysis = {}

        for spectra_type, spectra_data in ground_spectra.items():
            if (
                not isinstance(spectra_data, dict)
                or "joinedSignals" not in spectra_data
            ):
                spectra_analysis[spectra_type] = {
                    "valid": False,
                    "error": "No joinedSignals found",
                }
                continue

            signals_in_fragment = []
            signals_not_in_fragment = []
            signals_partially_in_fragment = []

            for signal in spectra_data["joinedSignals"]:
                signal_atoms = signal.get("atoms", [])

                # Check which atoms are in fragment
                atoms_in_fragment = [
                    atom for atom in signal_atoms if atom in fragment_atoms_in_original
                ]

                # Get element information for atoms
                atom_elements = [
                    {
                        "position": atom_idx,
                        "element": original_mol.GetAtomWithIdx(atom_idx).GetSymbol(),
                        "in_fragment": atom_idx in fragment_atoms_in_original,
                    }
                    for atom_idx in signal_atoms
                    if atom_idx < original_mol.GetNumAtoms()
                ]

                signal_info = {
                    "signal_id": signal.get("id", "unknown"),
                    "delta": signal.get("delta"),
                    "multiplicity": signal.get("multiplicity"),
                    "atom_count": len(signal_atoms),
                    "atoms": atom_elements,
                    "atoms_in_fragment": atoms_in_fragment,
                    "fragment_atom_count": len(atoms_in_fragment),
                }

                # Classify signal
                if (
                    len(atoms_in_fragment) == len(signal_atoms)
                    and len(atoms_in_fragment) > 0
                ):
                    # All atoms in fragment
                    signals_in_fragment.append(signal_info)
                elif len(atoms_in_fragment) > 0:
                    # Some atoms in fragment
                    signals_partially_in_fragment.append(signal_info)
                else:
                    # No atoms in fragment
                    signals_not_in_fragment.append(signal_info)

            spectra_analysis[spectra_type] = {
                "valid": True,
                "signals_fully_in_fragment": signals_in_fragment,
                "signals_partially_in_fragment": signals_partially_in_fragment,
                "signals_not_in_fragment": signals_not_in_fragment,
                "summary": {
                    "total_signals": len(spectra_data["joinedSignals"]),
                    "fully_in_fragment": len(signals_in_fragment),
                    "partially_in_fragment": len(signals_partially_in_fragment),
                    "not_in_fragment": len(signals_not_in_fragment),
                },
            }

        return {
            "success": True,
            "fragment_info": fragment_info,
            "spectra_analysis": spectra_analysis,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fragment_info": None,
            "spectra_analysis": {},
        }


def get_fragment_signals_summary(analysis_result: dict) -> dict:
    """
    Extract a clean summary of signals that belong to the fragment.

    Args:
        analysis_result: Result from analyze_fragment_mapping

    Returns:
        Clean summary of fragment signals
    """
    if not analysis_result["success"]:
        return {
            "success": False,
            "error": analysis_result.get("error", "Unknown error"),
        }

    fragment_signals = {}

    for spectra_type, spectra_data in analysis_result["spectra_analysis"].items():
        if spectra_data["valid"]:
            # Get signals that are fully or partially in fragment
            relevant_signals = (
                spectra_data["signals_fully_in_fragment"]
                + spectra_data["signals_partially_in_fragment"]
            )

            clean_signals = []
            for signal in relevant_signals:
                # Extract only fragment atoms and their elements
                fragment_atoms_info = [
                    atom for atom in signal["atoms"] if atom["in_fragment"]
                ]

                clean_signal = {
                    "delta": signal["delta"],
                    "multiplicity": signal["multiplicity"],
                    "fragment_atoms": [
                        {"position": atom["position"], "element": atom["element"]}
                        for atom in fragment_atoms_info
                    ],
                    "is_fully_in_fragment": len(fragment_atoms_info)
                    == signal["atom_count"],
                }
                clean_signals.append(clean_signal)

            fragment_signals[spectra_type] = clean_signals

    return {
        "success": True,
        "fragment_smiles": analysis_result["fragment_info"]["smiles"],
        "is_ion": analysis_result["fragment_info"]["is_ion"],
        "charge": analysis_result["fragment_info"]["charge"],
        "fragment_atoms": analysis_result["fragment_info"]["atoms_in_original"],
        "signals": fragment_signals,
    }


# Example usage function
def example_usage():
    """Example showing how to use the atom mapping functions."""
    smiles = "CC(c1c(CN(C)C)cccc1)Cl"
    max_retries = 5
    wait_seconds = 1
    ground_response = None
    for attempt in range(1, max_retries + 1):
        ground_response = get_raw_proton_spectrum.remote(smiles)
        # If the response is a string, treat it as a transient failure and retry
        if isinstance(ground_response, str):
            logger.info(
                f"Attempt {attempt}: get_raw_proton_spectrum returned a string: {ground_response}"
            )
            if attempt < max_retries:
                import time

                time.sleep(wait_seconds)
                return {
                    "valid": False,
                    "error": "spectra_failed",
                    "reason": ground_response,
                }
        else:
            break

    # At this point ground_response should be a dict-like object
    molfile = ground_response["data"]["molfile"]
    ground_spectra = {
        "h_nmr": {"joinedSignals": ground_response["data"]["joinedSignals"]}
    }

    # Test with methyl fragment (should match the first carbon with its hydrogens)
    fragment_smiles = "[CH-2]Cl"

    # Perform complete analysis
    analysis = analyze_fragment_mapping(fragment_smiles, molfile, ground_spectra)

    if analysis["success"]:
        fragment_info = analysis["fragment_info"]
        logger.info("✓ Fragment found")
        logger.info(f"  SMILES: {fragment_info['smiles']}")
        logger.info(f"  Is ion: {fragment_info['is_ion']}")
        if fragment_info["is_ion"]:
            logger.info(f"  Charge: {fragment_info['charge']:+d}")

        logger.info("\nAtom mapping:")
        for atom in fragment_info["atom_details"]:
            logger.info(
                f"  Fragment pos {atom['fragment_position']} -> Original pos {atom['original_position']} ({atom['element']})"
            )

        # Show relevant signals
        summary = get_fragment_signals_summary(analysis)
        logger.info("\nSignals containing fragment atoms:")

        for spectra_type, signals in summary["signals"].items():
            logger.info(f"\n{spectra_type.upper()}:")
            for signal in signals:
                atoms_str = ", ".join(
                    [
                        f"{atom['element']}{atom['position']}"
                        for atom in signal["fragment_atoms"]
                    ]
                )
                status = "fully" if signal["is_fully_in_fragment"] else "partially"
                logger.info(
                    f"  δ {signal['delta']} ({signal['multiplicity']}) - {atoms_str} [{status} in fragment]"
                )
    else:
        logger.error(f"✗ {analysis['error']}")

    return analysis


if __name__ == "__main__":
    smiles = "CC(c1c(CN(C)C)cccc1)Cl"
    fragments = ["[CH-2]Cl", "C[N-]Cc1ccccc1C(C)Cl", "[CH2-]c1ccccc1C(C)Cl"]
    # print(example_usage())
