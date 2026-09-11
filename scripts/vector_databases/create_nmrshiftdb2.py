from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rdkit import Chem

from corral.utils.rag import create_vector_database

load_dotenv("../.env", override=True)


def parse_nmr_properties(properties):
    """
    Parse NMR properties from a dictionary into a clean format.

    Args:
        properties (dict): Dictionary of properties from an RDKit molecule. The keys should include:
            - "INChI key" or "INChI" for SMILES
            - "Solvent" for solvent information
            - "Field Strength [MHz]" for field strength
            - "Temperature [K]" for temperature
            - "Spectrum 13C" and "Spectrum 1H" for NMR spectra

    Returns:
        dict: Dictionary with 'smiles', '13C_NMR', and '1H_NMR' keys or None if no spectra
    """
    # Extract SMILES
    smiles = None
    if "INChI" in properties:
        # If we need to convert InChI to SMILES, we would do it here
        # For now, we'll use InChI key as a placeholder
        smiles = properties.get("INChI key", "No SMILES available")

    # Parse solvent information
    solvents = {}
    solvent_names = {
        "Deuteriumoxide": "D2O",
        "Chloroform-D1": "CDCl3",
        "Acetone-D6": "Acetone-d6",
        "(CD3)2SO": "DMSO-d6",
        "Benzene-D6": "Benzene-d6",
        "Acetonitrile-D3": "CD3CN",
        "Methanol-D4": "CD3OD",
    }

    if "Solvent" in properties:
        solvent_data = properties["Solvent"].split()
        for solvent_entry in solvent_data:
            if ":" in solvent_entry:
                idx, solvent = solvent_entry.split(":")
                if solvent not in ["Unreported"]:
                    # Clean up solvent name
                    for old, new in solvent_names.items():
                        if old in solvent:
                            solvent = new
                            break
                    solvents[idx] = solvent

    # Extract field strength information
    field_strengths = {}
    if "Field Strength [MHz]" in properties:
        field_data = properties["Field Strength [MHz]"].split()
        for field_entry in field_data:
            if ":" in field_entry:
                idx, field = field_entry.split(":")
                if field not in ["Unreported"]:
                    field_strengths[idx] = field

    # Extract temperature information
    temperatures = {}
    if "Temperature [K]" in properties:
        temp_data = properties["Temperature [K]"].split()
        for temp_entry in temp_data:
            if ":" in temp_entry:
                idx, temp = temp_entry.split(":")
                if temp not in ["Unreported"]:
                    temperatures[idx] = temp

    # Process 13C NMR spectra
    c13_spectra = []
    for key, value in properties.items():
        if key.startswith("Spectrum 13C"):
            spectrum_idx = key.split()[-1]
            shifts = []
            for signal in value.split("|"):
                if signal:
                    shift_data = signal.split(";")
                    if len(shift_data) >= 1:
                        shifts.append(shift_data[0])

            if shifts:
                solvent_str = solvents.get(spectrum_idx, "CDCl3")
                field_str = field_strengths.get(spectrum_idx, "150")
                temp_str = temperatures.get(spectrum_idx, "")

                temp_display = f", {temp_str} K" if temp_str else ""

                # Format as conventional 13C NMR representation
                c13_nmr = f"13C NMR ({field_str} MHz, {solvent_str}{temp_display}) δ: {', '.join(shifts)}"
                c13_spectra.append(c13_nmr)

    # Process 1H NMR spectra
    h1_spectra = []
    for key, value in properties.items():
        if key.startswith("Spectrum 1H"):
            spectrum_idx = key.split()[-1]
            signals = []

            for signal in value.split("|"):
                if signal:
                    # Format: shift;coupling;multiplicity
                    parts = signal.split(";")
                    if len(parts) >= 3:
                        shift = parts[0]
                        n_hydrogens = int(parts[2])
                        # Convert multiplicity number to letter
                        mult_labels = {1: "s", 2: "d", 3: "t", 4: "q", 5: "quint"}
                        mult_label = mult_labels.get(n_hydrogens, "m")
                        signals.append(f"{shift} ({mult_label}, {n_hydrogens}H)")

            if signals:
                solvent_str = solvents.get(spectrum_idx, "CDCl3")
                field_str = field_strengths.get(spectrum_idx, "300")
                temp_str = temperatures.get(spectrum_idx, "")

                temp_display = f", {temp_str} K" if temp_str else ""

                # Format as conventional 1H NMR representation
                h1_nmr = f"1H NMR ({field_str} MHz, {solvent_str}{temp_display}) δ: {', '.join(signals)}"
                h1_spectra.append(h1_nmr)

    # Check if there are any spectra reported
    if not c13_spectra and not h1_spectra:
        return None

    # Create result dictionary
    result = {"smiles": smiles}

    if c13_spectra:
        result["13C_NMR"] = c13_spectra

    if h1_spectra:
        result["1H_NMR"] = h1_spectra

    return result


def convert_sd_to_vector_db(
    sd_file_path, collection_name="smiles_collection", db_path=None
):
    """
    Convert an SD file to a vector database, indexed by SMILES.

    Args:
        sd_file_path (str): Path to the SD file
        collection_name (str): Name for the vector database collection
        db_path (str): Path where to store the vector database

    Returns:
        str: Result message from create_vector_database
    """
    logger.info(f"Reading SD file: {sd_file_path}")

    # Create path for database if not specified
    if db_path is None:
        db_path = Path("vector_databases") / collection_name

    # Read molecules from SD file
    supplier = Chem.SDMolSupplier(sd_file_path)

    # Extract SMILES and other data from molecules
    entries = []
    smiles_list = []

    for i, mol in enumerate(supplier):
        if mol is not None:
            # Generate SMILES string
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)

            properties = {
                prop_name: mol.GetProp(prop_name) for prop_name in mol.GetPropNames()
            }
            # Parse NMR data
            nmr_data = parse_nmr_properties(properties)

            # Skip if no spectra reported
            if nmr_data is None:
                continue

            # Add SMILES from RDKit to the data
            nmr_data["smiles"] = smiles

            # Create formatted entry
            entry = f"SMILES: {smiles}"
            if "13C_NMR" in nmr_data:
                entry += f"\n13C_NMR: {nmr_data['13C_NMR'][0]}"
            if "1H_NMR" in nmr_data:
                entry += f"\n1H_NMR: {nmr_data['1H_NMR'][0]}"

            entries.append(entry)

            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i} molecules")

    logger.info(f"Successfully read {len(entries)} molecules from SD file")

    documents = []
    smiles = []

    for entry_data in entries:
        smiles_ = entry_data.split("\n")[0].replace("SMILES: ", "")
        if not isinstance(smiles_, str):
            continue

        smiles.append(smiles_)
        documents.append(entry_data)

    logger.info(f"Total unique SMILES found: {len(smiles)}")
    logger.info(f"Total entries processed: {len(documents)}")

    result = create_vector_database(
        chunks=documents,
        collection_name=collection_name,
        path=str(db_path),
        chunk_size=1024,
        update_mode="recreate",
        chemical=smiles,
    )

    logger.info(f"Created vector database at {db_path}")
    return result


if __name__ == "__main__":
    sd_file = "../nmrshiftdb2withsignals.sd"
    collection_name = "nmrshiftdb2"
    db_path = "../vector_databases/nmrshiftdb2"

    result = convert_sd_to_vector_db(sd_file, collection_name, db_path)
    logger.info(f"Result: {result}")
