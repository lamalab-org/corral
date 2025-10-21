import json

from loguru import logger
from rdkit import Chem
from retrosynthesis.retrosynthesis_utils import (
    _is_buyable,
    apply_template_retro,
    check_price,
    species_match,
    valid_smiles,
)


def collect_leaf_molecules(node):
    """
    Recursively collect all leaf molecules (molecules without children).
    These are the starting materials that need to be buyable.
    """
    node = json.loads(str(node).replace("'", '"'))
    if node["type"] == "mol":
        if "children" not in node or not node["children"]:
            # This is a leaf molecule (starting material)
            return [node["smiles"]]
        else:
            # This molecule has children, so collect from children
            leaf_molecules = []
            for child in node["children"]:
                leaf_molecules.extend(collect_leaf_molecules(child))
            return leaf_molecules
    elif node["type"] == "reaction":
        # For reaction nodes, collect from all children
        leaf_molecules = []
        for child in node["children"]:
            leaf_molecules.extend(collect_leaf_molecules(child))
        return leaf_molecules
    else:
        return []


def score_final(prediction: dict, target: float) -> float:
    """
    Function to score the retrosynthesis route based on the provided conditions.
    Returns 1.0 if all conditions are met, else returns 0.0.

    Args:
        prediction (dict): The retrosynthesis route in JSON format.
        target (float): Target price for the route.

    Returns:
        float: 1.0 if all conditions are met, 0.0 if any condition is violated.
    """

    def validate_reactions_with_products(node, expected_product=None):
        """
        Validate reactions by checking if reactants + template produce expected product.
        The expected_product is the parent molecule that this reaction should produce.
        """
        if node["type"] == "mol":
            current_smiles = node["smiles"]

            if "children" not in node or not node["children"]:
                # This molecule has child reactions, validate them with current_smiles as expected product
                for child in node["children"]:
                    if child[
                        "type"
                    ] == "reaction" and not validate_reactions_with_products(
                        child, current_smiles
                    ):
                        return False
                return True
            else:
                # Leaf molecule, no reaction to validate
                return True

        elif node["type"] == "reaction":
            # This is a reaction node - validate that it produces the expected_product
            if expected_product is not None:
                template_id = node["template_id"]

                # Get reactants (children of this reaction node)
                reactant_smiles = [
                    child["smiles"]
                    for child in node["children"]
                    if child["type"] == "mol"
                ]

                # Verify the reaction produces the expected product
                try:
                    ground_reactants = apply_template_retro(
                        expected_product, template_id
                    )
                    if not species_match(reactant_smiles, ground_reactants):
                        return False
                except Exception:
                    return False

            # Recursively validate all child molecules
            for child in node["children"]:
                if not validate_reactions_with_products(child):
                    return False
            return True
        else:
            return False

    try:
        # Step 1: Validate all reactions in the pathway
        if not validate_reactions_with_products(prediction):
            return 0.0

        # Step 2: Collect all leaf molecules (starting materials)
        leaf_molecules = collect_leaf_molecules(prediction)

        # Step 3: Check if all starting materials are buyable
        for smiles in leaf_molecules:
            if not valid_smiles(smiles):
                return 0.0
            if not _is_buyable(smiles):
                return 0.0

        # Step 4: Calculate total price
        total_price = 0.0
        for smiles in leaf_molecules:
            try:
                price = check_price(smiles)
                total_price += price
            except Exception:
                # If price cannot be determined, consider it as failure
                return 0.0

        # Step 5: Check if total price is within budget
        if total_price <= target:
            return 1.0
        else:
            return 0.0

    except Exception as e:
        # Any exception during validation means failure
        logger.warning(f"Exception during scoring: {e}")
        return 0.0


def check_reactants(prediction: dict, target: list) -> float:
    """
    Scoring function to check if the retrosynthesis route is valid and meets the criteria.

    Args:
        prediction (dict): The retrosynthesis route in JSON format.
        target (list): A list containing the target molecule SMILES and the maximum allowed price.

    Returns:
        float: 1.0 if all conditions are met, 0.0 if any condition is violated.
    """
    target = target[0]
    try:
        leaf_molecules = collect_leaf_molecules(prediction)
    except Exception as e:
        logger.warning(f"Exception during leaf molecule collection: {e}")
        return 0.0
    if not leaf_molecules:
        return 0.0

    # Convert all molecules to RDKit mol objects (canonical SMILES for comparison)
    # Use canonical SMILES as the key for comparison
    leaf_mols = {}
    for smiles in leaf_molecules:
        pred_mol = Chem.MolFromSmiles(smiles)
        if pred_mol is None:
            return 0.0
        # Remove atom mapping numbers
        for atom in pred_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        # Remove stereochemistry for comparison
        Chem.RemoveStereochemistry(pred_mol)
        canonical_smiles = Chem.MolToSmiles(pred_mol)
        leaf_mols[canonical_smiles] = pred_mol

    target_mols = {}
    for target_smiles in target:
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            return 0.0
        # Remove atom mapping numbers
        for atom in target_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        # Remove stereochemistry for comparison
        Chem.RemoveStereochemistry(target_mol)
        canonical_smiles = Chem.MolToSmiles(target_mol)
        target_mols[canonical_smiles] = target_mol

    return (
        1.0
        if all(target_canonical in leaf_mols for target_canonical in target_mols)
        else 0.0
    )
