import ast
import json

from loguru import logger
from rdkit import Chem
from retrosynthesis.retrosynthesis_utils import (
    _is_buyable,
    apply_template_retro,
    check_price,
    search_by_template,
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
                if child["type"] == "reaction" and not validate_reactions_with_products(
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
                child["smiles"] for child in node["children"] if child["type"] == "mol"
            ]

            # Verify the reaction produces the expected product
            try:
                ground_reactants = apply_template_retro(expected_product, template_id)
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
    prediction = prediction.replace("```json", "").replace("```", "").strip()
    try:
        # Step 1: Validate all reactions in the pathway
        if not validate_reactions_with_products(prediction):
            return 0.0

        # Step 2: Collect all leaf molecules (starting materials)
        leaf_molecules = collect_leaf_molecules(prediction)

        if not leaf_molecules:
            return 0.0
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


def score_final_without_price(prediction: dict, target: list) -> float:  # noqa: ARG001
    """
    Function to score the retrosynthesis route based on the provided conditions.
    Returns 1.0 if all conditions are met, else returns 0.0.

    Args:
        prediction (dict): The retrosynthesis route in JSON format.
        target (list): Target molecules for the route.

    Returns:
        float: 1.0 if all conditions are met, 0.0 if any condition is violated.
    """
    prediction = prediction.replace("```json", "").replace("```", "").strip()
    prediction = json.loads(prediction)
    try:
        # Step 1: Validate all reactions in the pathway
        if not validate_reactions_with_products(prediction):
            return 0.0

        # Step 2: Collect all leaf molecules (starting materials)
        leaf_molecules = collect_leaf_molecules(prediction)

        if not leaf_molecules:
            return 0.0

        # Step 3: Check if all starting materials are valid SMILES
        for smiles in leaf_molecules:
            if not valid_smiles(smiles):
                return 0.0

        return 1.0

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
    prediction = prediction.replace("```json", "").replace("```", "").strip()
    try:
        leaf_molecules = collect_leaf_molecules(prediction)
    except Exception as e:
        logger.warning(f"Exception during leaf molecule collection: {e}")
        return 0.0
    if not leaf_molecules:
        return 0.0

    try:
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
    except Exception as e:
        logger.warning(f"Exception during reactant checking: {e}")
        return 0.0


def check_template(prediction: str, target: str) -> float:
    try:
        prediction = json.loads(prediction)
    except Exception as e:
        logger.warning(f"Exception during JSON loading: {e}")
        return 0.0
    try:
        target = int(target)
        pred_template = prediction.get("template_id")
        if pred_template is None or pred_template != target:
            return 0.0

        ground_rxn = search_by_template(target)
        ground_rxn_mapped = ground_rxn["mapped_rxn"]
        pred_rxn = prediction.get("mapped_rxn")
        if pred_rxn is None or pred_rxn != ground_rxn_mapped:
            return 0.0

        return 1.0

    except Exception as e:
        logger.warning(f"Exception during template checking: {e}")
        return 0.0


def check_apply_template(prediction: dict, target: str) -> float:
    """
    Scoring function to check if applying the given template to the input molecule
    produces the expected output molecule.

    Args:
        prediction (dict): A dictionary containing 'input_molecule' and 'template_id'.
        target (str): The initial molecule of the retrosynthesis.

    Returns:
        float: 1.0 if the application is correct, 0.0 otherwise.
    """
    prediction_rxn = prediction.get("mapped_rxn")
    real_rxn = search_by_template(prediction.get("template_id")).get("mapped_rxn")
    if prediction_rxn != real_rxn:
        return 0.0

    try:
        precursors = apply_template_retro(target, prediction.get("template_id"))
        if precursors:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def check_list_molecules(prediction: list, target: list) -> float:
    """
    Scoring function to check if the predicted molecules match the target molecules.

    Args:
        prediction (list): A list containing the predicted molecules.
        target (list): A list of target molecule SMILES strings.

    Returns:
        float: 1.0 if all predicted molecules match the target molecules, 0.0 otherwise.
    """
    target = target[0]
    try:
        # Convert prediction to list if it's a string representation
        if isinstance(prediction, str):
            try:
                # Try ast.literal_eval first (safest for Python literals)
                prediction = ast.literal_eval(prediction)
            except (ValueError, SyntaxError):
                # If that fails, try json.loads (works for JSON-formatted strings)
                try:
                    prediction = json.loads(prediction)
                except json.JSONDecodeError:
                    # If both fail, return 0.0
                    logger.warning(
                        f"Could not convert prediction string to list: {prediction}"
                    )
                    return 0.0

        # Convert all molecules to RDKit mol objects (canonical SMILES for comparison)
        # Use canonical SMILES as the key for comparison
        for poss in prediction:
            leaf_mols = {}
            for smiles in poss:
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

            if all(target_canonical in leaf_mols for target_canonical in target_mols):
                return 1.0

        return 0.0
    except Exception as e:
        logger.warning(f"Exception during molecule checking: {e}")
        return 0.0
