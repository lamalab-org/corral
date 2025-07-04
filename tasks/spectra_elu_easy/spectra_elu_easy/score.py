import modal
from rdkit import Chem

get_isomeric_smiles = modal.Function.from_name("chemenv", "get_isomeric_smiles_pubchem")


def score_molecule_similarity(prediction: str, ground_truth: str) -> float:
    """
    Compare predicted and ground truth SMILES strings and calculate similarity score.

    Args:
        prediction (str): SMILES string of predicted molecule
        ground_truth (str): SMILES string of ground truth molecule

    Returns:
        float: Score between 0.0 and 1.0 based on similarity
    """
    if prediction == ground_truth:
        return 1.0
    try:
        pred_mol = Chem.MolFromSmiles(prediction)
    except Exception:
        pred_mol = None
    try:
        true_mol = Chem.MolFromSmiles(ground_truth)
    except Exception:
        true_mol = None

    if true_mol is None:
        raise ValueError("Invalid ground truth SMILES string.")

    if pred_mol is None:
        return 0.0

    if Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(true_mol):
        return 1.0

    return 0.0
