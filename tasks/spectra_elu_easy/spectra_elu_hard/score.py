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


def score_molecule_fragments_subtasks(
    prediction: list[str], ground_truth: str
) -> float:
    """Score the prediction based on whether all fragments are substructures of the ground truth molecule.

    Args:
        prediction (list[str]): List of SMILES strings representing fragments.
        ground_truth (str): SMILES string of the ground truth molecule.

    Returns:
        float: 1.0 if all fragments are substructures of the ground truth molecule,
               0.0 otherwise.
    """
    target_mol = Chem.MolFromSmiles(ground_truth)
    if target_mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    for frag in prediction:
        frag_mol = Chem.MolFromSmiles(frag)
        if frag_mol is None:
            return 0.0
        # Check if fragment is a substructure of the target molecule
        if not target_mol.HasSubstructMatch(frag_mol):
            return 0.0
    return 1.0
