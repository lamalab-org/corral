import modal
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

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
    # Convert possible names into SMILES
    try:
        prediction = get_isomeric_smiles.remote(prediction)
    except Exception:
        return 0.0

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

    # Check scaffold match
    pred_scaffold = MurckoScaffold.GetScaffoldForMol(pred_mol)
    true_scaffold = MurckoScaffold.GetScaffoldForMol(true_mol)
    if Chem.MolToSmiles(pred_scaffold) == Chem.MolToSmiles(true_scaffold):
        return 0.8

    # Check fingerprint similarity
    pred_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(pred_mol, 2, 1024)
    true_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(true_mol, 2, 1024)

    tanimoto = DataStructs.TanimotoSimilarity(pred_fp, true_fp)

    # Check if functional groups are present
    pred_groups = rdMolDescriptors.CalcNumRotatableBonds(pred_mol)
    true_groups = rdMolDescriptors.CalcNumRotatableBonds(true_mol)

    # Calculate similarity based on rotatable bonds
    # Using max to avoid division by zero
    rb_similarity = 1.0 - abs(pred_groups - true_groups) / (
        max(pred_groups, true_groups, 1)
    )

    final_score = 0.7 * tanimoto + 0.3 * rb_similarity
    return max(0.0, min(final_score, 1.0))
