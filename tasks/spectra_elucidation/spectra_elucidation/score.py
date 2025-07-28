import re
from collections import Counter

from loguru import logger
from rdkit import Chem

_HALOGENS = {"F", "Cl", "Br", "I"}

_element_pat = re.compile(r"([A-Z][a-z]?)(\d*)")
_paren_pat = re.compile(r"\(([^()]*)\)(\d*)")
_dot_pat = re.compile(r"·|\.")


def score_molecule(prediction: str, ground_truth: str) -> float:
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


def score_molecule_fragments(prediction: list[str], ground_truth: str) -> float:
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


def validate_molecular_formula(prediction, ground_truth):
    """
    Check if the predicted formula matches the formula from the ground truth SMILES.
    prediction: molecular formula string (e.g., C6H6)
    ground_truth: SMILES string
    """
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    actual_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    # Use parse_formula to compare element counts
    try:
        pred_counts = parse_formula(prediction)
        actual_counts = parse_formula(actual_formula)
    except Exception:
        return False
    return pred_counts == actual_counts


def score_formula_match(prediction, ground_truth):
    """Binary score: 1.0 if match, 0.0 if not"""
    # prediction: SMILES, ground_truth: formula
    return 1.0 if validate_molecular_formula(prediction, ground_truth) else 0.0


def _expand_parentheses(formula: str) -> str:
    """
    Recursively expand parentheses so that C6H5(CH3) becomes C6H5C1H3 etc.
    """
    while True:
        m = _paren_pat.search(formula)
        if not m:
            return formula
        inner, mult = m.groups()
        mult = int(mult or 1)
        expanded = "".join(
            f"{el}{int(cnt or 1)*mult}" for el, cnt in _element_pat.findall(inner)
        )
        formula = formula[: m.start()] + expanded + formula[m.end() :]


def _parse_simple(formula: str) -> dict[str, int]:
    """
    Parse an already-expanded, dot-free formula into {element: count}.
    """
    counts = Counter()
    for el, cnt in _element_pat.findall(formula):
        counts[el] += int(cnt or 1)
    return counts


def parse_formula(formula: str) -> dict[str, int]:
    """
    Parse molecular formula into an element-count mapping, handling
    parentheses and dot adducts.
    """
    parts = _dot_pat.split(formula.replace(" ", ""))
    total = Counter()
    for part in parts:
        expanded = _expand_parentheses(part)
        total += _parse_simple(expanded)
    return dict(total)


def calculate_dbe(molecular_formula: str) -> float:
    """
    Calculate the Double Bond Equivalent (degree of unsaturation).

    DBE = (2·C + 2 + N - H - X) / 2
    where X = total halogens (F, Cl, Br, I)
    """
    elems = parse_formula(molecular_formula)

    c = elems.get("C", 0)
    n = elems.get("N", 0)
    h = elems.get("H", 0)

    x = sum(elems.get(hal, 0) for hal in _HALOGENS)

    return (2 * c + 2 + n - h - x) / 2


def validate_dbe_consistency(prediction, ground_truth):
    """
    Validate that the predicted DBE value (integer) matches the DBE calculated from the ground truth SMILES string within a tolerance.

    Args:
        prediction (int or str): Predicted DBE value (should be an integer or string representing an integer).
        ground_truth (str): SMILES string of the ground truth molecule.
        tol (float, optional): Allowed tolerance for DBE difference. Default is 1.0.

    Returns:
        bool: True if the absolute difference between prediction and calculated DBE is within tolerance, False otherwise.
    """
    try:
        prediction = int(prediction)
    except ValueError:
        logger.error("Prediction must be an integer representing the DBE value.")
        return 0.0
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    # ground_truth can be a formula or a DBE value
    if isinstance(ground_truth, str):
        expected_dbe = calculate_dbe(ground_truth)
    else:
        expected_dbe = ground_truth
    return float(prediction == expected_dbe)


def score_isotopic_distribution(prediction, ground_truth):
    """
    Score the isotopic distribution of a molecule based on the provided prediction and ground truth.
    prediction: list of element symbols (e.g. ["C", "H", "O"])
    ground_truth: SMILES string
    """
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        logger.error("Invalid ground truth SMILES string.")
        return 0.0
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    elems = parse_formula(formula)
    if prediction is None and not elems:
        return 1.0
    if set(prediction) == set(elems.keys()):
        return 1.0
    else:
        return 0.0


def score_num_hydrogen_symmetry_classes(prediction, ground_truth):
    """Count unique hydrogen environments using canonical ranking"""
    try:
        prediction = int(prediction)
    except ValueError:
        logger.error(
            "Prediction must be an integer representing the number of hydrogen symmetry classes."
        )
        return 0.0
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    mh = Chem.AddHs(mol)
    orders = Chem.CanonicalRankAtoms(mh, breakTies=False)
    h_classes = set()
    for atom, sym_class in zip(mh.GetAtoms(), orders, strict=False):
        if atom.GetAtomicNum() == 1:  # Hydrogen
            h_classes.add(sym_class)
    return float(len(h_classes) == prediction)


def score_num_carbon_symmetry_classes(prediction, ground_truth):
    """Count unique carbon environments using canonical ranking"""
    try:
        prediction = int(prediction)
    except ValueError:
        logger.error(
            "Prediction must be an integer representing the number of carbon symmetry classes."
        )
        return 0.0
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    mc = Chem.AddHs(mol)
    orders = Chem.CanonicalRankAtoms(mc, breakTies=False)
    c_classes = set()
    for atom, sym_class in zip(mc.GetAtoms(), orders, strict=False):
        if atom.GetAtomicNum() == 6:  # Carbon
            c_classes.add(sym_class)
    return float(len(c_classes) == prediction)


def score_num_aromatic_carbons(prediction, ground_truth):
    """Count aromatic carbons in the molecule"""
    try:
        prediction = int(prediction)
    except ValueError:
        logger.error(
            "Prediction must be an integer representing the number of aromatic carbons."
        )
        return 0.0
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    aromatic_carbons = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetIsAromatic()
    )
    return float(aromatic_carbons == prediction)


def score_num_ch3_groups(prediction, ground_truth):
    """Count CH3 groups in the molecule"""
    try:
        prediction = int(prediction)
    except ValueError:
        logger.error(
            "Prediction must be an integer representing the number of CH3 groups."
        )
        return 0.0
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    ch3_groups = sum(
        1
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetTotalNumHs() == 3 and a.GetDegree() == 1
    )
    return float(ch3_groups == prediction)


def score_num_carbonyl_groups(prediction, ground_truth):
    """Count carbonyl groups (C=O) in the molecule"""
    try:
        prediction = int(prediction)
    except ValueError:
        logger.error(
            "Prediction must be an integer representing the number of carbonyl groups."
        )
        return 0.0
    mol = Chem.MolFromSmiles(ground_truth)
    if mol is None:
        raise ValueError("Invalid ground truth SMILES string.")
    carbonyl_groups = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and (
            (
                bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 8
            )
            or (
                bond.GetBeginAtom().GetAtomicNum() == 8
                and bond.GetEndAtom().GetAtomicNum() == 6
            )
        )
    )
    return float(carbonyl_groups == prediction)
