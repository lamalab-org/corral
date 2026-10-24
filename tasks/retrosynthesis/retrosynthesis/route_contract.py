"""Task-level route requirements, independent of reaction-catalog validation."""

from rdkit import Chem


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid product SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Match the existing route grader's stereochemistry-agnostic comparison.
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def validate_contract(contract: dict) -> None:
    """Reject missing or inconsistent private task configuration at setup."""
    _canonical(contract["product_smiles"])
    for key in ("min_reactions", "max_reactions"):
        value = contract.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    if contract.get("min_reactions", 0) > contract.get("max_reactions", float("inf")):
        raise ValueError("Inconsistent route reaction limits")


def matches_contract(route: dict, reactions: int, contract: dict) -> bool:
    """Check the requested product and reaction count without revealing a route."""
    return (
        route.get("type") == "mol"
        and _canonical(route["smiles"]) == _canonical(contract["product_smiles"])
        and reactions >= contract.get("min_reactions", 0)
        and reactions <= contract.get("max_reactions", float("inf"))
    )
