import json
import random
import re
import uuid
from collections import Counter
from itertools import combinations
from pathlib import Path

import modal
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

_ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)(\d*)")
_PAREN_PAT = re.compile(r"\(([^()]*)\)(\d*)")
_DOT_PAT = re.compile(r"·|\.")

get_isomers = modal.Function.lookup(
    "chemenv", "get_compound_isomers_pubchem_by_formula"
)
get_ir_nmr = modal.Function.lookup("chemenv", "get_ir_prediction")
get_h_nmr = modal.Function.lookup("chemenv", "get_h_nmr_prediction")
get_c_nmr = modal.Function.lookup("chemenv", "get_c13_nmr_prediction")


def differs_by_one_atom(parent_formula, fragment_formula):
    """Check if fragment differs from parent by exactly one atom"""
    total_diff = 0
    all_elements = set(parent_formula.keys()) | set(fragment_formula.keys())

    for element in all_elements:
        parent_count = parent_formula.get(element, 0)
        fragment_count = fragment_formula.get(element, 0)
        diff = abs(parent_count - fragment_count)
        total_diff += diff

    # If total difference is exactly 1, it means one atom was added/removed
    return total_diff == 1 or total_diff == 0


def _parse_simple(formula: str) -> dict[str, int]:
    """
    Parse an already-expanded, dot-free formula into {element: count}.
    """
    counts = Counter()
    for el, cnt in _ELEMENT_PAT.findall(formula):
        counts[el] += int(cnt or 1)
    return counts


def _expand_parentheses(formula: str) -> str:
    """
    Recursively expand parentheses so that C6H5(CH3) becomes C6H5C1H3 etc.
    """
    while True:
        m = _PAREN_PAT.search(formula)
        if not m:
            return formula
        inner, mult = m.groups()
        mult = int(mult or 1)
        expanded = "".join(
            f"{el}{int(cnt or 1)*mult}" for el, cnt in _ELEMENT_PAT.findall(inner)
        )
        formula = formula[: m.start()] + expanded + formula[m.end() :]


def parse_molecular_formula(formula: str) -> dict[str, int]:
    """
    Parse molecular formula into an element-count mapping, handling
    parentheses and dot adducts.
    """
    parts = _DOT_PAT.split(formula.replace(" ", ""))
    total = Counter()
    for part in parts:
        expanded = _expand_parentheses(part)
        total += _parse_simple(expanded)
    return dict(total)


def enumerate_fragments_from_smiles(
    smi: str,
    max_cuts: int = 2,
    skip_ring_bonds: bool = True,
    min_heavy_atoms: int = 2,
    max_combos: int = 100000,
):
    """
    Generate fragment SMILES by breaking up to `max_cuts` bonds.
    Returns a sorted list of unique canonical SMILES of fragments.
    Preserves ions by setting formal charges instead of adding hydrogens.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Candidate bonds: heavy-atom bonds only; optionally skip ring bonds
    cand_bond_idxs = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if skip_ring_bonds and b.IsInRing():
            continue
        cand_bond_idxs.append(b.GetIdx())

    # Early exit: nothing to cut
    if not cand_bond_idxs or max_cuts < 1:
        return []

    uniq = set()
    tried = 0

    for ncuts in range(1, min(max_cuts, len(cand_bond_idxs)) + 1):
        for cutset in combinations(cand_bond_idxs, ncuts):
            tried += 1
            if tried > max_combos:
                # Safety valve against combinatorial explosion
                break

            rw = Chem.RWMol(mol)
            cut_atom_info = {}

            # Count how many bonds each atom will lose
            for bidx in cutset:
                b = rw.GetBondWithIdx(bidx)
                begin_idx = b.GetBeginAtomIdx()
                end_idx = b.GetEndAtomIdx()

                cut_atom_info[begin_idx] = cut_atom_info.get(begin_idx, 0) + 1
                cut_atom_info[end_idx] = cut_atom_info.get(end_idx, 0) + 1

            # Before removing bonds, adjust formal charges on cut atoms
            for atom_idx, bonds_lost in cut_atom_info.items():
                atom = rw.GetAtomWithIdx(atom_idx)
                current_charge = atom.GetFormalCharge()

                # Adjust charge based on bonds lost
                new_charge = current_charge - bonds_lost
                atom.SetFormalCharge(new_charge)

            # Now remove the selected bonds
            for bidx in sorted(
                cutset, reverse=True
            ):  # Remove in reverse order to maintain indices
                b = rw.GetBondWithIdx(bidx)
                rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            # Get connected components as fragments
            try:
                frags = Chem.rdmolops.GetMolFrags(
                    rw.GetMol(), asMols=True, sanitizeFrags=False
                )

                for frag in frags:
                    # Filter tiny pieces
                    if (
                        sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() > 1)
                        < min_heavy_atoms
                    ):
                        continue

                    try:
                        # Try to sanitize the fragment with ionic charges
                        Chem.SanitizeMol(frag, catchErrors=False)
                        smi_frag = Chem.MolToSmiles(
                            frag, isomericSmiles=True, canonical=True
                        )
                        uniq.add(smi_frag)
                    except Exception:
                        # If sanitization fails with ionic charges, try without sanitization
                        try:
                            smi_frag = Chem.MolToSmiles(
                                frag, isomericSmiles=True, canonical=True
                            )
                            uniq.add(smi_frag)
                        except Exception:
                            continue  # Skip this fragment if it can't be processed

            except Exception:
                # If fragmentation fails completely, skip this cut combination
                continue

        else:
            continue
        break  # broke due to max_combos

    return sorted(uniq)


def return_possible_fragments(h_smiles: str) -> list[str]:
    """[BRIEF] Return a list of fragments of the sample at hand by removing one or two atoms from the molecule. [/BRIEF]

    [DETAILED] This function generates some fragments from the sample at hand. It returns a list of unique fragment SMILES strings. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to generate fragments for the sample at hand.
    - When you want to explore different possible options based on the spectra.
    - When you are at an endpoint and need to consider some potential fragments. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure that the correct step is to generate fragments. Ensure that you really tried to guess all the posibilities from the information available. [/PREREQUISITE]
    2. [CURRENT] Call this tool to generate fragments for the sample at hand. [/CURRENT]
    3. [FOLLOW_UP] Evaluate the fragments for getting the final molecule. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - This tool takes the spectra information and generates fragments based on known molecules.
    - It uses a combination of cheminformatics techniques to identify potential fragment structures.
    - It leverages existing databases and algorithms to find some candidates.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `return_possible_fragments()`
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        list[str]:
            [BRIEF] A list of unique fragment SMILES strings. [/BRIEF]
            [DETAILED] This list contains all the unique SMILES representations of the fragments found by database lookup. [/DETAILED]
            [EXAMPLES] Example SMILES strings: ["C1=CC=CC=C1", "C1=CC=CC=C1O", ...] [/EXAMPLES]

    [RAISES] Exceptions:
        None
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - This tool might not return all the fragments, but there the ones returned are 100% accurate.
        - The quality of the generated fragments depends on the underlying database and its coverage.
    [/LIMITATIONS]
    """
    mol = Chem.MolFromSmiles(h_smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    parent_formula = CalcMolFormula(mol)

    fragments = enumerate_fragments_from_smiles(h_smiles)

    final_fragments = []
    for fragment in fragments:
        try:
            fragment_mol = Chem.MolFromSmiles(fragment)
        except Exception:
            continue
        if fragment_mol is None:
            continue
        fragment_formula = CalcMolFormula(fragment_mol)
        if not differs_by_one_atom(
            parse_molecular_formula(parent_formula),
            parse_molecular_formula(fragment_formula),
        ):
            final_fragments.append(fragment)

    random.shuffle(final_fragments)
    return final_fragments


if __name__ == "__main__":
    smiles = [
        "O=C(N(C)C)C1=CC=CC=C1C(Cl)C",
        "C=C[C@H](O)[C@H]1CO1",
        "Cc1ccccc1C(NC)CC(C)C(C)=S",
        "CCCCCCC#CC#C",
        "O=C(OC)C1=CC=C(C=C1)C(CN)C",
        "CC(c1ccc(CCc2ccccc2)cc1)=O",
        "O=C(NC)C1=CC=CC=C1CCl",
        "CCOC(=O)C1=CC=C(C=C1)NC2=CN=C(C=C2)Cl",
        "O=C(OC)C(C=C1)=CC=C1C2=CC3=CC=CC=C3C=C2",
        "ClC1=CC=C(C=C1)C(CC)Br",
        "C/C(C(NC)=O)=C(C)\\C",
        "BrC1=CC=C(C=C1)CCN",
        "C/C(C(O)=O)=C(C)\\C",
        "C/C(C(N(C)C)=O)=C(C)\\C",
        "C1=CC=C(C=C1)C(=O)N",
        "O=C(OC)C1=CC=C(C=C1)CCN",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "O=C(c1ccccc1)c2ccccc2",
        "ClC1=CC=C(C2CCCCC2)C=C1",
        "Cc1ccccc1C(NC)C(c2c(C)cccc2)NC",
    ]
    out_dir = Path("tasks_json")
    out_dir.mkdir(exist_ok=True)
    task_counter = 0
    for smile in smiles:
        task_dir = out_dir / f"task_{task_counter:05d}"
        task_dir.mkdir(exist_ok=True)
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        number_atoms = mol.GetNumAtoms()
        fragments = return_possible_fragments(smile)
        fragment_counter = 0
        for fragment in fragments:
            task = {}
            molecular_formula = CalcMolFormula(Chem.MolFromSmiles(fragment))
            isomers = get_isomers.remote(formula=molecular_formula, limit=3)
            if len(isomers) < 3:
                continue
            sampled_isomers = isomers
            target_scores = {isomer: 0 for isomer in sampled_isomers}
            target_scores[fragment] = 1
            items = list(target_scores.items())
            random.shuffle(items)
            target_scores = dict(items)
            h_nmr = get_h_nmr.remote(smiles=smile)
            c_nmr = get_c_nmr.remote(smiles=smile)
            examples = [
                {
                    "input": f"What of the four fragments is more likely to match the next C-NMR and H-NMR spectra?\n\nH-NMR: {h_nmr}\n\nC-NMR: {c_nmr}",
                    "target": None,
                    "target_scores": str(target_scores),
                }
            ]
            task["name"] = f"Fragment Matching Task {smile}"
            task["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, task["name"]))
            task["description"] = (
                f"Determine which fragment is more likely to match the next C-NMR and H-NMR spectra of the molecule {smile}."
            )
            task["examples"] = examples
            task["keywords"] = [
                "chemistry",
                "spectra",
                "nmr",
                "c-nmr",
                "h-nmr",
                "fragments",
            ]
            task["metrics"] = ["multiple_choice_grade"]
            task["preferred_score"] = "multiple_choice_grade"
            task_filename = task_dir / f"task_{task_counter:03d}.json"
            with task_filename.open("w") as f:
                json.dump([task], f, indent=2)
            task_counter += 1
            fragment_counter += 1
            if fragment_counter >= 5:
                break
