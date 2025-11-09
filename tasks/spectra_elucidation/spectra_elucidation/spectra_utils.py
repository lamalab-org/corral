import re
from collections import Counter
from itertools import combinations

import requests
from rdkit import Chem

_ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)(\d*)")
_PAREN_PAT = re.compile(r"\(([^()]*)\)(\d*)")
_DOT_PAT = re.compile(r"·|\.")


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


def format_hsqc_spectrum(zones_dict: dict) -> str:
    """
    Parse HSQC spectra data from dictionary format to standard NMR notation.

    Args:
        zones_dict (dict): Dictionary containing zones data with signals
        frequency (str): NMR frequency (default: "600 MHz")
        solvent (str): NMR solvent (default: "DMSO-d6")

    Returns:
        str: Formatted HSQC notation string
    """
    if "zones" not in zones_dict or "values" not in zones_dict["zones"]:
        return "Invalid input format"

    signals_data = []

    # Extract signals from each zone
    for zone in zones_dict["zones"]["values"]:
        if "signals" in zone:
            for signal in zone["signals"]:
                # Extract chemical shifts
                h_delta = signal["x"]["delta"]  # 1H chemical shift
                c_delta = signal["y"]["delta"]  # 13C chemical shift

                # Count hydrogen atoms from x.atoms (1H nuclei)
                h_count = len(signal["x"]["atoms"]) if "atoms" in signal["x"] else 1

                # Store the data for sorting
                signals_data.append(
                    {"h_delta": h_delta, "c_delta": c_delta, "h_count": h_count}
                )

    # Sort signals by 1H chemical shift (ascending order)
    signals_data.sort(key=lambda x: x["h_delta"])

    # Format each signal
    formatted_signals = []
    for signal in signals_data:
        h_delta = signal["h_delta"]
        c_delta = signal["c_delta"]
        h_count = signal["h_count"]

        # Format chemical shifts (remove trailing zeros)
        h_str = f"{h_delta:.2f}".rstrip("0").rstrip(".")
        c_str = f"{c_delta:.1f}".rstrip("0").rstrip(".")

        # Create the signal string
        signal_str = f"{h_str}/{c_str} ({h_count}H)"
        formatted_signals.append(signal_str)

    # Combine all signals
    signals_part = ", ".join(formatted_signals)

    # Create final HSQC string
    return f"HSQC: delta H/delta C {signals_part}."


def convert_ms_spectrum_to_string(spectrum_data):
    """
    Convert MS spectrum data from JSON format to string format.

    Args:
        spectrum_data (list): List of dictionaries with 'x' (m/z) and 'y' (intensity) keys

    Returns:
        str: Formatted string in the format "m/z 100.1 (intensity 500), 101.2 (intensity 450), ..."
    """
    if not spectrum_data:
        return ""

    # Convert each data point to the desired format
    formatted_peaks = []
    for peak in spectrum_data:
        mz = peak["x"]
        intensity = peak["y"]
        formatted_peaks.append(f"{mz} (intensity {intensity})")

    # Join all peaks with ", " and prepend "m/z "
    return "m/z " + ", ".join(formatted_peaks)


def make_api_call(url: str, payload: dict) -> dict:
    """
    Make a POST request to the specified URL with the given payload.

    Args:
        url (str): The URL to which the request is sent.
        payload (dict): The data to be sent in the request body.

    Returns:
        dict: The JSON response from the server.
    """
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()
