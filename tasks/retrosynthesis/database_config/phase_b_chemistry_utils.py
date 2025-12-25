"""
Chemistry utility functions for Phase B production database.

This module provides functions to extract:
- Bond changes (formed, broken, order changed)
- Functional groups (formed, broken)

Replace the placeholder implementations with your actual code.
"""

from collections import Counter

from loguru import logger
from rdkit import Chem
from rxnutils.chem.reaction import ChemicalReaction


def _mol_info_from_smiles_list(smiles_list: list[str]) -> tuple[dict, dict]:
    """
    Extract bond and atom map information from list of SMILES.

    Args:
        smiles_list (list[str]): List of SMILES strings.

    Returns:
        tuple (dict, dict):
            bonds: dict of {(map1, map2): {'order': ..., 'aromatic': bool}}
            amapZ: dict of {mapNum: atomic_number}
    """
    bonds = {}
    amapZ = {}
    next_map = 1

    # First pass: find max existing map number
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            m = a.GetAtomMapNum()
            if m:
                next_map = max(next_map, m + 1)

    # Second pass: process all molecules
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Assign map numbers to unmapped atoms
        for a in mol.GetAtoms():
            m = a.GetAtomMapNum()
            if m:
                amapZ[m] = a.GetAtomicNum()
            else:
                a.SetAtomMapNum(next_map)
                amapZ[next_map] = a.GetAtomicNum()
                next_map += 1

        # Extract bonds
        for b in mol.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            if not (m1 and m2):
                continue
            key = tuple(sorted((m1, m2)))
            bt = b.GetBondType()
            arom = bt == Chem.BondType.AROMATIC or (
                a1.GetIsAromatic() and a2.GetIsAromatic()
            )
            order = "AROMATIC" if arom else b.GetBondTypeAsDouble()
            bonds[key] = {"order": order, "aromatic": arom}

    return bonds, amapZ


def _pairZ_label(map_pair, rZ, pZ):
    """Create atomic number pair label (e.g., '6-8' for C-O bond)"""
    m1, m2 = map_pair
    z1 = rZ.get(m1, pZ.get(m1))
    z2 = rZ.get(m2, pZ.get(m2))
    if z1 is None or z2 is None:
        return None
    a, b = sorted((z1, z2))
    return f"{a}-{b}"


def obtain_bonds(mapped_rxn: str) -> dict[str, list[str]]:
    """
    Extract bond changes from mapped reaction SMILES.

    Filters out internal bonds within leaving groups (only reports bonds
    where at least one atom persists in products).

    Args:
        mapped_rxn (str): Atom-mapped reaction SMILES (reactants>>products)

    Returns:
        dict[str, list[str]]: Dict with keys:
        - 'formed': List of formed bonds (e.g., ['6-7', '6-8'])
        - 'broken': List of broken bonds
        - 'order_changed': List of bonds with order changes (e.g., ['6-6 (1.0->2.0)'])
    """
    try:
        rxn = ChemicalReaction(mapped_rxn)
        reactants_smiles = rxn.reactants_list
        products_smiles = rxn.products_list

        # Extract bond/atom info
        rbonds, rZ = _mol_info_from_smiles_list(reactants_smiles)
        pbonds, pZ = _mol_info_from_smiles_list(products_smiles)

        # Identify persistent atoms
        persistent_atoms = set(rZ.keys()) & set(pZ.keys())

        rkeys, pkeys = set(rbonds), set(pbonds)

        # Broken bonds (only if at least one atom persists)
        broken_keys = []
        for k in rkeys - pkeys:
            m1, m2 = k
            if m1 in persistent_atoms or m2 in persistent_atoms:
                broken_keys.append(k)

        # Formed bonds
        formed_keys = list(pkeys - rkeys)

        # Order changed bonds
        order_changed_keys = [
            k for k in (rkeys & pkeys) if rbonds[k]["order"] != pbonds[k]["order"]
        ]

        # Build output lists
        formed = []
        broken = []
        order_changed = []

        for k in formed_keys:
            label = _pairZ_label(k, rZ, pZ)
            if label:
                formed.append(label)

        for k in broken_keys:
            label = _pairZ_label(k, rZ, pZ)
            if label:
                broken.append(label)

        for k in order_changed_keys:
            label = _pairZ_label(k, rZ, pZ)
            if label:
                old = rbonds[k]["order"]
                new = pbonds[k]["order"]
                order_changed.append(f"{label} ({old}->{new})")

        return {
            "formed": sorted(formed),
            "broken": sorted(broken),
            "order_changed": sorted(order_changed),
        }

    except Exception as e:
        logger.warning(f"Bond detection failed for {mapped_rxn[:50]}...: {e}")
        return {"formed": [], "broken": [], "order_changed": []}


# SMARTS patterns for functional groups
FG_SMARTS = {
    "azo": "[#6]-N=N-[#6]",
    "diazo": "[#6,#1][CX3]([#6,#1])=[$([NX2+]=[NX1-]),$([NX2-]#[NX1+])]",
    "aryl diazonium": "[c]-[NX2+]#[NX1]",
    "azide": "[#6][$([NX2]=[NX2+]=[NX1-]),$([NX2-]-[NX2+]#[NX1])]",
    "nitrile": "[CX2]#[NX1]",
    "isonitrile": "[NX2+]#[CX1-]",
    "nitro": "[#6][NX3+](=O)[O-]",
    "aldehydes": "[#6,#1][CX3](=O)[H]",
    "aliphatic hydroxyls": "[C;!$(C=[O,S,N])][OX2H1]",
    "aromatic hydroxyls": "[c][OX2H1]",
    "aliphatic carboxylic acid": "[C][CX3](=O)[O;H1,-]",
    "aromatic carboxylic acid": "[c][CX3](=O)[O;H1,-]",
    "acyl chloride": "[#6][CX3](=O)Cl",
    "six-membered aromatic rings": "[n,c]1[n,c][n,c][n,c][n,c][n,c]1",  # includes fused systems, e.g. naphthalene, quinoline
    "benzene rings": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # single benzene ring; excludes fused systems
    "phenyl group": "[cH0]1[cH][cH][cH][cH][cH]1",  # a.k.a mono-substituted benzene
    "ortho di-substituted benzene": "[cH0]1[cH0]([!$([c,n,s,o])])[cH][cH][cH][cH]1",  # excludes fused systems
    "meta di-substituted benzene": "[cH0]1[cH1][cH0][cH][cH][cH]1",
    "para di-substituted benzene": "[cH0]1[cH1][cH][cH0][cH][cH]1",
    "pyrridine": "[nR1]1[cR1][cR1][cR1][cR1][cR1]1",  # single pyrridine ring; excludes fused systems
    "pyridazine": "[nR1]1[nR1][cR1][cR1][cR1][cR1]1",  # single pyrridine ring; excludes fused systems
    "pyrimidine": "[nR1]1[cR1][nR1][cR1][cR1][cR1]1",  # single pyrridine ring; excludes fused systems
    "pyrazine": "[nR1]1[cR1][cR1][nR1][cR1][cR1]1",  # single pyrridine ring; excludes fused systems
    "amide": "[#6,#1][CX3](=O)[NX3]([#1,#6;!$(C=[O,S,N])])[#1,#6;!$(C=[O,S,N])]",
    "imide": "[#6,#1][CX3](=O)[NX3]([CX3](=O)[#6,#1])[#1,#6;!$(C=[O,S,N])]",
    "thioamide": "[#6,#1][CX3](=S)[NX3]([#1,#6;!$(C=[O,S,N])])[#1,#6;!$(C=[O,S,N])]",
    "amidine": "[#6,#1][CX3](=[NX2])[NX3]([#1,#6;!$(C=[O,S,N])])[#1,#6;!$(C=[O,S,N])]",
    "guanidine": "[#1,#6;!$(C=[O,S,N])][NX3]([#1,#6;!$(C=[O,S,N])])[CX3](=[NX2])[NX3]([#1,#6;!$(C=[O,S,N])])[#1,#6;!$(C=[O,S,N])]",
    "C-C double bonds": "[C]=[C]",
    "C-C triple bonds": "[C]#[C]",
    "imine": "[#6,#1][CX3]([#6,#1])=[NX2][#6,#1]",
    "cyclopropane": "C1CC1",
    "ketones": "[#6][C](=O)[#6]",
    "carboxylic esters": "[#6,#1][CX3](=O)[OX2H0][#6;!$(C=[O,S,N])]",  # excludes carbonats, carbamates, and anhydrides
    "carbonate": "[#6;!$(C=[O,S,N])][OX2H0][CX3](=O)[OX2H0][#6;!$(C=[O,S,N])]",
    "carbamate": "[#6;!$(C=[O,S,N])][OX2H0][CX3](=O)[NX3]",
    "anhydride": "[#6,#1;!$(C=[O,S,N])][CX3](=O)[OX2H0][CX3](=O)[#6,#1;!$(C=[O,S,N])]",
    "urea": "[NX3][CX3](=O)[NX3]",
    "thiourea": "[NX3][CX3](=S)[NX3]",
    "ether oxygens": "[#6;!$(C=[O,S,N])][OX2H0;!$([O;r3])][#6;!$(C=[O,S,N])]",  # excludes epoxides
    "epoxides": "O1CC1",
    "aziridines": "N1CC1",
    "four-membered rings": "[*]1~[*]~[*]~[*]1",
    "five-membered rings": "[*]1~[*]~[*]~[*]~[*]1",
    "trifluoromethyl": "[CX4H0](F)(F)F",
    "primary amines": "[#6;!$(C=[O,S,N])][NX3H2]",  # excludes amides
    "secondary amines": "[NX3H1;!$(NC=[O,S,N]);!$([N;r3])]([#6])[#6]",  # excludes amides and aziridines
    "tertiary amines": "[NX3;!$(NC=[O,S,N]);!$([N;r3])]([#6])([#6])[#6]",  # excludes amides and aziridines
    "t-butyl": "[CX4H0]([CH3])([CH3])[CH3]",
    "thiol": "[#6][SX2H1][H]",
    "thioethers": "[#6][SX2H0][#6]",  # excludes sulfoxides and sulfones
    "thioketones": "[#6][CX3](=S)[#6]",
    "sulfoxides": "[#6][$([SX3](=O)),$([SX3+]([O-]))][#6]",
    "sulfones": "[#6][$([SX4](=O)(=O)),$([S+2X4]([O-])([O-]))][#6]",
    "sulfonic": "[#6][$([SX4](=O)(=O)),$([S+2X4]([O-])([O-]))][OX2;H,-]",
    "sulfonamides": "[#6][$([SX4](=O)(=O)),$([S+2X4]([O-])([O-]))]N",
    "methoxy": "[#6;!$(C=[O,S,N])][OX2H0][CH3]",
    "ethoxy": "[#6;!$(C=[O,S,N])][OX2H0][CH2][CH3]",
    "isopropoxy": "[#6;!$(C=[O,S,N])][OX2H0][CH]([CH3])[CH3]",
    "phenoxy": "[#6;!$(C=[O,S,N])][OX2H0]c1[cH1][cH1][cH1][cH1][cH1]1",
    "alkyl halide": "[CX4][F,Cl,Br,I]",
    "aryl halide": "[c][F,Cl,Br,I]",
    "halide": "[#6][F,Cl,Br,I]",  # any carbon bonded to halogen (F, Cl, Br, or I)
    "furan": "[oR1]1[cR1][cR1][cR1][cR1]1",  # single ring; excludes fused systems
    "thiophene": "[sR1]1[cR1][cR1][cR1][cR1]1",  # single ring; excludes fused systems
    "pyrrole": "[nR1]1[cR1][cR1][cR1][cR1]1",  # single ring; excludes fused systems
    "imidazole": "[nR1]1[cR1][nR1][cR1][cR1]1",  # single ring; excludes fused systems
    "oxazole": "[cR1]1[oR1][cR1][nR1][cR1]1",  # single ring; excludes fused systems
    "thiazole": "[cR1]1[sR1][cR1][nR1][cR1]1",  # single ring; excludes fused systems
    "cyanate": "[#6]OC#N",
    "isocyanate": "[#6]N=C=O",
    "isothiocyanate": "[#6]N=C=S",
    "trimethylsilyl": "[Si]([CH3])([CH3])[CH3]",
    "t-butyldimethylsilyl": "[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]",
}

# Precompile patterns
FG_PATTERNS = {}
for name, smarts in FG_SMARTS.items():
    patt = Chem.MolFromSmarts(smarts)
    if patt is not None:
        FG_PATTERNS[name] = patt


def detect_functional_groups_in_molecule(smiles: str) -> list[str]:
    """
    Detect functional groups in a single molecule.

    Args:
        smiles (str): SMILES string (can be atom-mapped)

    Returns:
        list[str]: List of functional group names detected (with duplicates
                   if multiple instances of the same FG are present)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # Remove atom map numbers to ensure SMARTS patterns match correctly
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    detected = []
    for name, patt in FG_PATTERNS.items():
        matches = mol.GetSubstructMatches(patt)
        # Add the functional group name once for each match
        detected.extend([name] * len(matches))

    return sorted(detected)


def get_functional_groups(mapped_rxn: str) -> dict[str, list[str]]:
    """
    Detect functional groups formed and broken in reaction.

    Tracks counts of each functional group type, so if a molecule has
    two instances of the same FG and one is removed, that change is detected.

    Args:
        mapped_rxn (str): Atom-mapped reaction SMILES

    Returns:
        dict[str, list[str]]: Dict with keys:
        - 'formed': List of FG names formed in products (with duplicates for multiple instances)
        - 'broken': List of FG names broken from reactants (with duplicates for multiple instances)
    """
    try:
        # Split reaction
        parts = mapped_rxn.split(">>")
        if len(parts) != 2:
            logger.warning(f"Invalid reaction format: {mapped_rxn[:50]}...")
            return {"formed": [], "broken": []}

        reactants_str = parts[0]
        products_str = parts[1]

        # Detect FGs in all reactants (as list to preserve counts)
        reactant_fgs = []
        for r_smiles in reactants_str.split("."):
            # Strip only outer parentheses if present (some reaction formats wrap molecules)
            r_smiles = r_smiles.strip()
            if r_smiles.startswith("(") and r_smiles.endswith(")"):
                r_smiles = r_smiles[1:-1]
            reactant_fgs.extend(detect_functional_groups_in_molecule(r_smiles))

        # Detect FGs in all products (as list to preserve counts)
        product_fgs = []
        for p_smiles in products_str.split("."):
            # Strip only outer parentheses if present
            p_smiles = p_smiles.strip()
            if p_smiles.startswith("(") and p_smiles.endswith(")"):
                p_smiles = p_smiles[1:-1]
            product_fgs.extend(detect_functional_groups_in_molecule(p_smiles))

        # Use Counter to compute differences with counts
        reactant_counts = Counter(reactant_fgs)
        product_counts = Counter(product_fgs)

        # Formed: FGs that appear more in products than reactants
        formed = []
        for fg, count in product_counts.items():
            diff = count - reactant_counts.get(fg, 0)
            if diff > 0:
                formed.extend([fg] * diff)

        # Broken: FGs that appear more in reactants than products
        broken = []
        for fg, count in reactant_counts.items():
            diff = count - product_counts.get(fg, 0)
            if diff > 0:
                broken.extend([fg] * diff)

        return {"formed": sorted(formed), "broken": sorted(broken)}

    except Exception as e:
        logger.warning(f"FG detection failed for {mapped_rxn[:50]}...: {e}")
        return {"formed": [], "broken": []}
