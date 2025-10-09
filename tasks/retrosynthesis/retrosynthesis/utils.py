import re
from time import sleep
from typing import Any

import psycopg2
from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
from psycopg2.extras import RealDictCursor
from rdkit import Chem
from retrosynthesis.constants import FG_PATTERNS
from rxnutils.chem.reaction import ChemicalReaction

from corral.utils.modal import remote_call
from corral.utils.tool_helpers import make_api_request

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en,es-ES;q=0.9,es;q=0.8",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

PRODUCTION_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "reactions_production_db",
    "user": "postgres",
    "password": "postgres",
}


def get_production_connection():
    """Connect to Phase B production database"""
    conn = psycopg2.connect(**PRODUCTION_DB_CONFIG)
    conn.set_client_encoding("UTF8")
    return conn


def search_by_template(template_id: str) -> dict[str, Any] | None:
    """
    Retrieve complete reaction information by reaction ID.

    Args:
        reaction_id (str): The unique reaction identifier (primary key)

    Returns:
        dict[str, Any]: dict containing all reaction table fields, or None if not found

    Example:
        >>> reaction = search_by_template(42)
        >>> if reaction:
        ...     print(reaction['retro_smarts_template'])
        ...     print(reaction['product_smiles'])
    """
    conn = get_production_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute(
            """
            SELECT
                reaction_id,
                template_hash,
                retro_smarts_template,
                canonical_smarts_template,
                mapped_rxn,
            FROM reactions
            WHERE reaction_id = %s
        """,
            (template_id,),
        )

        result = cursor.fetchone()

        if result:
            logger.info(f"Found reaction with reaction_id: {template_id}")
            return dict(result)
        else:
            logger.warning(f"No reaction found with reaction_id: {template_id}")
            return None

    except Exception as e:
        logger.error(f"Error retrieving reaction by reaction_id '{template_id}': {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def search_reactions_by_criteria(
    functional_groups_formed: list[str] | None = None,
    functional_groups_broken: list[str] | None = None,
    bonds_formed: list[str] | None = None,
    bonds_broken: list[str] | None = None,
    bonds_order_changed: list[str] | None = None,
    reference_smiles: str | None = None,
    use_product_fingerprint: bool = True,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Query reactions database by functional groups and bonds (broken/formed/order_changed),
    with optional exact substructure matching and similarity-based ranking.

    This function performs a multi-stage query:
    1. Filters reactions by functional groups (formed/broken) via junction tables
    2. Filters reactions by bonds (formed/broken/order_changed) via junction tables
    3. If reference_smiles provided:
       - Exact substructure match using RDKit's @> operator (correctness filter)
       - Similarity computed for ranking only (does not filter results)
    4. Returns comprehensive reaction information

    Args:
        functional_groups_formed: List of functional group keys that must be formed
            Example: ['carbonyl', 'ester']
        functional_groups_broken: List of functional group keys that must be broken
            Example: ['alcohol', 'amine']
        bonds_formed: List of bond labels that must be formed
            Example: ['6-7', '8-9']
        bonds_broken: List of bond labels that must be broken
            Example: ['5-6', '7-8']
        bonds_order_changed: List of bond labels with order changes
            Example: ['6-6 (1.0->2.0)']
        reference_smiles: SMILES string for chemical filtering
            If provided, results will be filtered by exact substructure match:
            reference molecule must contain the template pattern (qmol).
            Similarity is computed only for ranking, not filtering.
        use_product_fingerprint: If True, compare against product fingerprints/patterns,
            otherwise compare against reactant fingerprints/patterns (default True)
        limit: Maximum number of results to return (default 100)

    Returns:
        List of dictionaries, each containing:
            - reaction_id: Unique reaction identifier
            - template_hash: Unique template hash
            - retro_smarts_template: Retrosynthetic SMARTS pattern
            - canonical_smarts_template: Forward SMARTS pattern
            - mapped_rxn: Example mapped reaction SMILES
            - product_smiles: Product SMILES
            - reactant_smiles: Reactant SMILES
            - dataset: Source dataset name
            - similarity: Tanimoto similarity score (if reference_smiles provided)

    Raises:
        ValueError: If reference_smiles is invalid
        psycopg2.Error: If database query fails

    Examples:
        # Search for reactions that form esters and break alcohols
        >>> results = search_reactions_by_criteria(
        ...     functional_groups_formed=['ester'],
        ...     functional_groups_broken=['alcohol'],
        ...     limit=10
        ... )

        # Search with bond constraints and chemical filtering
        >>> results = search_reactions_by_criteria(
        ...     bonds_formed=['6-7'],
        ...     reference_smiles='CCO',
        ...     limit=50
        ... )

        # Complex query with multiple criteria
        >>> results = search_reactions_by_criteria(
        ...     functional_groups_formed=['carbonyl', 'ester'],
        ...     functional_groups_broken=['alcohol'],
        ...     bonds_formed=['6-7', '6-8'],
        ...     bonds_broken=['7-8'],
        ...     reference_smiles='CC(=O)OCC',
        ...     use_product_fingerprint=True,
        ...     limit=20
        ... )
    """
    conn = get_production_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Build the base query with Common Table Expressions (CTEs) for each filter
        params = []
        cte_list = []

        # CTE 0: Compute reference molecule and fingerprint once (if provided)
        if reference_smiles:
            try:
                # Validate SMILES
                mol = Chem.MolFromSmiles(reference_smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES string: {reference_smiles}")

                # Compute reference molecule and fingerprint once
                cte_list.append("""
                    q AS (
                        SELECT
                            mol_from_smiles(%s) AS qmol,
                            rdkit_fp(mol_from_smiles(%s)) AS qfp
                    )
                """)
                params.extend([reference_smiles, reference_smiles])
            except Exception as e:
                logger.error(
                    f"Error processing reference SMILES '{reference_smiles}': {e}"
                )
                raise ValueError(f"Invalid reference SMILES: {e}") from e

        # CTE 1: Filter by functional groups formed
        if functional_groups_formed:
            cte_list.append("""
                fg_formed AS (
                    SELECT rfgf.reaction_id
                    FROM reaction_functional_groups_formed rfgf
                    JOIN functional_groups fg ON rfgf.functional_group_id = fg.functional_group_id
                    WHERE fg.functional_group_key = ANY(%s)
                    GROUP BY rfgf.reaction_id
                    HAVING COUNT(DISTINCT fg.functional_group_key) = %s
                )
            """)
            params.extend([functional_groups_formed, len(functional_groups_formed)])

        # CTE 2: Filter by functional groups broken
        if functional_groups_broken:
            cte_list.append("""
                fg_broken AS (
                    SELECT rfgb.reaction_id
                    FROM reaction_functional_groups_broken rfgb
                    JOIN functional_groups fg ON rfgb.functional_group_id = fg.functional_group_id
                    WHERE fg.functional_group_key = ANY(%s)
                    GROUP BY rfgb.reaction_id
                    HAVING COUNT(DISTINCT fg.functional_group_key) = %s
                )
            """)
            params.extend([functional_groups_broken, len(functional_groups_broken)])

        # CTE 3: Filter by bonds formed
        if bonds_formed:
            cte_list.append("""
                bonds_formed_cte AS (
                    SELECT rbf.reaction_id
                    FROM reaction_bonds_formed rbf
                    JOIN bonds b ON rbf.bond_id = b.bond_id
                    WHERE b.bond_label = ANY(%s)
                    GROUP BY rbf.reaction_id
                    HAVING COUNT(DISTINCT b.bond_label) = %s
                )
            """)
            params.extend([bonds_formed, len(bonds_formed)])

        # CTE 4: Filter by bonds broken
        if bonds_broken:
            cte_list.append("""
                bonds_broken_cte AS (
                    SELECT rbb.reaction_id
                    FROM reaction_bonds_broken rbb
                    JOIN bonds b ON rbb.bond_id = b.bond_id
                    WHERE b.bond_label = ANY(%s)
                    GROUP BY rbb.reaction_id
                    HAVING COUNT(DISTINCT b.bond_label) = %s
                )
            """)
            params.extend([bonds_broken, len(bonds_broken)])

        # CTE 5: Filter by bonds with order changed
        if bonds_order_changed:
            cte_list.append("""
                bonds_order_cte AS (
                    SELECT rboc.reaction_id
                    FROM reaction_bonds_order_changed rboc
                    JOIN bonds b ON rboc.bond_id = b.bond_id
                    WHERE b.bond_label = ANY(%s)
                    GROUP BY rboc.reaction_id
                    HAVING COUNT(DISTINCT b.bond_label) = %s
                )
            """)
            params.extend([bonds_order_changed, len(bonds_order_changed)])

        # Build the main query
        query = "WITH " + ",".join(cte_list) + "\n" if cte_list else ""

        # Determine which columns to select
        fp_column = (
            "product_pattern_fp" if use_product_fingerprint else "reactant_pattern_fp"
        )
        qmol_column = "product_qmol" if use_product_fingerprint else "reactant_qmol"

        # Select from reactions table
        if reference_smiles:
            # Include similarity for ranking (computed from pre-calculated CTE)
            query += f"""
                SELECT DISTINCT
                    r.reaction_id,
                    r.template_hash,
                    r.retro_smarts_template,
                    r.canonical_smarts_template,
                    r.mapped_rxn,
                    r.product_smiles,
                    r.reactant_smiles,
                    r.dataset,
                    r.source_row_id,
                    r.derive_version,
                    tanimoto_sml(q.qfp, r.{fp_column}) AS similarity
                FROM reactions r
                CROSS JOIN q
            """
        else:
            query += """
                SELECT DISTINCT
                    r.reaction_id,
                    r.template_hash,
                    r.retro_smarts_template,
                    r.canonical_smarts_template,
                    r.mapped_rxn,
                    r.product_smiles,
                    r.reactant_smiles,
                    r.dataset,
                    r.source_row_id,
                    r.derive_version
                FROM reactions r
            """

        # Add WHERE clause to join all CTEs
        where_conditions = []
        if functional_groups_formed:
            where_conditions.append(
                "r.reaction_id IN (SELECT reaction_id FROM fg_formed)"
            )
        if functional_groups_broken:
            where_conditions.append(
                "r.reaction_id IN (SELECT reaction_id FROM fg_broken)"
            )
        if bonds_formed:
            where_conditions.append(
                "r.reaction_id IN (SELECT reaction_id FROM bonds_formed_cte)"
            )
        if bonds_broken:
            where_conditions.append(
                "r.reaction_id IN (SELECT reaction_id FROM bonds_broken_cte)"
            )
        if bonds_order_changed:
            where_conditions.append(
                "r.reaction_id IN (SELECT reaction_id FROM bonds_order_cte)"
            )

        # Add exact substructure match for reference_smiles (correctness filter)
        if reference_smiles:
            where_conditions.append(f"r.{qmol_column} IS NOT NULL")
            where_conditions.append(f"q.qmol @> r.{qmol_column}")

        if where_conditions:
            query += "\nWHERE " + " AND ".join(where_conditions)

        # Add ordering and limit
        if reference_smiles:
            # Order by similarity (best matches first)
            query += "\nORDER BY similarity DESC"
        else:
            query += "\nORDER BY r.reaction_id"

        query += "\nLIMIT %s"
        params.append(limit)

        # Execute query
        logger.info(
            f"Executing query with filters: "
            f"fg_formed={functional_groups_formed}, "
            f"fg_broken={functional_groups_broken}, "
            f"bonds_formed={bonds_formed}, "
            f"bonds_broken={bonds_broken}, "
            f"bonds_order_changed={bonds_order_changed}, "
            f"reference_smiles={reference_smiles}"
        )

        cursor.execute(query, params)
        results = cursor.fetchall()

        logger.info(f"Found {len(results)} matching reactions")

        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Error executing reaction search query: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def apply_template_retro(product_smiles: str, template_id: str) -> list[str]:
    """
    Verify a retrosynthesis step by checking if the given reaction template can produce the expected product.
    This function should be replaced with a call to a retrosynthesis prediction model or API.

    product_smiles: str
        The SMILES of the expected product molecule.
    template_id: str
        The identifier of the reaction template to use.

    Returns:
        list[str]: A list of SMILES strings representing the predicted reactants.
    """
    reaction_data = search_by_template(template_id)
    if reaction_data is None:
        raise ValueError(f"Template ID {template_id} not found in database.")
    try:
        rxn = ChemicalReaction(reaction_data["mapped_rxn"])
        rxn.generate_reaction_template()
        return rxn.retro_template.apply(product_smiles)
    except Exception as e:
        raise Exception(
            f"Error applying template {template_id} to {product_smiles}: {e}"
        ) from e


def extract_chemical_info(element: Tag) -> dict[str, Any]:
    """
    Extract chemical information from HTML element
    Returns a dictionary with chemical name, CAS number, purity, amount, and price
    """
    # Initialize result dictionary
    chemical_info = {
        "chemical_name": None,
        "cas_number": None,
        "amount": None,
        "price_per_100g": None,
    }

    # Extract chemical name
    chemical_name_cell = element.find("td", class_="chemical-name")
    if chemical_name_cell:
        # Get the text and clean it up
        name_text = chemical_name_cell.get_text().strip()
        # Remove the icon and extra whitespace, split by newlines and take first part
        chemical_info["chemical_name"] = name_text.split("\n")[0].strip()

    # Extract CAS number
    cas_cell = element.find("td", class_="cas-cell")
    if cas_cell:
        cas_span = cas_cell.find("span", class_="cas-tooltip")
        if cas_span and cas_span.get("data-cas"):
            chemical_info["cas_number"] = cas_span.get("data-cas")
        else:
            # Fallback: extract from text
            cas_text = cas_cell.get_text().strip()
            cas_match = re.search(r"\d+-\d+-\d+", cas_text)
            if cas_match:
                chemical_info["cas_number"] = cas_match.group()

    # Extract purity
    purity_cell = element.find("td", class_="purity-column")
    if purity_cell:
        purity_div = purity_cell.find("div", class_="purity-value")
        if purity_div:
            chemical_info["purity"] = purity_div.get_text().strip()

    # Extract amount/packaging
    packaging_cells = element.find_all("td")
    for cell in packaging_cells:
        price_per_100_div = cell.find("div", class_="price-per-100")
        packaging_info_div = cell.find("div", class_="packaging-info")

        if price_per_100_div and packaging_info_div:
            amount_text = price_per_100_div.get_text().strip()
            if not amount_text.startswith("$"):  # Make sure it's not a price
                chemical_info["amount"] = amount_text
            break

    # Extract prices
    price_cells = element.find_all("div", class_="price-per-100")
    for price_cell in price_cells:
        price_text = price_cell.get_text().strip()
        if "per 100 g" in price_text:
            chemical_info["price_per_100g"] = price_text

    # Extract supplier info
    supplier_cell = element.find("td", class_="supplier-cell")
    if supplier_cell:
        supplier_name_link = supplier_cell.find("a", class_="supplier-name")
        if supplier_name_link:
            chemical_info["supplier"] = supplier_name_link.get_text().strip()

    return chemical_info


def extract_chemicals(text: str) -> list[dict[str, Any]]:
    """Extract chemical information dictionaries from text using BeautifulSoup."""
    soup = BeautifulSoup(text, "html.parser")
    rows = soup.select("tr.expandable-row")
    return [extract_chemical_info(row) for row in rows]


def search_catalog(cas: str) -> list[dict[str, Any]] | str:
    """Searches a catalog for available precursors. Returns a list of chemical info dicts or a not-found message."""

    sleep(5)
    try:
        response = make_api_request(
            url=f"https://www.chemicalsuppliers.com/buy-?cas_numbers%5B%5D={cas}&physical_state_filter=all",
            method="GET",
            headers=HEADERS,
            params={},
            verbose=False,
            json=False,
        )
        chemicals = extract_chemicals(response)
        return chemicals if chemicals else []
    except Exception as e:
        raise Exception(f"Error searching catalog for CAS {cas}: {e}") from e


def _is_buyable(smiles: str) -> bool:
    chemicals = search_catalog(smiles)
    return bool(chemicals)


def check_price(smiles: str) -> float:
    """
    Function to check the price of a molecule given its SMILES.
    This should be replaced with a function that queries a pricing database or API.

    smiles: str
        The SMILES of the molecule to check.

    Returns:
        float: The price of the molecule in USD.
    """
    cas_number = remote_call(function_name="return_cas_number", env_name="chemenv")(
        compound=smiles
    )
    chemicals = search_catalog(cas_number)
    price = chemicals[0].get("price_per_100g", "$0")
    try:
        return float(price.replace("$", "").replace(",", ""))
    except Exception as e:
        raise ValueError(f"Could not parse price: {price}") from e


def valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    return Chem.MolFromSmiles(smiles) is not None


def species_match(predicted: list[str], ground_truth: list[str]) -> bool:
    """Check if two lists of species (SMILES) match, ignoring order."""
    # First check if both lists have the same length
    if len(ground_truth) != len(predicted):
        return False

    # Validate all SMILES strings
    for smiles in ground_truth + predicted:
        if not valid_smiles(smiles):
            return False

    # Convert all SMILES to canonical form for comparison
    actual_canonical = []
    predicted_canonical = []

    for smiles in ground_truth:
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        if mol is None:
            return False
        actual_canonical.append(Chem.MolToSmiles(mol))

    for smiles in predicted:
        mol = Chem.MolFromSmiles(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        if mol is None:
            return False
        predicted_canonical.append(Chem.MolToSmiles(mol))

    # Check if both lists contain the same canonical SMILES (ignoring order)
    return sorted(actual_canonical) == sorted(predicted_canonical)


def apply_template_forward(reactants: str, template_id: str) -> list[str]:
    """
    Apply a reaction template in the forward direction to predict products.

    reactants: str
        The SMILES of the reactants, separated by dots if multiple.
    template_id: str
        The identifier of the reaction template to use.

    Returns:
        list[str]: The predicted product SMILES strings.
    """
    reaction_data = search_by_template(template_id)
    try:
        rxn = ChemicalReaction(reaction_data["mapped_rxn"])
        rxn.generate_reaction_template()
        return rxn.canonical_template.apply(reactants)
    except Exception as e:
        raise Exception(
            f"Error applying template {template_id} to {reactants}: {e}"
        ) from e


def _fragment_mapped_smiles(mol: Chem.Mol, atom_indices: tuple[int, ...]) -> str:
    """
    Returns a fragment SMILES for the specified atoms with atom-map numbers set to
    their original indices + 1. Only the fragment is emitted.
    """
    # Work on a copy to avoid mutating the caller's mol
    mc = Chem.Mol(mol)
    # Clear any pre-existing map numbers
    for a in mc.GetAtoms():
        a.SetAtomMapNum(0)
    # Assign map numbers (index + 1 so it's easy to read)
    for idx in atom_indices:
        mc.GetAtomWithIdx(idx).SetAtomMapNum(idx + 1)
    # Emit just the fragment; RDKit preserves atom-map numbers (":n") in SMILES
    return Chem.MolFragmentToSmiles(
        mc,
        atomsToUse=list(atom_indices),
        isomericSmiles=True,
        canonical=True,
        rootedAtAtom=atom_indices[0],
        allHsExplicit=False,
        allBondsExplicit=False,
    )


def _get_full_mapped_smiles(
    mol: Chem.Mol, all_atom_indices: list[tuple[int, ...]]
) -> str:
    """
    Returns the full molecule SMILES with atom-map numbers for all detected functional group atoms.
    """
    # Work on a copy to avoid mutating the caller's mol
    mc = Chem.Mol(mol)
    # Clear any pre-existing map numbers
    for a in mc.GetAtoms():
        a.SetAtomMapNum(0)

    # Collect all unique atom indices from all functional groups
    mapped_atoms = set()
    for atom_indices in all_atom_indices:
        mapped_atoms.update(atom_indices)

    # Assign map numbers (index + 1 so it's easy to read)
    for idx in sorted(mapped_atoms):
        mc.GetAtomWithIdx(idx).SetAtomMapNum(idx + 1)

    # Return the full molecule SMILES with mappings
    return Chem.MolToSmiles(mc, isomericSmiles=True, canonical=True)


def detect_functional_groups_in_molecule(smiles: str) -> list[str]:
    """
    Detect functional groups in a single molecule.

    Args:
        smiles (str): SMILES string

    Returns:
        list[str]: List of functional group names detected
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    detected = set()
    for name, patt in FG_PATTERNS.items():
        if mol.HasSubstructMatch(patt):
            detected.add(name)

    return sorted(detected)


def get_functional_groups(product: str) -> list[str]:
    """
    Detect functional groups formed and broken in reaction.

    Args:
        mapped_rxn (str): Atom-mapped reaction SMILES

    Returns:
        dict[str, list[str]]: Dict with keys:
        - 'formed': List of FG names formed in products
        - 'broken': List of FG names broken from reactants
    """
    return detect_functional_groups_in_molecule(product)


def summarize_groups_with_full_mapping(smiles: str, result_dict, use_collapsed=True):
    """
    Enhanced version that includes the full molecule SMILES with all functional group atoms mapped.
    Returns a dict where:
    - Keys are mapped SMILES fragments
    - Values are dicts containing:
      - 'group': functional group name
      - 'positions': tuple of atom indices
      - 'full_mapped_smiles': full molecule with all functional group atoms mapped
    """
    data = result_dict["collapsed"] if use_collapsed else result_dict["raw"]

    if not data:
        return {"groups": {}, "full_mapped_smiles": smiles}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"groups": {}, "full_mapped_smiles": smiles}

    # Get all positions for full mapping
    all_positions = [hit["positions"] for hit in data]
    full_mapped_smiles = _get_full_mapped_smiles(mol, all_positions)

    groups = {}
    for hit in data:
        mapped_smiles = hit["mapped_smiles"]
        groups[mapped_smiles] = {
            "group": hit["group"],
            "positions": hit["positions"],
            "smarts": hit["smarts"],
        }

    return {"groups": groups, "full_mapped_smiles": full_mapped_smiles}


def get_molecule_summary(smiles: str, result_dict, use_collapsed=True) -> str:
    """
    Format a molecule's functional group analysis as a formatted string.

    Args:
        name: Name/identifier for the molecule
        smiles: The original SMILES string
        result_dict: Result from detect_functional_groups()
        use_collapsed: Whether to use collapsed results

    Returns:
        Formatted string with molecule info and functional groups
    """
    lines = []
    lines.append(f"{smiles}")

    # Using the enhanced version with full mapping
    summary = summarize_groups_with_full_mapping(smiles, result_dict, use_collapsed)
    lines.append(f"Full mapped SMILES: {summary['full_mapped_smiles']}")
    lines.append("Groups found:")

    if summary["groups"]:
        for mapped_smiles, info in summary["groups"].items():
            # Convert 0-based positions to 1-based to match the atom map numbers in SMILES
            mapped_positions = tuple(pos + 1 for pos in info["positions"])
            lines.append(
                f"  - {info['group']:16s} pos={mapped_positions}  frag={mapped_smiles}"
            )
    else:
        lines.append("  - No functional groups detected")

    return "\n".join(lines)
