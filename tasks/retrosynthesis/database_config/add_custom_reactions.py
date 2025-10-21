"""
Add Custom Reactions to Production Database

This script allows you to add custom reaction SMILES to the production database.
It processes each reaction to extract:
- Reaction templates (retro and forward SMARTS)
- RDKit query molecules and fingerprints
- Bond changes (formed/broken/order_changed)
- Functional groups (formed/broken)

After processing, it prints the reaction_id for each successfully added reaction.

Usage:
    python add_custom_reactions.py

The script will prompt you to enter reaction SMILES, or you can modify
the CUSTOM_REACTIONS list in the script directly.
"""

from typing import Any

import psycopg2
from loguru import logger
from phase_b_chemistry_utils import get_functional_groups, obtain_bonds
from psycopg2.extras import RealDictCursor

# Import chemistry tools
from rxnutils.chem.reaction import ChemicalReaction

# Production database configuration
PRODUCTION_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "reactions_production_db",
    "user": "postgres",
    "password": "postgres",
}

# Derive version for these custom additions
DERIVE_VERSION = "custom_v1"

# RDKit function names
RDKIT_QMOL_FUNC = "qmol_from_smarts"
RDKIT_FP_FUNC = "rdkit_fp"

CUSTOM_REACTIONS = [
    "([CH3:1]/[C:2]([c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1)=[CH:9]/[CH2:10][CH2:11][CH2:12][C:13](=[O:14])[C:15]([F:16])([F:17])[F:18])>>([CH3:1][C@H:2]([c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1)[C@@H:9]1[CH2:10][CH2:11][CH2:12][C@@:13]1([OH:14])[C:15]([F:16])([F:17])[F:18])",
    "[CH2:6]=[CH:7][CH:8]=[CH2:9].[CH3:1][O:2][C:3](=[O:4])[C:5]1=[CH:10][C:11](=[O:12])[CH2:13]1>>[CH3:1][O:2][C:3](=[O:4])[C@:5]12[CH2:6][CH:7]=[CH:8][CH2:9][C@H:10]1[C:11](=[O:12])[CH2:13]2",
    "[CH:14]#[C:15][Si:16]([CH3:17])([CH3:18])[CH3:19].O=[C:5]([C:3]([O:2][CH3:1])=[O:4])[CH2:6][CH2:7][c:8]1[cH:9][cH:10][cH:11][cH:12][cH:13]1>>[CH3:1][O:2][C:3](=[O:4])[C:5]1(/[CH:6]=[CH:7]/[c:8]2[cH:9][cH:10][cH:11][cH:12][cH:13]2)[CH:14]=[C:15]1[Si:16]([CH3:17])([CH3:18])[CH3:19]",
    "O=[C:2]([c:3]1[cH:4][cH:5][c:6]([O:7][CH3:8])[cH:9][cH:10]1)[c:11]1[cH:12][cH:13][c:14]([O:15][CH3:16])[cH:17][cH:18]1.[Li][CH2:1][Si](C)(C)C>>[CH2:1]=[C:2]([c:3]1[cH:4][cH:5][c:6]([O:7][CH3:8])[cH:9][cH:10]1)[c:11]1[cH:12][cH:13][c:14]([O:15][CH3:16])[cH:17][cH:18]1",
    "(CC(C)(C)[Si](C)(C)[O:17][CH2:16][C:14]([C:12]([CH2:11][C@@H:10]1[C:4]([C:2](=[CH2:1])[CH3:3])=[CH:5][C:6](=[O:7])[C@H:8]1[CH3:9])=[O:13])=[CH2:15])>>([CH2:1]=[C:2]([CH3:3])[C:4]1=[CH:5][C:6](=[O:7])[C@@H:8]([CH3:9])[C@@H:10]1[CH2:11][C:12](=[O:13])[C:14](=[CH2:15])[CH2:16][OH:17])",
    "([CH2:1]=[C:2]([CH3:3])[C:4]1=[CH:5][C@H:6]([OH:7])[C@@H:8]([CH3:9])[C@@H:10]1[CH2:11][C@H:12]([OH:13])[C:14](=[CH2:15])[CH2:16][O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24])>>([CH2:1]=[C:2]([CH3:3])[C:4]1=[CH:5][C:6](=[O:7])[C@@H:8]([CH3:9])[C@@H:10]1[CH2:11][C:12](=[O:13])[C:14](=[CH2:15])[CH2:16][O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24])",
    "[CH2:1]=[C:2]([CH3:3])[C:4]1=[CH:5][C@H:6]([OH:7])[C@@H:8]([CH3:9])[C@@H:10]1[CH2:11][CH:12]=[O:13].I[C:14](=[CH2:15])[CH2:16][O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24]>>[CH2:1]=[C:2]([CH3:3])[C:4]1=[CH:5][C@H:6]([OH:7])[C@@H:8]([CH3:9])[C@@H:10]1[CH2:11][C@H:12]([OH:13])[C:14](=[CH2:15])[CH2:16][O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24]",
    "(CC(C)(C)[Si](C)(C)[O:17][C@H:16]1[CH2:15][C@H:6]([CH2:7]/[CH:8]=[CH:9]/[CH:10]=[CH:11]/[C:12](=[O:13])[OH:14])[O:5][C@H:4](/[CH:3]=[C:2](/[CH3:1])[CH2:20][OH:21])[C@@H:18]1[CH3:19])>>([CH3:1]/[C:2](=[CH:3]/[C@H:4]1[O:5][C@@H:6]([CH2:7]/[CH:8]=[CH:9]/[CH:10]=[CH:11]/[C:12](=[O:13])[OH:14])[CH2:15][C@H:16]([OH:17])[C@H:18]1[CH3:19])[CH2:20][OH:21])",
    "O=[CH:8][CH2:7][C@@H:6]1[O:5][C@H:4](/[CH:3]=[C:2](/[CH3:1])[CH2:27][OH:28])[C@H:25]([CH3:26])[C@@H:16]([O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24])[CH2:15]1.CCOP(=O)(OCC)[CH2:9]/[CH:10]=[CH:11]/[C:12](=[O:13])[O:14]CC>>[CH3:1]/[C:2](=[CH:3]/[C@H:4]1[O:5][C@@H:6]([CH2:7]/[CH:8]=[CH:9]/[CH:10]=[CH:11]/[C:12](=[O:13])[OH:14])[CH2:15][C@H:16]([O:17][Si:18]([CH3:19])([CH3:20])[C:21]([CH3:22])([CH3:23])[CH3:24])[C@H:25]1[CH3:26])[CH2:27][OH:28]",
    "(CCO[C:8]([CH2:7][C@@H:6]1[O:5][C@H:4](/[CH:3]=[C:2](/[CH3:1])[C:22](OC)=[O:23])[C@H:20]([CH3:21])[C@@H:11]([O:12][Si:13]([CH3:14])([CH3:15])[C:16]([CH3:17])([CH3:18])[CH3:19])[CH2:10]1)=[O:9])>>([CH3:1]/[C:2](=[CH:3]/[C@H:4]1[O:5][C@@H:6]([CH2:7][CH:8]=[O:9])[CH2:10][C@H:11]([O:12][Si:13]([CH3:14])([CH3:15])[C:16]([CH3:17])([CH3:18])[CH3:19])[C@H:20]1[CH3:21])[CH2:22][OH:23])",
    "O=S(=O)(O[Si:11]([CH3:12])([CH3:13])[C:14]([CH3:15])([CH3:16])[CH3:17])C(F)(F)F.[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([OH:10])[C@@H:18]([CH3:19])[C@@H:20](/[CH:21]=[C:22](/[CH3:23])[C:24](=[O:25])[O:26][CH3:27])[O:28]1>>[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([O:10][Si:11]([CH3:12])([CH3:13])[C:14]([CH3:15])([CH3:16])[CH3:17])[C@@H:18]([CH3:19])[C@@H:20](/[CH:21]=[C:22](/[CH3:23])[C:24](=[O:25])[O:26][CH3:27])[O:28]1",
    "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([OH:10])[C@@H:11]([CH3:12])[C@@H:13]([CH:14]=[O:18])[O:21]1.C[CH:16](P(=O)(OCC(F)(F)F)OCC(F)(F)F)[C:15]#[C:17][O:19][CH3:20]>>[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH2:6][C@H:7]1[CH2:8][C@H:9]([OH:10])[C@@H:11]([CH3:12])[C@@H:13](/[CH:14]=[C:15](/[CH3:16])[C:17](=[O:18])[O:19][CH3:20])[O:21]1",
    "[O:1]=[C:2]1[CH2:3][CH2:4][CH2:5][CH:6]2[O:7][c:8]3[cH:9][cH:10][cH:11][cH:12][c:13]3[CH:14]=[C:15]12>>[OH:1][C@@H:2]1[CH2:3][CH2:4][CH2:5][C@@H:6]2[O:7][c:8]3[cH:9][cH:10][cH:11][cH:12][c:13]3[CH:14]=[C:15]12",
    "C1CN2CCN1CC2.[O:1]=[C:2]1[CH2:3][CH2:4][CH2:5][CH:6]=[CH:15]1.O=[CH:14][c:13]1[c:8]([OH:7])[cH:9][cH:10][cH:11][cH:12]1>>[O:1]=[C:2]1[CH2:3][CH2:4][CH2:5][CH:6]2[O:7][c:8]3[cH:9][cH:10][cH:11][cH:12][c:13]3[CH:14]=[C:15]12",
    "[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][c:7]([NH:8][C:9](=[O:10])[c:11]2[n:12][n:13][nH:14][c:28]2[NH2:29])[cH:30]1.Cl[CH2:15][c:16]1[cH:17][cH:18][c:19]([CH2:20][N:21]2[CH2:22][CH:23]([F:24])[CH2:25]2)[cH:26][cH:27]1>>[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][c:7]([NH:8][C:9](=[O:10])[c:11]2[n:12][n:13][n:14]([CH2:15][c:16]3[cH:17][cH:18][c:19]([CH2:20][N:21]4[CH2:22][CH:23]([F:24])[CH2:25]4)[cH:26][cH:27]3)[c:28]2[NH2:29])[cH:30]1",
    "Cl[CH2:10][Cl:11].[F:1][CH:2]1[CH2:3][N:4]([CH2:5][c:6]2[cH:7][cH:8][cH:9][cH:12][cH:13]2)[CH2:14]1>>[F:1][CH:2]1[CH2:3][N:4]([CH2:5][c:6]2[cH:7][cH:8][c:9]([CH2:10][Cl:11])[cH:12][cH:13]2)[CH2:14]1",
    "O[CH:2]1[CH2:3][N:4]([CH2:5][c:6]2[cH:7][cH:8][cH:9][cH:10][cH:11]2)[CH2:12]1>>[F:1][CH:2]1[CH2:3][N:4]([CH2:5][c:6]2[cH:7][cH:8][cH:9][cH:10][cH:11]2)[CH2:12]1",
    "[CH3:1][c:2]1[cH:7][cH:6][c:5]([NH:8][S:9](=[O:10])([c:12]2[cH:17][cH:16][c:15](/[CH:18]=[CH:19]/[C:20](O)=[O:21])[cH:14][cH:13]2)=[O:11])[cH:4][cH:3]1.[NH2:28][c:26]3[cH:27][cH:22][cH:23][cH:24][c:25]3[NH2:29]>>[CH3:1][c:2]4[cH:7][cH:6][c:5]([NH:8][S:9](=[O:10])([c:12]5[cH:17][cH:16][c:15](/[CH:18]=[CH:19]/[C:20]([NH:28][c:26]6[c:25]([NH2:29])[cH:24][cH:23][cH:22][cH:27]6)=[O:21])[cH:14][cH:13]5)=[O:11])[cH:4][cH:3]4",
    "O=[CH:1][c:2]1[cH:18][cH:17][c:5]([S:6]([NH:7][c:8]2[cH:14][cH:13][c:11]([CH3:12])[cH:10][cH:9]2)(=[O:15])=[O:16])[cH:4][cH:3]1.O=C([CH2:19][C:20]([OH:22])=[O:21])O>>[CH3:12][c:11]3[cH:13][cH:14][c:8]([NH:7][S:6](=[O:15])([c:5]4[cH:17][cH:18][c:2](/[CH:1]=[CH:19]/[C:20]([OH:22])=[O:21])[cH:3][cH:4]4)=[O:16])[cH:9][cH:10]3",
    "[CH3:1][c:2]1[cH:8][cH:7][c:5]([NH2:6])[cH:4][cH:3]1.Cl[S:9](=[O:10])([c:12]2[cH:19][cH:18][c:15]([CH:16]=[O:17])[cH:14][cH:13]2)=[O:11]>>[CH3:1][c:2]3[cH:8][cH:7][c:5]([NH:6][S:9](=[O:10])([c:12]4[cH:19][cH:18][c:15]([CH:16]=[O:17])[cH:14][cH:13]4)=[O:11])[cH:4][cH:3]3",
    "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][N:11]([c:12]2[cH:13][c:14]([NH2:15])[cH:26][c:27]3[c:28]2[O:29][C:30]2([CH2:31][CH2:32][CH2:33]2)[CH2:34][CH2:35]3)[CH2:36][CH2:37]1.Cl[S:16](=[O:17])(=[O:18])[c:19]1[cH:20][cH:21][cH:22][cH:23][c:24]1[F:25]>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][N:11]([c:12]2[cH:13][c:14]([NH:15][S:16](=[O:17])(=[O:18])[c:19]3[cH:20][cH:21][cH:22][cH:23][c:24]3[F:25])[cH:26][c:27]3[c:28]2[O:29][C:30]2([CH2:31][CH2:32][CH2:33]2)[CH2:34][CH2:35]3)[CH2:36][CH2:37]1",
    "O=[N+:15]([O-])[c:14]1[cH:13][c:12]([N:11]2[CH2:10][CH2:9][N:8]([C:6]([O:5][C:2]([CH3:1])([CH3:3])[CH3:4])=[O:7])[CH2:27][CH2:26]2)[c:18]2[c:17]([cH:16]1)[CH2:25][CH2:24][C:20]1([O:19]2)[CH2:21][CH2:22][CH2:23]1>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][N:11]([c:12]2[cH:13][c:14]([NH2:15])[cH:16][c:17]3[c:18]2[O:19][C:20]2([CH2:21][CH2:22][CH2:23]2)[CH2:24][CH2:25]3)[CH2:26][CH2:27]1",
    "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][NH:11][CH2:28][CH2:29]1.Br[c:12]1[cH:13][c:14]([N+:15](=[O:16])[O-:17])[cH:18][c:19]2[c:20]1[O:21][C:22]1([CH2:23][CH2:24][CH2:25]1)[CH2:26][CH2:27]2>>[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[N:8]1[CH2:9][CH2:10][N:11]([c:12]2[cH:13][c:14]([N+:15](=[O:16])[O-:17])[cH:18][c:19]3[c:20]2[O:21][C:22]2([CH2:23][CH2:24][CH2:25]2)[CH2:26][CH2:27]3)[CH2:28][CH2:29]1",
    "Br[Br:7].[O:1]=[N+:2]([O-:3])[c:4]1[cH:5][cH:6][c:8]2[c:9]([cH:10]1)[CH2:11][CH2:12][C:13]1([CH2:14][CH2:15][CH2:16]1)[O:17]2>>[O:1]=[N+:2]([O-:3])[c:4]1[cH:5][c:6]([Br:7])[c:8]2[c:9]([cH:10]1)[CH2:11][CH2:12][C:13]1([CH2:14][CH2:15][CH2:16]1)[O:17]2",
    "O=[C:10]1[c:8]2[c:7]([cH:6][cH:5][c:4]([N+:2](=[O:1])[O-:3])[cH:9]2)[O:16][C:12]2([CH2:11]1)[CH2:13][CH2:14][CH2:15]2>>[O:1]=[N+:2]([O-:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9]1)[CH2:10][CH2:11][C:12]1([CH2:13][CH2:14][CH2:15]1)[O:16]2",
    "[O:1]=[C:2]([CH3:3])[c:17]1[c:9]([OH:8])[cH:10][cH:11][c:12]([N+:13](=[O:14])[O-:15])[cH:16]1.O=[C:4]1[CH2:5][CH2:6][CH2:7]1>>[O:1]=[C:2]1[CH2:3][C:4]2([CH2:5][CH2:6][CH2:7]2)[O:8][c:9]2[cH:10][cH:11][c:12]([N+:13](=[O:14])[O-:15])[cH:16][c:17]21",
]


def get_production_connection():
    """Connect to production database"""
    conn = psycopg2.connect(**PRODUCTION_DB_CONFIG)
    conn.set_client_encoding("UTF8")
    return conn


def test_database_connection():
    """Test connection and verify schema exists"""
    try:
        conn = get_production_connection()
        cursor = conn.cursor()

        # Check if reactions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'reactions'
            )
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            logger.error("Error: 'reactions' table does not exist in the database!")
            logger.error(
                "Please run the phase_b_production.py script first to create the schema."
            )
            cursor.close()
            conn.close()
            return False

        # Verify RDKit extension
        cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'rdkit'")
        rdkit_exists = cursor.fetchone()[0] > 0

        if not rdkit_exists:
            logger.error("Error: RDKit extension is not installed!")
            cursor.close()
            conn.close()
            return False

        logger.info("✓ Database connection successful")
        logger.info("✓ Schema verified")
        logger.info("✓ RDKit extension available")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def generate_templates(mapped_rxn: str) -> dict[str, str] | None:
    """
    Generate retro and forward SMARTS templates from mapped reaction.

    Returns:
        Dict with keys:
        - 'retro_smarts': Product-side SMARTS (for retro applicability)
        - 'canonical_smarts': Canonical template
        - 'template_hash': Hash computed from rxn.retro_template.hash_from_bits()
        - 'product_smiles': Product SMILES (for fingerprint generation)
        - 'reactant_smiles': Reactant SMILES (for fingerprint generation)
    """
    try:
        rxn = ChemicalReaction(mapped_rxn)
        rxn.generate_reaction_template()

        # Compute hash using the retro_template method
        template_hash = rxn.retro_template.hash_from_bits()

        # Extract SMILES for fingerprint generation
        reactants_smiles = ".".join(rxn.reactants_list) if rxn.reactants_list else None
        products_smiles = ".".join(rxn.products_list) if rxn.products_list else None

        templates = {
            "retro_smarts": rxn.retro_template.smarts,
            "canonical_smarts": rxn.canonical_template.smarts,
            "template_hash": template_hash,
            "product_smiles": products_smiles,
            "reactant_smiles": reactants_smiles,
        }

        # Validation: Ensure retro SMARTS is valid
        if not templates["retro_smarts"] or ">>" not in templates["retro_smarts"]:
            logger.warning(
                f"Invalid retro SMARTS generated: {templates['retro_smarts']}"
            )
            return None

        return templates

    except Exception as e:
        rxn_preview = mapped_rxn[:100] + "..." if len(mapped_rxn) > 100 else mapped_rxn
        logger.error(f"Template generation failed: {e} | Reaction: {rxn_preview}")
        return None


def process_single_reaction(
    mapped_rxn: str, metadata: dict | None = None
) -> dict[str, Any]:
    """
    Process chemistry for a single custom reaction.

    Args:
        mapped_rxn: Atom-mapped reaction SMILES
        metadata: Optional dict with additional metadata (name, source, etc.)

    Returns:
        Dict with processing results including success status, templates, bonds, fgs
    """
    result = {"mapped_rxn": mapped_rxn, "success": False, "metadata": metadata or {}}

    try:
        # Step 1: Generate templates
        logger.info("Generating templates...")
        templates = generate_templates(mapped_rxn)
        if not templates:
            result["error"] = "Failed to generate templates"
            return result

        result["templates"] = templates
        logger.info(
            f"✓ Templates generated (hash: {templates['template_hash'][:16]}...)"
        )

        # Step 2: Extract bonds
        logger.info("Extracting bonds...")
        try:
            bonds = obtain_bonds(mapped_rxn)
            result["bonds"] = bonds
            logger.info(
                f"✓ Bonds extracted (formed: {len(bonds['formed'])}, broken: {len(bonds['broken'])}, order_changed: {len(bonds['order_changed'])})"
            )
        except Exception as e:
            result["error"] = f"obtain_bonds() failed: {e}"
            return result

        # Step 3: Extract functional groups
        logger.info("Extracting functional groups...")
        try:
            fgs = get_functional_groups(mapped_rxn)
            result["fgs"] = fgs
            logger.info(
                f"✓ Functional groups extracted (formed: {len(fgs['formed'])}, broken: {len(fgs['broken'])})"
            )
        except Exception as e:
            result["error"] = f"get_functional_groups() failed: {e}"
            return result

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def upsert_bond(cursor, bond_label: str) -> int:
    """Insert or get bond ID"""
    cursor.execute(
        """
        INSERT INTO bonds (bond_label)
        VALUES (%s)
        ON CONFLICT (bond_label) DO UPDATE SET bond_label = EXCLUDED.bond_label
        RETURNING bond_id
    """,
        (bond_label,),
    )
    return cursor.fetchone()[0]


def upsert_functional_group(cursor, fg_key: str) -> int:
    """Insert or get functional group ID"""
    cursor.execute(
        """
        INSERT INTO functional_groups (functional_group_key)
        VALUES (%s)
        ON CONFLICT (functional_group_key) DO UPDATE SET functional_group_key = EXCLUDED.functional_group_key
        RETURNING functional_group_id
    """,
        (fg_key,),
    )
    return cursor.fetchone()[0]


def insert_reaction_to_database(conn, result: dict[str, Any]) -> dict[str, Any] | None:
    """
    Insert or update a processed reaction in the database.

    Args:
        conn: Database connection
        result: Result dict from process_single_reaction()

    Returns:
        Dict with:
        - 'reaction_id': The reaction ID (new or existing)
        - 'is_updated': True if an existing entry was updated, False if newly inserted
        - 'template_hash': The template hash
        Returns None if operation fails
    """
    if not result["success"]:
        logger.error(f"Cannot insert failed reaction: {result.get('error')}")
        return None

    cursor = conn.cursor()

    try:
        templates = result["templates"]
        bonds = result["bonds"]
        fgs = result["fgs"]
        metadata = result["metadata"]

        # Get metadata fields
        dataset = metadata.get("dataset", "custom")
        source_row_id = metadata.get("source_row_id", None)

        # Step 0: Check if template already exists
        logger.info("Checking if template already exists...")
        cursor.execute(
            """
            SELECT reaction_id FROM reactions WHERE template_hash = %s
        """,
            (templates["template_hash"],),
        )

        existing_row = cursor.fetchone()
        if existing_row:
            existing_id = existing_row[0]
            logger.warning(
                f"⚠ Template already exists in database with reaction_id: {existing_id}"
            )
            logger.warning(f"  Template hash: {templates['template_hash']}")
            logger.warning("  Updating existing entry with new reaction data...")

            # Update the existing reaction with new data
            update_sql = """
                UPDATE reactions SET
                    retro_smarts_template = %s,
                    canonical_smarts_template = %s,
                    mapped_rxn = %s,
                    product_smiles = %s,
                    reactant_smiles = %s,
                    dataset = %s,
                    source_row_id = %s,
                    product_qmol = qmol_from_smarts(split_part(%s, '>>', 1)::cstring),
                    reactant_qmol = qmol_from_smarts(split_part(%s, '>>', 2)::cstring),
                    product_pattern_fp = CASE WHEN %s IS NOT NULL THEN rdkit_fp(mol_from_smiles(%s)) ELSE NULL END,
                    reactant_pattern_fp = CASE WHEN %s IS NOT NULL THEN rdkit_fp(mol_from_smiles(%s)) ELSE NULL END,
                    derive_version = %s
                WHERE reaction_id = %s
            """

            cursor.execute(
                update_sql,
                (
                    templates["retro_smarts"],  # 1. retro_smarts_template
                    templates.get("canonical_smarts"),  # 2. canonical_smarts_template
                    result["mapped_rxn"],  # 3. mapped_rxn
                    templates.get("product_smiles"),  # 4. product_smiles
                    templates.get("reactant_smiles"),  # 5. reactant_smiles
                    dataset,  # 6. dataset
                    source_row_id,  # 7. source_row_id
                    templates["retro_smarts"],  # 8. for product_qmol split
                    templates["retro_smarts"],  # 9. for reactant_qmol split
                    templates.get(
                        "product_smiles"
                    ),  # 10. CASE WHEN check for product_pattern_fp
                    templates.get(
                        "product_smiles"
                    ),  # 11. rdkit_fp(mol_from_smiles(product))
                    templates.get(
                        "reactant_smiles"
                    ),  # 12. CASE WHEN check for reactant_pattern_fp
                    templates.get(
                        "reactant_smiles"
                    ),  # 13. rdkit_fp(mol_from_smiles(reactant))
                    DERIVE_VERSION,  # 14. derive_version
                    existing_id,  # 15. WHERE reaction_id
                ),
            )

            logger.info(f"✓ Updated existing reaction with reaction_id: {existing_id}")
            reaction_id = existing_id
            is_updated = True

            # Delete existing bonds and functional groups before inserting new ones
            logger.info("Clearing existing bonds and functional groups...")
            cursor.execute(
                "DELETE FROM reaction_bonds_formed WHERE reaction_id = %s",
                (reaction_id,),
            )
            cursor.execute(
                "DELETE FROM reaction_bonds_broken WHERE reaction_id = %s",
                (reaction_id,),
            )
            cursor.execute(
                "DELETE FROM reaction_bonds_order_changed WHERE reaction_id = %s",
                (reaction_id,),
            )
            cursor.execute(
                "DELETE FROM reaction_functional_groups_formed WHERE reaction_id = %s",
                (reaction_id,),
            )
            cursor.execute(
                "DELETE FROM reaction_functional_groups_broken WHERE reaction_id = %s",
                (reaction_id,),
            )

        else:
            # Template doesn't exist, insert new reaction
            is_updated = False
            logger.info("Inserting new reaction into database...")

            insert_sql = """
                INSERT INTO reactions (
                    template_hash,
                    retro_smarts_template,
                    canonical_smarts_template,
                    mapped_rxn,
                    product_smiles,
                    reactant_smiles,
                    dataset,
                    source_row_id,
                    staging_id,
                    product_qmol,
                    reactant_qmol,
                    product_pattern_fp,
                    reactant_pattern_fp,
                    derive_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    qmol_from_smarts(split_part(%s, '>>', 1)::cstring),
                    qmol_from_smarts(split_part(%s, '>>', 2)::cstring),
                    CASE WHEN %s IS NOT NULL THEN rdkit_fp(mol_from_smiles(%s)) ELSE NULL END,
                    CASE WHEN %s IS NOT NULL THEN rdkit_fp(mol_from_smiles(%s)) ELSE NULL END,
                    %s
                )
                RETURNING reaction_id
            """

            cursor.execute(
                insert_sql,
                (
                    templates["template_hash"],  # 1. template_hash
                    templates["retro_smarts"],  # 2. retro_smarts_template
                    templates.get("canonical_smarts"),  # 3. canonical_smarts_template
                    result["mapped_rxn"],  # 4. mapped_rxn
                    templates.get("product_smiles"),  # 5. product_smiles
                    templates.get("reactant_smiles"),  # 6. reactant_smiles
                    dataset,  # 7. dataset
                    source_row_id,  # 8. source_row_id
                    None,  # 9. staging_id (NULL for custom reactions)
                    templates["retro_smarts"],  # 10. for product_qmol split
                    templates["retro_smarts"],  # 11. for reactant_qmol split
                    templates.get(
                        "product_smiles"
                    ),  # 12. CASE WHEN check for product_pattern_fp
                    templates.get(
                        "product_smiles"
                    ),  # 13. rdkit_fp(mol_from_smiles(product))
                    templates.get(
                        "reactant_smiles"
                    ),  # 14. CASE WHEN check for reactant_pattern_fp
                    templates.get(
                        "reactant_smiles"
                    ),  # 15. rdkit_fp(mol_from_smiles(reactant))
                    DERIVE_VERSION,  # 16. derive_version
                ),
            )

            reaction_id = cursor.fetchone()[0]
            logger.info(f"✓ Reaction inserted with reaction_id: {reaction_id}")

        # Step 2: Insert bonds
        logger.info("Inserting bonds...")

        # Bonds formed
        if bonds["formed"]:
            for bond in bonds["formed"]:
                bond_id = upsert_bond(cursor, bond)
                cursor.execute(
                    """
                    INSERT INTO reaction_bonds_formed (reaction_id, bond_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, bond_id) DO NOTHING
                """,
                    (reaction_id, bond_id),
                )
            logger.info(f"  ✓ {len(bonds['formed'])} bonds formed")

        # Bonds broken
        if bonds["broken"]:
            for bond in bonds["broken"]:
                bond_id = upsert_bond(cursor, bond)
                cursor.execute(
                    """
                    INSERT INTO reaction_bonds_broken (reaction_id, bond_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, bond_id) DO NOTHING
                """,
                    (reaction_id, bond_id),
                )
            logger.info(f"  ✓ {len(bonds['broken'])} bonds broken")

        # Bonds order changed
        if bonds["order_changed"]:
            for bond in bonds["order_changed"]:
                bond_id = upsert_bond(cursor, bond)
                cursor.execute(
                    """
                    INSERT INTO reaction_bonds_order_changed (reaction_id, bond_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, bond_id) DO NOTHING
                """,
                    (reaction_id, bond_id),
                )
            logger.info(f"  ✓ {len(bonds['order_changed'])} bonds order changed")

        # Step 3: Insert functional groups
        logger.info("Inserting functional groups...")

        # Functional groups formed
        if fgs["formed"]:
            for fg_key in fgs["formed"]:
                fg_id = upsert_functional_group(cursor, fg_key)
                cursor.execute(
                    """
                    INSERT INTO reaction_functional_groups_formed (reaction_id, functional_group_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, functional_group_id) DO NOTHING
                """,
                    (reaction_id, fg_id),
                )
            logger.info(f"  ✓ {len(fgs['formed'])} functional groups formed")

        # Functional groups broken
        if fgs["broken"]:
            for fg_key in fgs["broken"]:
                fg_id = upsert_functional_group(cursor, fg_key)
                cursor.execute(
                    """
                    INSERT INTO reaction_functional_groups_broken (reaction_id, functional_group_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, functional_group_id) DO NOTHING
                """,
                    (reaction_id, fg_id),
                )
            logger.info(f"  ✓ {len(fgs['broken'])} functional groups broken")

        conn.commit()
        cursor.close()

        return {
            "reaction_id": reaction_id,
            "is_updated": is_updated,
            "template_hash": templates["template_hash"],
        }

    except Exception as e:
        logger.error(f"Error inserting reaction into database: {e}")
        conn.rollback()
        cursor.close()
        return None


def get_reaction_info(conn, reaction_id: int) -> dict | None:
    """
    Retrieve complete information about a reaction from the database.

    Args:
        conn: Database connection
        reaction_id: ID of the reaction to retrieve

    Returns:
        Dict with complete reaction information
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Get basic reaction info
        cursor.execute(
            """
            SELECT
                reaction_id,
                template_hash,
                retro_smarts_template,
                canonical_smarts_template,
                mapped_rxn,
                product_smiles,
                reactant_smiles,
                dataset,
                source_row_id,
                staging_id,
                derive_version,
                created_at
            FROM reactions
            WHERE reaction_id = %s
        """,
            (reaction_id,),
        )

        reaction = cursor.fetchone()
        if not reaction:
            return None

        reaction = dict(reaction)

        # Get bonds formed
        cursor.execute(
            """
            SELECT b.bond_label
            FROM reaction_bonds_formed rbf
            JOIN bonds b ON rbf.bond_id = b.bond_id
            WHERE rbf.reaction_id = %s
        """,
            (reaction_id,),
        )
        reaction["bonds_formed"] = [row["bond_label"] for row in cursor.fetchall()]

        # Get bonds broken
        cursor.execute(
            """
            SELECT b.bond_label
            FROM reaction_bonds_broken rbb
            JOIN bonds b ON rbb.bond_id = b.bond_id
            WHERE rbb.reaction_id = %s
        """,
            (reaction_id,),
        )
        reaction["bonds_broken"] = [row["bond_label"] for row in cursor.fetchall()]

        # Get bonds order changed
        cursor.execute(
            """
            SELECT b.bond_label
            FROM reaction_bonds_order_changed rboc
            JOIN bonds b ON rboc.bond_id = b.bond_id
            WHERE rboc.reaction_id = %s
        """,
            (reaction_id,),
        )
        reaction["bonds_order_changed"] = [
            row["bond_label"] for row in cursor.fetchall()
        ]

        # Get functional groups formed
        cursor.execute(
            """
            SELECT fg.functional_group_key
            FROM reaction_functional_groups_formed rfgf
            JOIN functional_groups fg ON rfgf.functional_group_id = fg.functional_group_id
            WHERE rfgf.reaction_id = %s
        """,
            (reaction_id,),
        )
        reaction["functional_groups_formed"] = [
            row["functional_group_key"] for row in cursor.fetchall()
        ]

        # Get functional groups broken
        cursor.execute(
            """
            SELECT fg.functional_group_key
            FROM reaction_functional_groups_broken rfgb
            JOIN functional_groups fg ON rfgb.functional_group_id = fg.functional_group_id
            WHERE rfgb.reaction_id = %s
        """,
            (reaction_id,),
        )
        reaction["functional_groups_broken"] = [
            row["functional_group_key"] for row in cursor.fetchall()
        ]

        cursor.close()
        return reaction

    except Exception as e:
        logger.error(f"Error retrieving reaction info: {e}")
        cursor.close()
        return None


def print_reaction_summary(reaction_info: dict):
    """Print a formatted summary of reaction information"""
    logger.info("\n" + "=" * 80)
    logger.info("REACTION INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Reaction ID:          {reaction_info['reaction_id']}")
    logger.info(f"Template Hash:        {reaction_info['template_hash']}")
    logger.info(f"Dataset:              {reaction_info['dataset']}")
    logger.info(f"Derive Version:       {reaction_info['derive_version']}")
    logger.info(f"Created At:           {reaction_info['created_at']}")
    logger.info()
    logger.info(f"Mapped Reaction:      {reaction_info['mapped_rxn']}")
    logger.info(f"Product SMILES:       {reaction_info['product_smiles']}")
    logger.info(f"Reactant SMILES:      {reaction_info['reactant_smiles']}")
    logger.info()
    logger.info(f"Retro SMARTS:         {reaction_info['retro_smarts_template']}")
    logger.info(f"Canonical SMARTS:     {reaction_info['canonical_smarts_template']}")
    logger.info()
    logger.info(
        f"Bonds Formed:         {', '.join(reaction_info['bonds_formed']) if reaction_info['bonds_formed'] else 'None'}"
    )
    logger.info(
        f"Bonds Broken:         {', '.join(reaction_info['bonds_broken']) if reaction_info['bonds_broken'] else 'None'}"
    )
    logger.info(
        f"Bonds Order Changed:  {', '.join(reaction_info['bonds_order_changed']) if reaction_info['bonds_order_changed'] else 'None'}"
    )
    logger.info()
    logger.info(
        f"FGs Formed:           {', '.join(reaction_info['functional_groups_formed']) if reaction_info['functional_groups_formed'] else 'None'}"
    )
    logger.info(
        f"FGs Broken:           {', '.join(reaction_info['functional_groups_broken']) if reaction_info['functional_groups_broken'] else 'None'}"
    )
    logger.info("=" * 80)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def add_reactions_from_list(
    reactions: list[str], interactive: bool = False
) -> list[dict[str, Any]]:
    """
    Add a list of reactions to the database.

    Args:
        reactions: List of atom-mapped reaction SMILES
        interactive: If True, prompt for metadata for each reaction

    Returns:
        List of dicts with results for each reaction
    """
    if not reactions:
        logger.warning("No reactions provided!")
        return []

    logger.info("=" * 80)
    logger.info(f"ADDING {len(reactions)} CUSTOM REACTIONS TO DATABASE")
    logger.info("=" * 80)

    # Test database connection
    if not test_database_connection():
        logger.error("Database connection failed. Aborting.")
        return []

    conn = get_production_connection()
    results = []

    for idx, mapped_rxn in enumerate(reactions, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"PROCESSING REACTION {idx}/{len(reactions)}")
        logger.info("=" * 80)
        logger.info(f"Reaction SMILES: {mapped_rxn}")

        # Prepare metadata
        metadata = {}
        if interactive:
            name = input(
                f"Enter name for reaction {idx} (optional, press Enter to skip): "
            ).strip()
            if name:
                metadata["name"] = name
            source = input(
                f"Enter source for reaction {idx} (optional, press Enter to skip): "
            ).strip()
            if source:
                metadata["source_row_id"] = source

        # Process chemistry
        result = process_single_reaction(mapped_rxn, metadata)

        if not result["success"]:
            logger.error(f"✗ Failed to process reaction {idx}: {result.get('error')}")
            results.append(
                {
                    "index": idx,
                    "mapped_rxn": mapped_rxn,
                    "success": False,
                    "error": result.get("error"),
                    "reaction_id": None,
                    "template_hash": None,
                }
            )
            continue

        # Insert or update in database
        db_result = insert_reaction_to_database(conn, result)

        if db_result:
            reaction_id = db_result["reaction_id"]
            is_updated = db_result["is_updated"]

            if is_updated:
                logger.info(
                    f"✓ Updated existing reaction with reaction_id: {reaction_id}"
                )
                results.append(
                    {
                        "index": idx,
                        "mapped_rxn": mapped_rxn,
                        "success": True,
                        "is_updated": True,
                        "reaction_id": reaction_id,
                        "template_hash": result["templates"]["template_hash"],
                        "message": "Existing entry updated with new reaction data",
                    }
                )
            else:
                logger.info(f"✓ Successfully added reaction {idx}")
                results.append(
                    {
                        "index": idx,
                        "mapped_rxn": mapped_rxn,
                        "success": True,
                        "is_updated": False,
                        "reaction_id": reaction_id,
                        "template_hash": result["templates"]["template_hash"],
                    }
                )
        else:
            logger.error(f"✗ Failed to insert/update reaction {idx} in database")
            results.append(
                {
                    "index": idx,
                    "mapped_rxn": mapped_rxn,
                    "success": False,
                    "is_updated": False,
                    "error": "Database operation failed",
                    "reaction_id": None,
                    "template_hash": result["templates"].get("template_hash"),
                }
            )

    conn.close()

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    new_entries = [
        r for r in results if r["success"] and not r.get("is_updated", False)
    ]
    updated = [r for r in results if r["success"] and r.get("is_updated", False)]
    failed = [r for r in results if not r["success"]]

    logger.info(f"Total reactions processed: {len(results)}")
    logger.info(f"New entries added:         {len(new_entries)}")
    logger.info(f"Existing entries updated:  {len(updated)}")
    logger.info(f"Failed:                    {len(failed)}")

    if new_entries:
        logger.info("")
        logger.info("New reactions added:")
        for r in new_entries:
            logger.info(
                f"  Reaction {r['index']}: reaction_id={r['reaction_id']}, template_hash={r['template_hash'][:16]}..."
            )

    if updated:
        logger.info("")
        logger.info("Existing reactions updated:")
        for r in updated:
            logger.info(
                f"  Reaction {r['index']}: reaction_id={r['reaction_id']}, template_hash={r['template_hash'][:16]}..."
            )

    if failed:
        logger.info("")
        logger.info("Failed reactions:")
        for r in failed:
            logger.info(f"  Reaction {r['index']}: {r.get('error', 'Unknown error')}")

    logger.info("=" * 80)

    return results


def interactive_mode():
    """Interactive mode to input reactions one by one"""
    logger.info("=" * 80)
    logger.info("INTERACTIVE MODE")
    logger.info("=" * 80)
    logger.info("Enter reaction SMILES (atom-mapped) one per line.")
    logger.info("Press Enter on an empty line when done.")
    logger.info("")

    reactions = []
    while True:
        rxn = input(
            f"Reaction {len(reactions) + 1} (or press Enter to finish): "
        ).strip()
        if not rxn:
            break
        reactions.append(rxn)

    if not reactions:
        logger.info("No reactions entered.")
        return None

    return add_reactions_from_list(reactions, interactive=True)


def main():
    """Main entry point"""
    logger.info("\n" + "=" * 80)
    logger.info("ADD CUSTOM REACTIONS TO PRODUCTION DATABASE")
    logger.info("=" * 80)
    logger.info(" ")

    # Check if we have reactions in CUSTOM_REACTIONS list
    if CUSTOM_REACTIONS and CUSTOM_REACTIONS[0] != "":
        logger.info(
            f"Found {len(CUSTOM_REACTIONS)} reaction(s) in CUSTOM_REACTIONS list."
        )
        choice = input("Process these reactions? (y/n): ").strip().lower()

        if choice == "y":
            results = add_reactions_from_list(CUSTOM_REACTIONS)

            # Optionally retrieve and print full info for each added reaction
            if results:
                view_details = (
                    input("\nView detailed information for added reactions? (y/n): ")
                    .strip()
                    .lower()
                )
                if view_details == "y":
                    conn = get_production_connection()
                    for r in results:
                        if r["success"]:
                            info = get_reaction_info(conn, r["reaction_id"])
                            if info:
                                print_reaction_summary(info)
                    conn.close()
            return

    # Otherwise, use interactive mode
    logger.info("No reactions found in CUSTOM_REACTIONS list.")
    choice = input("Enter reactions interactively? (y/n): ").strip().lower()

    if choice == "y":
        results = interactive_mode()

        # Optionally retrieve and print full info
        if results:
            view_details = (
                input("\nView detailed information for added reactions? (y/n): ")
                .strip()
                .lower()
            )
            if view_details == "y":
                conn = get_production_connection()
                for r in results:
                    if r["success"]:
                        info = get_reaction_info(conn, r["reaction_id"])
                        if info:
                            print_reaction_summary(info)
                conn.close()
    else:
        logger.info("Exiting.")


if __name__ == "__main__":
    main()
