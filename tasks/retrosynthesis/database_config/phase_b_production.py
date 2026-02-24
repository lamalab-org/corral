"""
Phase B — Production Database Builder

Derives chemistry artifacts from Phase A staging data and populates production schema:
- Reaction templates with retro/forward SMARTS
- RDKit query molecules and pattern fingerprints
- Bonds (formed/broken/order_changed)
- Functional groups (formed/broken)
- Molecules cache for repeated queries

Database: reactions_production_db
Depends on: Phase A staging database (reactions_raw_db)
"""

import sys
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import psycopg2
from loguru import logger
from phase_b_chemistry_utils import get_functional_groups, obtain_bonds
from psycopg2.extras import RealDictCursor, execute_batch, execute_values

# Import your chemistry tools
from rxnutils.chem.reaction import ChemicalReaction
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from retrosynthesis.config import get_db_config, get_staging_db_config

# Processing parameters
BATCH_SIZE = 10000  # Rows per batch
DERIVE_VERSION = "v1"  # Version tag for this run
NUM_WORKERS = None  # Number of parallel workers for chemistry processing (None = use all available CPUs)

# RDKit function names (adjust if your cartridge version differs)
# To verify: run \df *qmol* and \df *fp* in psql
RDKIT_QMOL_FUNC = "qmol_from_smarts"  # Creates query molecule from SMARTS
RDKIT_FP_FUNC = "rdkit_fp"  # RDKit fingerprint for molecules (alternatives: morganbv_fp, layered_fp, maccs_fp)
RDKIT_SUBSET_OP = "%<="  # Substructure match operator


class ErrorStage(Enum):
    """Enumeration of all possible ETL error stages for precise tracking"""

    TEMPLATE_GENERATION = (
        "template_generation"  # ChemicalReaction.generate_reaction_template() failed
    )
    QMOL_PRODUCT = "qmol_product"  # qmol_from_smarts() failed for product
    QMOL_REACTANT = "qmol_reactant"  # qmol_from_smarts() failed for reactant
    PATTERN_FP_PRODUCT = "pattern_fp_product"  # pattern_fp() failed for product
    PATTERN_FP_REACTANT = "pattern_fp_reactant"  # pattern_fp() failed for reactant
    BONDS = "bonds"  # obtain_bonds() failed
    FGS = "fgs"  # get_functional_groups() failed
    PREPARATION = "preparation"  # General preparation error

    def __str__(self):
        return self.value


class ErrorTracker:
    """Track ETL errors in memory with aggregation capabilities"""

    def __init__(self, derive_version: str):
        self.derive_version = derive_version
        # Count by error stage
        self.error_counts = defaultdict(int)
        # Store sample errors for debugging (limit to avoid memory issues)
        self.error_samples = defaultdict(list)
        self.max_samples_per_stage = 10

    def log_error(
        self,
        staging_id: int,
        raw_hash: str,
        mapped_rxn: str,
        stage: ErrorStage,
        message: str,
    ):
        """Log an error to in-memory tracker and file log"""
        # Increment counter
        self.error_counts[stage] += 1

        # Store sample if we haven't hit the limit
        if len(self.error_samples[stage]) < self.max_samples_per_stage:
            self.error_samples[stage].append(
                {
                    "staging_id": staging_id,
                    "raw_hash": raw_hash,
                    "mapped_rxn": mapped_rxn[:100] + "..."
                    if len(mapped_rxn) > 100
                    else mapped_rxn,
                    "message": str(message),
                }
            )

        # Log to file with full context
        logger.warning(
            f"Error [{stage.value}] staging_id={staging_id} hash={raw_hash[:8]}: {message}"
        )

    def get_summary(self) -> dict[str, int]:
        """Get summary of errors by stage"""
        return dict(self.error_counts)

    def get_total_errors(self) -> int:
        """Get total error count across all stages"""
        return sum(self.error_counts.values())

    def print_report(self):
        """Print detailed error report"""
        logger.info("=" * 80)
        logger.info("ERROR REPORT")
        logger.info("=" * 80)
        logger.info(f"Derive version: {self.derive_version}")
        logger.info(f"Total errors: {self.get_total_errors()}")
        logger.info("")

        if self.error_counts:
            logger.info("Errors by stage:")
            for stage in ErrorStage:
                count = self.error_counts.get(stage, 0)
                if count > 0:
                    logger.info(f"  {stage.value:25s}: {count:6d}")

            logger.info("")
            logger.info("Sample errors (first 10 per stage):")
            for stage, samples in self.error_samples.items():
                if samples:
                    logger.info(f"\n  [{stage.value}]")
                    for i, sample in enumerate(
                        samples[:5], 1
                    ):  # Show only first 5 in summary
                        logger.info(
                            f"    {i}. staging_id={sample['staging_id']} - {sample['message'][:80]}"
                        )
        else:
            logger.info("No errors encountered! ✓")

        logger.info("=" * 80)


def get_staging_connection():
    """Connect to Phase A staging database"""
    conn = psycopg2.connect(**get_staging_db_config())
    conn.set_client_encoding("UTF8")
    return conn


def get_production_connection():
    """Connect to Phase B production database"""
    conn = psycopg2.connect(**get_db_config())
    conn.set_client_encoding("UTF8")
    return conn


def create_production_database():
    """Create the production database if it doesn't exist"""
    # Get database configuration
    db_config = get_db_config()

    # Connect to default postgres database to create new DB
    conn = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database="postgres",
        user=db_config["user"],
        password=db_config["password"],
    )
    conn.set_client_encoding("UTF8")
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_config["database"],),
        )
        exists = cursor.fetchone()

        if not exists:
            logger.info(f"Creating database {db_config['database']}...")
            cursor.execute(f"CREATE DATABASE {db_config['database']}")
            logger.info("Database created successfully")
        else:
            logger.info(f"Database {db_config['database']} already exists")

    finally:
        cursor.close()
        conn.close()


def create_production_schema():
    """Create all production tables and indexes"""
    conn = get_production_connection()
    cursor = conn.cursor()

    try:
        logger.info("Creating RDKit extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS rdkit;")

        # Validate RDKit functions are available
        logger.info("Validating RDKit functions...")
        try:
            cursor.execute(f"SELECT {RDKIT_QMOL_FUNC}('c1ccccc1');")
            cursor.execute(f"SELECT {RDKIT_FP_FUNC}(mol_from_smiles('c1ccccc1'));")
            logger.info(
                f"✓ RDKit functions validated: {RDKIT_QMOL_FUNC}(), {RDKIT_FP_FUNC}()"
            )
        except Exception as e:
            logger.error(f"✗ RDKit function validation failed: {e}")
            logger.error(
                f"  Check that {RDKIT_QMOL_FUNC} and {RDKIT_FP_FUNC} exist in your RDKit cartridge"
            )
            logger.error(
                "  Run: \\df *qmol* and \\df *fp* in psql to see available functions"
            )
            raise

        logger.info("Creating production schema...")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reactions (
                reaction_id                 SERIAL PRIMARY KEY,
                template_hash               TEXT NOT NULL UNIQUE,

                -- Templates (SMARTS)
                retro_smarts_template       TEXT NOT NULL,
                canonical_smarts_template   TEXT NOT NULL,

                -- Representative example reaction
                mapped_rxn                  TEXT NOT NULL,
                product_smiles              TEXT,
                reactant_smiles             TEXT,

                -- Provenance (from the representative example)
                dataset                     TEXT,
                source_row_id               TEXT,
                staging_id                  BIGINT,

                -- RDKit chemistry columns
                product_qmol                qmol,
                reactant_qmol               qmol,
                product_pattern_fp          bfp,
                reactant_pattern_fp         bfp,

                -- Metadata
                derive_version              TEXT,
                created_at                  TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bonds (
                bond_id         SERIAL PRIMARY KEY,
                bond_label      TEXT UNIQUE NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reaction_bonds_formed (
                reaction_id     INTEGER NOT NULL REFERENCES reactions(reaction_id) ON DELETE CASCADE,
                bond_id         INTEGER NOT NULL REFERENCES bonds(bond_id) ON DELETE CASCADE,
                PRIMARY KEY (reaction_id, bond_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reaction_bonds_broken (
                reaction_id     INTEGER NOT NULL REFERENCES reactions(reaction_id) ON DELETE CASCADE,
                bond_id         INTEGER NOT NULL REFERENCES bonds(bond_id) ON DELETE CASCADE,
                PRIMARY KEY (reaction_id, bond_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reaction_bonds_order_changed (
                reaction_id     INTEGER NOT NULL REFERENCES reactions(reaction_id) ON DELETE CASCADE,
                bond_id         INTEGER NOT NULL REFERENCES bonds(bond_id) ON DELETE CASCADE,
                PRIMARY KEY (reaction_id, bond_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS functional_groups (
                functional_group_id     SERIAL PRIMARY KEY,
                functional_group_key    TEXT UNIQUE NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reaction_functional_groups_formed (
                reaction_id             INTEGER NOT NULL REFERENCES reactions(reaction_id) ON DELETE CASCADE,
                functional_group_id     INTEGER NOT NULL REFERENCES functional_groups(functional_group_id) ON DELETE CASCADE,
                PRIMARY KEY (reaction_id, functional_group_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reaction_functional_groups_broken (
                reaction_id             INTEGER NOT NULL REFERENCES reactions(reaction_id) ON DELETE CASCADE,
                functional_group_id     INTEGER NOT NULL REFERENCES functional_groups(functional_group_id) ON DELETE CASCADE,
                PRIMARY KEY (reaction_id, functional_group_id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS molecules (
                molecule_id     SERIAL PRIMARY KEY,
                name            TEXT,
                smiles          TEXT UNIQUE NOT NULL,
                mol             mol,
                pattern_fp      bfp,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """)

        logger.info("Creating indexes...")

        # Reactions indexes
        # Note: template_hash is already UNIQUE, so it has an automatic index

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reactions_product_fp
            ON reactions USING gist(product_pattern_fp);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reactions_reactant_fp
            ON reactions USING gist(reactant_pattern_fp);
        """)

        # Reactions provenance indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reactions_dataset
            ON reactions(dataset);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reactions_staging_id
            ON reactions(staging_id);
        """)

        # Bonds table index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_label
            ON bonds(bond_label);
        """)

        # Bonds junction tables indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_formed_reaction
            ON reaction_bonds_formed(reaction_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_formed_bond
            ON reaction_bonds_formed(bond_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_broken_reaction
            ON reaction_bonds_broken(reaction_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_broken_bond
            ON reaction_bonds_broken(bond_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_order_changed_reaction
            ON reaction_bonds_order_changed(reaction_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bonds_order_changed_bond
            ON reaction_bonds_order_changed(bond_id);
        """)

        # FG indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fg_key
            ON functional_groups(functional_group_key);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rfg_formed_reaction
            ON reaction_functional_groups_formed(reaction_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rfg_formed_fg
            ON reaction_functional_groups_formed(functional_group_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rfg_broken_reaction
            ON reaction_functional_groups_broken(reaction_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rfg_broken_fg
            ON reaction_functional_groups_broken(functional_group_id);
        """)

        # Molecules indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_molecules_mol_gist
            ON molecules USING gist(mol);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_molecules_pattern_fp
            ON molecules USING gist(pattern_fp);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_molecules_smiles
            ON molecules(smiles);
        """)

        conn.commit()
        logger.info("Schema created successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating schema: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def generate_templates(mapped_rxn: str) -> dict[str, str] | None:
    """
    Generate retro and forward SMARTS templates from mapped reaction.

    Notes:
    - Uses rxnutils.ChemicalReaction which employs rdchiral template extraction
    - retro_smarts: Product-side SMARTS for retrosynthetic queries
    - The library validates unmapped product atoms don't exceed limits
    - Multi-product reactions are handled via SMARTS concatenation

    Args:
        mapped_rxn (str): Mapped reaction SMILES.

    Returns:
        dict[str, str] | None: Dict with keys:
        - 'retro_smarts': Product-side SMARTS (for retro applicability)
        - 'canonical_smarts': Canonical template
        - 'template_hash': Hash computed from rxn.retro_template.hash_from_bits()
        - 'reactant_smarts': Reactant-side SMARTS (optional)
        - 'product_smiles': Product SMILES (for fingerprint generation)
        - 'reactant_smiles': Reactant SMILES (for fingerprint generation)
    """
    try:
        rxn = ChemicalReaction(mapped_rxn)
        rxn.generate_reaction_template()

        # Compute hash using the retro_template method
        template_hash = rxn.retro_template.hash_from_bits()

        # Extract SMILES for fingerprint generation
        # Join multiple reactants/products with '.'
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

        # Check for single quotes in SMARTS (should be handled by parameterization)
        if "'" in templates["retro_smarts"]:
            logger.debug(
                f"SMARTS contains single quotes (OK with params): {templates['retro_smarts'][:50]}..."
            )

        return templates
    except Exception as e:
        # Include reaction SMILES in error log to identify problematic reactions
        rxn_preview = mapped_rxn[:100] + "..." if len(mapped_rxn) > 100 else mapped_rxn
        logger.warning(f"Template generation failed: {e} | Reaction: {rxn_preview}")
        return None


def process_single_reaction_chemistry(row_data: dict[str, Any]) -> dict[str, Any]:
    """
    Process chemistry for a single reaction (parallelizable function).

    This function extracts templates, bonds, and functional groups for one reaction.
    It's designed to be called in parallel using multiprocessing.

    Args:
        row_data (dict[str, Any]): Dict containing 'staging_id', 'mapped_rxn', 'raw_hash', etc.

    Returns:
        dict[str, Any]: Dict with results including:
        - 'success': bool
        - 'staging_id': int
        - 'error_stage': ErrorStage (if failed)
        - 'error_message': str (if failed)
        - 'templates': dict (if successful)
        - 'bonds': dict (if successful)
        - 'fgs': dict (if successful)
    """
    staging_id = row_data["staging_id"]
    mapped_rxn = row_data["mapped_rxn"]
    raw_hash = row_data["raw_hash"]

    result = {
        "staging_id": staging_id,
        "raw_hash": raw_hash,
        "mapped_rxn": mapped_rxn,
        "dataset": row_data.get("dataset"),
        "id_in_csv": row_data.get("id_in_csv"),
        "success": False,
    }

    try:
        # Step 1: Generate templates
        templates = generate_templates(mapped_rxn)
        if not templates:
            result["error_stage"] = ErrorStage.TEMPLATE_GENERATION
            result["error_message"] = "Failed to generate templates"
            return result

        result["templates"] = templates

        # Step 2: Extract bonds
        try:
            bonds = obtain_bonds(mapped_rxn)
            result["bonds"] = bonds
        except Exception as e:
            result["error_stage"] = ErrorStage.BONDS
            result["error_message"] = f"obtain_bonds() failed: {e}"
            return result

        # Step 3: Extract functional groups
        try:
            fgs = get_functional_groups(mapped_rxn)
            result["fgs"] = fgs
        except Exception as e:
            result["error_stage"] = ErrorStage.FGS
            result["error_message"] = f"get_functional_groups() failed: {e}"
            return result

        result["success"] = True
        return result

    except Exception as e:
        result["error_stage"] = ErrorStage.PREPARATION
        result["error_message"] = str(e)
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


def process_batch(
    production_conn: Any,
    batch: list[dict[str, Any]],
    error_tracker: ErrorTracker,
    num_workers: int | None = None,
) -> dict[str, int]:
    """Process a batch of staging reactions using bulk insert with execute_values

    Args:
        production_conn (Any): Connection to production database
        batch (list[dict[str, Any]]): List of reaction data dictionaries
        error_tracker (ErrorTracker): ErrorTracker instance for logging errors
        num_workers (int | None): Number of parallel workers (None = use all CPUs)

    Returns:
        dict[str, int]: Statistics about processing (processed, inserted, skipped_duplicate, skipped_error)
    """
    stats = {"processed": 0, "inserted": 0, "skipped_duplicate": 0, "skipped_error": 0}

    prod_cursor = production_conn.cursor()

    # Phase 1: Process chemistry in parallel
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    logger.info(
        f"Processing batch of {len(batch)} reactions using {num_workers} parallel workers..."
    )

    # Process all reactions in parallel
    chemistry_results = []
    if num_workers > 1 and len(batch) > 1:
        # Use multiprocessing for parallel chemistry processing
        with Pool(processes=num_workers) as pool:
            chemistry_results = pool.map(process_single_reaction_chemistry, batch)
    else:
        # Fallback to sequential processing if only 1 worker or 1 reaction
        chemistry_results = [process_single_reaction_chemistry(row) for row in batch]

    # Phase 2: Prepare data for bulk insert from successful chemistry results
    reactions_to_insert = []  # List of tuples for bulk insert
    reaction_metadata = []  # Metadata for post-insert processing (bonds, FGs)

    for result in chemistry_results:
        stats["processed"] += 1

        if not result["success"]:
            # Log error and skip
            error_tracker.log_error(
                result["staging_id"],
                result["raw_hash"],
                result["mapped_rxn"],
                result["error_stage"],
                result["error_message"],
            )
            stats["skipped_error"] += 1
            continue

        # Extract data from successful result
        staging_id = result["staging_id"]
        templates = result["templates"]
        template_hash = templates["template_hash"]
        bonds = result["bonds"]
        fgs = result["fgs"]

        # Prepare values tuple for bulk insert into reactions table
        # Each template gets ONE representative example (the first one encountered)
        # IMPORTANT: SMARTS/SMILES are passed as parameters (not f-strings) to prevent SQL injection
        # Note: Each CASE WHEN needs the value twice (once for NULL check, once for the function call)
        reaction_values = (
            template_hash,  # 1. template_hash
            templates["retro_smarts"],  # 2. retro_smarts_template
            templates.get("canonical_smarts"),  # 3. canonical_smarts_template
            result["mapped_rxn"],  # 4. mapped_rxn (representative example)
            templates.get("product_smiles"),  # 5. product_smiles
            templates.get("reactant_smiles"),  # 6. reactant_smiles
            result.get("dataset"),  # 7. dataset
            result.get("id_in_csv"),  # 8. source_row_id
            staging_id,  # 9. staging_id
            templates["retro_smarts"],  # 10. retro_smarts for splitting to product side
            templates[
                "retro_smarts"
            ],  # 11. retro_smarts for splitting to reactant side
            templates.get(
                "product_smiles"
            ),  # 12. CASE WHEN check for product_pattern_fp
            templates.get("product_smiles"),  # 13. rdkit_fp(mol_from_smiles(product))
            templates.get(
                "reactant_smiles"
            ),  # 14. CASE WHEN check for reactant_pattern_fp
            templates.get("reactant_smiles"),  # 15. rdkit_fp(mol_from_smiles(reactant))
            DERIVE_VERSION,  # 16. derive_version
        )

        reactions_to_insert.append(reaction_values)
        reaction_metadata.append(
            {
                "template_hash": template_hash,
                "staging_id": staging_id,
                "bonds": bonds,
                "fgs": fgs,
            }
        )

    # Phase 3: Bulk insert unique templates into reactions table
    # Note: RDKit errors (qmol_from_smarts, pattern_fp) are caught by PostgreSQL
    # and will cause the entire batch to fail.
    if reactions_to_insert:
        try:
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
                ) VALUES %s
                ON CONFLICT (template_hash) DO NOTHING
                RETURNING reaction_id, template_hash
            """

            # Template for execute_values - this defines what goes in each VALUES (...)
            # execute_values will replace %s in the INSERT with this template repeated for each row
            values_template = """(
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                qmol_from_smarts(split_part(%s, '>>', 1)::cstring),
                qmol_from_smarts(split_part(%s, '>>', 2)::cstring),
                CASE WHEN %s IS NOT NULL THEN rdkit_fp(mol_from_smiles(%s)) ELSE NULL END,
                CASE WHEN %s IS NOT NULL THEN rdkit_fp(mol_from_smiles(%s)) ELSE NULL END,
                %s
            )"""

            # Use page_size equal to batch size to insert all at once
            inserted_rows = execute_values(
                prod_cursor,
                insert_sql,
                reactions_to_insert,
                template=values_template,
                page_size=len(reactions_to_insert),
                fetch=True,
            )

            # Build mapping of template_hash -> reaction_id for inserted reactions
            reaction_id_map = (
                {row[1]: row[0] for row in inserted_rows} if inserted_rows else {}
            )

            stats["inserted"] = len(reaction_id_map)
            stats["skipped_duplicate"] = len(reactions_to_insert) - stats["inserted"]

            logger.info(
                f"Bulk insert (reactions): {stats['inserted']} inserted, {stats['skipped_duplicate']} duplicates"
            )

        except Exception as e:
            logger.error(f"Bulk insert (reactions) failed: {e}")
            production_conn.rollback()
            raise

    # Phase 4: Insert bonds and functional groups for successfully inserted templates
    template_hashes = [m["template_hash"] for m in reaction_metadata]
    if template_hashes:
        # Fetch reaction_ids for all template_hashes in this batch
        prod_cursor.execute(
            "SELECT template_hash, reaction_id FROM reactions WHERE template_hash = ANY(%s)",
            (template_hashes,),
        )
        reaction_id_map = {row[0]: row[1] for row in prod_cursor.fetchall()}

    # Track unique templates processed for bonds/FGs (avoid duplicates within batch)
    processed_templates = set()

    for metadata in reaction_metadata:
        template_hash = metadata["template_hash"]

        # Skip if we've already processed this template in this batch
        if template_hash in processed_templates:
            continue

        # Check if this template exists in the database
        if template_hash not in reaction_id_map:
            continue

        processed_templates.add(template_hash)
        reaction_id = reaction_id_map[template_hash]
        bonds = metadata["bonds"]
        fgs = metadata["fgs"]

        try:
            # Insert bonds formed
            if bonds["formed"]:
                # Upsert all bonds first and collect (reaction_id, bond_id) pairs
                bond_pairs = []
                for bond in bonds["formed"]:
                    bond_id = upsert_bond(prod_cursor, bond)
                    bond_pairs.append((reaction_id, bond_id))
                # Batch insert all relationships
                execute_batch(
                    prod_cursor,
                    """
                    INSERT INTO reaction_bonds_formed (reaction_id, bond_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, bond_id) DO NOTHING
                """,
                    bond_pairs,
                )

            # Insert bonds broken
            if bonds["broken"]:
                bond_pairs = []
                for bond in bonds["broken"]:
                    bond_id = upsert_bond(prod_cursor, bond)
                    bond_pairs.append((reaction_id, bond_id))
                execute_batch(
                    prod_cursor,
                    """
                    INSERT INTO reaction_bonds_broken (reaction_id, bond_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, bond_id) DO NOTHING
                """,
                    bond_pairs,
                )

            # Insert bonds with order changes
            if bonds["order_changed"]:
                bond_pairs = []
                for bond in bonds["order_changed"]:
                    bond_id = upsert_bond(prod_cursor, bond)
                    bond_pairs.append((reaction_id, bond_id))
                execute_batch(
                    prod_cursor,
                    """
                    INSERT INTO reaction_bonds_order_changed (reaction_id, bond_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, bond_id) DO NOTHING
                """,
                    bond_pairs,
                )

            # Insert functional groups formed
            if fgs["formed"]:
                fg_pairs = []
                for fg_key in fgs["formed"]:
                    fg_id = upsert_functional_group(prod_cursor, fg_key)
                    fg_pairs.append((reaction_id, fg_id))
                execute_batch(
                    prod_cursor,
                    """
                    INSERT INTO reaction_functional_groups_formed (reaction_id, functional_group_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, functional_group_id) DO NOTHING
                """,
                    fg_pairs,
                )

            # Insert functional groups broken
            if fgs["broken"]:
                fg_pairs = []
                for fg_key in fgs["broken"]:
                    fg_id = upsert_functional_group(prod_cursor, fg_key)
                    fg_pairs.append((reaction_id, fg_id))
                execute_batch(
                    prod_cursor,
                    """
                    INSERT INTO reaction_functional_groups_broken (reaction_id, functional_group_id)
                    VALUES (%s, %s)
                    ON CONFLICT (reaction_id, functional_group_id) DO NOTHING
                """,
                    fg_pairs,
                )

        except Exception as e:
            logger.error(
                f"Error processing bonds/FGs for reaction_id={reaction_id}: {e}"
            )
            # Note: reaction was already inserted, so we don't increment skipped_error
            # The reaction is still counted in stats['inserted']
            continue

    production_conn.commit()
    prod_cursor.close()

    return stats


def run_etl_pipeline():
    """Main ETL pipeline - process all staging reactions"""
    logger.info("=" * 80)
    logger.info("PHASE B ETL PIPELINE START")
    logger.info("=" * 80)

    # Determine number of workers
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else cpu_count()
    logger.info(f"Using {num_workers} parallel workers for chemistry processing")

    # Initialize error tracker
    error_tracker = ErrorTracker(DERIVE_VERSION)

    staging_conn = get_staging_connection()
    production_conn = get_production_connection()

    try:
        # Get total count
        staging_cursor = staging_conn.cursor()
        staging_cursor.execute("SELECT COUNT(*) FROM staging_reactions")
        total_count = staging_cursor.fetchone()[0]
        logger.info(f"Total staging reactions to process: {total_count}")

        # Create dict cursor for fetching data
        staging_cursor = staging_conn.cursor(cursor_factory=RealDictCursor)

        # Process in batches
        total_stats = {
            "processed": 0,
            "inserted": 0,
            "skipped_duplicate": 0,
            "skipped_error": 0,
        }

        offset = 0
        with tqdm(total=total_count, desc="Processing reactions") as pbar:
            while offset < total_count:
                # Fetch batch from staging
                staging_cursor.execute(
                    """
                    SELECT
                        staging_id,
                        raw_hash,
                        dataset,
                        id_in_csv,
                        mapped_rxn
                    FROM staging_reactions
                    ORDER BY staging_id
                    LIMIT %s OFFSET %s
                """,
                    (BATCH_SIZE, offset),
                )

                batch = [dict(row) for row in staging_cursor.fetchall()]
                if not batch:
                    break

                # Process batch with parallel chemistry processing
                batch_stats = process_batch(
                    production_conn, batch, error_tracker, num_workers
                )

                # Update totals
                for key in total_stats:
                    total_stats[key] += batch_stats[key]

                pbar.update(len(batch))
                offset += BATCH_SIZE

        # Final stats
        logger.info("=" * 80)
        logger.info("PHASE B ETL PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total processed:       {total_stats['processed']}")
        logger.info(f"Successfully inserted: {total_stats['inserted']}")
        logger.info(f"Skipped (duplicate):   {total_stats['skipped_duplicate']}")
        logger.info(f"Skipped (error):       {total_stats['skipped_error']}")

        # Query final counts
        prod_cursor = production_conn.cursor()
        prod_cursor.execute("SELECT COUNT(*) FROM reactions")
        reaction_count = prod_cursor.fetchone()[0]

        prod_cursor.execute("SELECT COUNT(*) FROM functional_groups")
        fg_count = prod_cursor.fetchone()[0]

        logger.info("=" * 80)
        logger.info("Production database stats:")
        logger.info(f"  - Unique templates (with examples): {reaction_count}")
        logger.info(f"  - Functional groups:                {fg_count}")
        logger.info("=" * 80)

        # Print error report
        error_tracker.print_report()

    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise
    finally:
        staging_conn.close()
        production_conn.close()


def validate_setup():
    """B5: Run sanity checks on the production database"""
    logger.info("Running validation checks...")

    conn = get_production_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Check 1: Sample reaction exists
        cursor.execute("SELECT COUNT(*) as count FROM reactions")
        count = cursor.fetchone()["count"]
        assert count > 0, "No reactions in production database!"
        logger.info(f"✓ Found {count} unique templates")

        # Check 2: Chemistry columns populated
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM reactions
            WHERE product_qmol IS NOT NULL AND product_pattern_fp IS NOT NULL
        """)
        chem_count = cursor.fetchone()["count"]
        logger.info(
            f"✓ {chem_count}/{count} reactions have chemistry columns populated"
        )

        # Check 3: Verify mapped_rxn and provenance fields populated
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM reactions
            WHERE mapped_rxn IS NOT NULL AND dataset IS NOT NULL
        """)
        prov_count = cursor.fetchone()["count"]
        logger.info(
            f"✓ Provenance: {prov_count}/{count} reactions have example mapped_rxn and dataset"
        )

        # Check 5: Bonds tables populated
        cursor.execute("SELECT COUNT(*) as count FROM reaction_bonds_formed")
        formed_count = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM reaction_bonds_broken")
        broken_count = cursor.fetchone()["count"]
        logger.info(f"✓ Bonds: {formed_count} formed, {broken_count} broken")

        # Check 6: FG tables populated
        cursor.execute("SELECT COUNT(*) as count FROM functional_groups")
        fg_count = cursor.fetchone()["count"]
        cursor.execute(
            "SELECT COUNT(*) as count FROM reaction_functional_groups_formed"
        )
        rfg_count = cursor.fetchone()["count"]
        logger.info(
            f"✓ Functional groups: {fg_count} unique, {rfg_count} reaction-FG associations"
        )

        # Check 7: Sample substructure query (if reactions exist)
        if count > 0:
            cursor.execute("""
                SELECT reaction_id, retro_smarts_template
                FROM reactions
                WHERE product_qmol IS NOT NULL
                LIMIT 1
            """)
            sample = cursor.fetchone()
            if sample:
                logger.info(
                    f"✓ Sample reaction {sample['reaction_id']} has valid query molecule"
                )

        logger.info("All validation checks passed! ✓")

    except AssertionError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    """Main execution flow"""
    try:
        logger.info("Phase B - Production Database Builder")
        logger.info(f"Derive version: {DERIVE_VERSION}")
        logger.info(f"Batch size: {BATCH_SIZE}")

        # Step 1: Create database
        logger.info("\n[1/4] Creating production database...")
        create_production_database()

        # Step 2: Create schema
        logger.info("\n[2/4] Creating production schema...")
        create_production_schema()

        # Step 3: Run ETL pipeline
        logger.info("\n[3/4] Running ETL pipeline...")
        run_etl_pipeline()

        # Step 4: Validate
        logger.info("\n[4/4] Running validation...")
        validate_setup()

        logger.info("\n" + "=" * 80)
        logger.info("PHASE B COMPLETE - Production database ready!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Phase B failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
