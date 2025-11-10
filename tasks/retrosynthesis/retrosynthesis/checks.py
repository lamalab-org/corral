import psycopg2
from loguru import logger
from retrosynthesis.config import get_db_config

# Expected schema - tables and their required columns
EXPECTED_SCHEMA = {
    "reactions": [
        "reaction_id",
        "template_hash",
        "retro_smarts_template",
        "canonical_smarts_template",
        "mapped_rxn",
        "product_smiles",
        "reactant_smiles",
        "dataset",
        "source_row_id",
        "staging_id",
        "product_qmol",
        "reactant_qmol",
        "product_pattern_fp",
        "reactant_pattern_fp",
        "derive_version",
        "created_at",
    ],
    "bonds": [
        "bond_id",
        "bond_label",
    ],
    "reaction_bonds_formed": [
        "reaction_id",
        "bond_id",
    ],
    "reaction_bonds_broken": [
        "reaction_id",
        "bond_id",
    ],
    "reaction_bonds_order_changed": [
        "reaction_id",
        "bond_id",
    ],
    "functional_groups": [
        "functional_group_id",
        "functional_group_key",
    ],
    "reaction_functional_groups_formed": [
        "reaction_id",
        "functional_group_id",
    ],
    "reaction_functional_groups_broken": [
        "reaction_id",
        "functional_group_id",
    ],
    "molecules": [
        "molecule_id",
        "name",
        "smiles",
        "mol",
        "pattern_fp",
        "created_at",
    ],
}


def check_database():
    """
    Check that the production database is accessible and has the expected schema.

    This function performs the following checks:
    1. Database connectivity
    2. RDKit extension availability
    3. Presence of all required tables
    4. Presence of all required columns in each table

    Raises:
        ConnectionError: If unable to connect to the database
        ValueError: If the database schema doesn't match expectations
    """
    conn = None
    try:
        # Get database configuration
        db_config = get_db_config()

        # Check 1: Database connectivity
        logger.info("Checking database connectivity...")
        try:
            conn = psycopg2.connect(**db_config)
            conn.set_client_encoding("UTF8")
            logger.info(
                f"✓ Successfully connected to database '{db_config['database']}'"
            )
        except psycopg2.OperationalError as e:
            error_msg = f"Failed to connect to database '{db_config['database']}': {e}"
            logger.error(f"✗ {error_msg}")
            raise ConnectionError(error_msg) from e

        cursor = conn.cursor()
        # Check 2: RDKit extension
        logger.info("Checking RDKit extension...")
        try:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'rdkit'
                );
            """)
            rdkit_exists = cursor.fetchone()[0]
            if not rdkit_exists:
                error_msg = "RDKit extension is not installed in the database"
                logger.error(f"✗ {error_msg}")
                raise ValueError(error_msg)
            logger.info("✓ RDKit extension is available")
        except Exception as e:
            logger.error(f"✗ Error checking RDKit extension: {e}")
            raise

        # Check 3: Required tables exist
        logger.info("Checking required tables...")
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing_tables = set(EXPECTED_SCHEMA.keys()) - existing_tables
        if missing_tables:
            error_msg = f"Missing required tables: {', '.join(sorted(missing_tables))}"
            logger.error(f"✗ {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"✓ All {len(EXPECTED_SCHEMA)} required tables exist")

        # Check 4: Required columns exist in each table
        logger.info("Checking required columns in each table...")
        schema_errors = []

        for table_name, expected_columns in EXPECTED_SCHEMA.items():
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
            """,
                (table_name,),
            )

            existing_columns = {row[0] for row in cursor.fetchall()}
            missing_columns = set(expected_columns) - existing_columns

            if missing_columns:
                error_msg = f"Table '{table_name}' is missing columns: {', '.join(sorted(missing_columns))}"
                schema_errors.append(error_msg)
                logger.error(f"✗ {error_msg}")
            else:
                logger.info(
                    f"✓ Table '{table_name}' has all {len(expected_columns)} required columns"
                )

        if schema_errors:
            raise ValueError("Schema validation failed:\n" + "\n".join(schema_errors))

        # Check 5: Verify database has data (optional warning)
        cursor.execute("SELECT COUNT(*) FROM reactions")
        reaction_count = cursor.fetchone()[0]
        if reaction_count == 0:
            logger.warning(
                "⚠ Warning: 'reactions' table is empty. The database may not be fully populated."
            )
        else:
            logger.info(f"✓ Database contains {reaction_count} reactions")

        logger.info("Database validation completed successfully!")

    except Exception:
        logger.error("Database validation FAILED!")
        raise

    finally:
        if conn:
            conn.close()
