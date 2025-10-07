"""
Phase A — Staging & Normalization

This script creates a PostgreSQL staging table ('staging_reactions') and loads mapped reaction data
from USPTO and ORD CSV files (uspto_data_mapped.csv, ord_data_mapped.csv) with deduplication and validation.
It performs:
    - Table creation (if not exists)
    - Data normalization and validation
    - Deduplication by reaction hash
    - Batch insertion for efficiency
    - Summary statistics reporting

Requirements:
    pip install psycopg2-binary pandas rdkit
"""

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import psycopg2
from loguru import logger
from psycopg2 import sql
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "reactions_raw_db",
    "user": "postgres",  # Docker default user
    "password": "postgres",  # Docker default password
}

STAGING_TABLE = "staging_reactions"

USPTO_CSV = "uspto_data_mapped.csv"
ORD_CSV = "ord_data_mapped.csv"

BATCH_SIZE = 10000  # Number of rows to insert at once


CREATE_STAGING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS staging_reactions (
    -- Identity
    staging_id BIGSERIAL PRIMARY KEY,
    raw_hash CHAR(64) UNIQUE NOT NULL,  -- Template fingerprint hash for deduplication

    -- Provenance
    source_file TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    id_in_csv TEXT,
    dataset TEXT,

    -- Raw data (preserved as-is from CSV)
    date_raw TEXT,
    reaction_smiles TEXT,
    reaction_smiles_clean TEXT,
    yield_raw TEXT,
    badmolecules INTEGER,
    reactant_size INTEGER,
    product_size INTEGER,
    confidence DOUBLE PRECISION,

    -- Required chemistry
    mapped_rxn TEXT NOT NULL,

    -- Audit
    ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Indexing
    CONSTRAINT chk_line_no_positive CHECK (line_no > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_staging_raw_hash ON staging_reactions(raw_hash);
CREATE INDEX IF NOT EXISTS idx_staging_source ON staging_reactions(source_file, line_no);
CREATE INDEX IF NOT EXISTS idx_staging_dataset ON staging_reactions(dataset);
CREATE INDEX IF NOT EXISTS idx_staging_ingested ON staging_reactions(ingested_at);
"""


def compute_raw_hash(mapped_rxn: str) -> str:
    """
    Compute hash for deduplication using simple string hashing.
    Uses Python's built-in hashlib to generate a SHA256 hash of the mapped reaction SMILES.

    Args:
        mapped_rxn: Mapped reaction SMILES
        dataset: Dataset identifier (not used, kept for API compatibility)
        id_val: ID from CSV (not used, kept for API compatibility)

    Returns:
        Hash string (SHA256 hexdigest)

    Raises:
        Exception: If reaction SMILES is empty
    """

    if not mapped_rxn or not mapped_rxn.strip():
        raise ValueError("Reaction SMILES cannot be empty")

    # Compute SHA256 hash of the mapped reaction SMILES string
    return hashlib.sha256(mapped_rxn.encode("utf-8")).hexdigest()


def normalize_row_data(
    row: dict[str, Any], source_file: str, line_no: int
) -> dict[str, Any] | None:
    """
    Normalize and prepare a CSV row for insertion.

    Args:
        row: Dictionary from CSV DictReader
        source_file: Name of source CSV file
        line_no: Line number in CSV (1-indexed, excluding header)

    Returns:
        Normalized dictionary ready for DB insertion, or None if critical data missing
    """
    # Extract mapped_rxn (required)
    mapped_rxn = row.get("mapped_rxn", "").strip()
    if not mapped_rxn:
        return None

    # Determine dataset
    dataset = (
        row.get("Dataset", row.get("dataset", "")).strip() or Path(source_file).stem
    )

    # Extract ID
    id_in_csv = row.get("ID", row.get("id", "")).strip()

    # Compute hash - this will validate the reaction SMILES
    try:
        raw_hash = compute_raw_hash(mapped_rxn)
    except Exception:
        # If we can't compute a valid hash, the reaction SMILES is invalid
        return None

    # Normalize other fields
    return {
        "source_file": source_file,
        "line_no": line_no,
        "id_in_csv": id_in_csv or None,
        "dataset": dataset or None,
        "date_raw": row.get("Date", row.get("Year", "")).strip() or None,
        "reaction_smiles": row.get("ReactionSmiles", "").strip() or None,
        "reaction_smiles_clean": row.get("ReactionSmilesClean", "").strip() or None,
        "yield_raw": row.get("Yield", "").strip() or None,
        "badmolecules": _safe_int(row.get("BadMolecules")),
        "reactant_size": _safe_int(row.get("ReactantSize")),
        "product_size": _safe_int(row.get("ProductSize")),
        "confidence": _safe_float(row.get("confidence")),
        "mapped_rxn": mapped_rxn,
        "raw_hash": raw_hash,
    }


def _safe_int(value: Any) -> int | None:
    """Convert to int, return None if invalid."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))  # Handle "42.0" strings
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> float | None:
    """Convert to float, return None if invalid."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def create_tables(conn):
    """Create staging table if it doesn't exist."""
    with conn.cursor() as cur:
        logger.info("Creating staging_reactions table...")
        cur.execute(CREATE_STAGING_TABLE_SQL)

        conn.commit()
    logger.info("✓ Table created successfully")


def load_csv_to_staging(
    conn,
    csv_path: str,
    source_file: str,
) -> dict[str, int]:
    """
    Load a CSV file into the staging table with deduplication and validation.

    Args:
        conn: Database connection
        csv_path: Path to CSV file
        source_file: Identifier for the source file

    Returns:
        Dictionary with statistics (inserted, skipped, errors)
    """
    stats = {
        "total_rows": 0,
        "inserted": 0,
        "duplicate_hash": 0,
        "skipped_errors": 0,
        "added_to_batch": 0,  # Track rows added to batch
    }

    logger.info(f"Loading {csv_path}...")

    # Check existing hashes to avoid duplicates - store hash to mapped_rxn mapping
    with conn.cursor() as cur:
        cur.execute(
            "SELECT raw_hash, mapped_rxn, source_file, line_no, id_in_csv FROM staging_reactions"
        )
        existing_hashes = {}
        for row in cur.fetchall():
            existing_hashes[row[0]] = {
                "mapped_rxn": row[1],
                "source_file": row[2],
                "line_no": row[3],
                "id_in_csv": row[4],
            }

    logger.info(f"  Found {len(existing_hashes)} existing hashes in database")

    # Prepare batches
    batch = []

    with Path(csv_path).open(encoding="utf-8") as f:
        # Use csv.DictReader to handle TSV/CSV
        # These files are tab-delimited (TSV format)
        reader = csv.DictReader(f, delimiter="\t")

        for line_no, row in enumerate(reader, start=2):  # Start at 2 (line 1 is header)
            stats["total_rows"] += 1

            # Normalize row
            normalized = normalize_row_data(row, source_file, line_no)

            if normalized is None:
                # Skip rows with errors - only persist valid reactions
                stats["skipped_errors"] += 1
                continue

            # Check for duplicate hash
            if normalized["raw_hash"] in existing_hashes:
                stats["duplicate_hash"] += 1
                # existing = existing_hashes[normalized['raw_hash']]
                # print(f"\n⚠️  DUPLICATE FOUND (hash: {normalized['raw_hash'][:16]}...)")
                # print(f"   NEW: {source_file}:{line_no} (ID: {normalized.get('id_in_csv', 'N/A')})")
                # print(f"        mapped_rxn: {normalized['mapped_rxn']}")
                # print(f"   EXISTING: {existing['source_file']}:{existing['line_no']} (ID: {existing.get('id_in_csv', 'N/A')})")
                # print(f"        mapped_rxn: {existing['mapped_rxn']}")
                # print()
                continue

            # Add to batch
            batch.append(normalized)
            stats["added_to_batch"] += 1

            # Deduplicate within the current batch before insertion
            if len(batch) >= BATCH_SIZE:
                # Remove duplicates within this batch
                unique_batch = {}
                batch_dupes = 0
                for item in batch:
                    if item["raw_hash"] not in unique_batch:
                        unique_batch[item["raw_hash"]] = item
                    else:
                        batch_dupes += 1

                batch_list = list(unique_batch.values())
                stats["duplicate_hash"] += batch_dupes

                # Insert deduplicated batch
                inserted = _insert_batch(conn, batch_list)
                stats["inserted"] += inserted

                # Verify accounting
                expected_processed = (
                    stats["inserted"]
                    + stats["duplicate_hash"]
                    + stats["skipped_errors"]
                )
                if expected_processed != stats["total_rows"]:
                    logger.warning(
                        f"\n⚠️  ACCOUNTING MISMATCH at row {stats['total_rows']}:"
                    )
                    logger.warning(f"     Total rows: {stats['total_rows']}")
                    logger.warning(f"     Inserted: {stats['inserted']}")
                    logger.warning(f"     Duplicates: {stats['duplicate_hash']}")
                    logger.warning(f"     Skipped: {stats['skipped_errors']}")
                    logger.warning(f"     Sum: {expected_processed}")
                    logger.warning(
                        f"     Missing: {stats['total_rows'] - expected_processed}"
                    )
                    logger.warning(f"     Added to batch: {stats['added_to_batch']}")
                    logger.warning(f"     Batch size before dedup: {len(batch)}")
                    logger.warning(f"     Batch size after dedup: {len(batch_list)}")
                    logger.warning(f"     Rows actually inserted: {inserted}")

                # Track hashes AFTER successful insertion
                for item in batch_list:
                    existing_hashes[item["raw_hash"]] = {
                        "mapped_rxn": item["mapped_rxn"],
                        "source_file": source_file,
                        "line_no": item["line_no"],
                        "id_in_csv": item.get("id_in_csv"),
                    }

                batch = []

                logger.info(
                    f"  Processed {stats['total_rows']} rows... "
                    f"({stats['inserted']} inserted, {stats['duplicate_hash']} dupes, "
                    f"{stats['skipped_errors']} skipped)"
                )

    # Insert remaining rows
    if batch:
        logger.info(f"\n  Processing final batch of {len(batch)} rows...")
        # Remove duplicates within this batch
        unique_batch = {}
        batch_dupes = 0
        for item in batch:
            if item["raw_hash"] not in unique_batch:
                unique_batch[item["raw_hash"]] = item
            else:
                batch_dupes += 1

        batch_list = list(unique_batch.values())
        stats["duplicate_hash"] += batch_dupes

        logger.info(
            f"    Final batch before dedup: {len(batch)}, after dedup: {len(batch_list)}, dupes: {batch_dupes}"
        )

        # Insert deduplicated batch
        inserted = _insert_batch(conn, batch_list)
        stats["inserted"] += inserted

        logger.info(f"    Rows actually inserted from final batch: {inserted}")

        # Track hashes AFTER successful insertion
        for item in batch_list:
            existing_hashes[item["raw_hash"]] = {
                "mapped_rxn": item["mapped_rxn"],
                "source_file": source_file,
                "line_no": item["line_no"],
                "id_in_csv": item.get("id_in_csv"),
            }

        logger.info(
            f"  Processed {stats['total_rows']} rows... "
            f"({stats['inserted']} inserted, {stats['duplicate_hash']} dupes, "
            f"{stats['skipped_errors']} skipped)"
        )

    conn.commit()

    # Verify final accounting
    expected_total = (
        stats["inserted"] + stats["duplicate_hash"] + stats["skipped_errors"]
    )
    logger.info("\n  ===== FINAL ACCOUNTING =====")
    logger.info(f"  Total rows read:        {stats['total_rows']}")
    logger.info(f"  Inserted:               {stats['inserted']}")
    logger.info(f"  Duplicate hashes:       {stats['duplicate_hash']}")
    logger.info(f"  Skipped (errors):       {stats['skipped_errors']}")
    logger.info(f"  Added to batch total:   {stats['added_to_batch']}")
    logger.info(f"  Sum (ins+dup+skip):     {expected_total}")
    if expected_total != stats["total_rows"]:
        logger.warning(
            f"  ⚠️  MISMATCH: Missing {stats['total_rows'] - expected_total} rows!"
        )
    else:
        logger.info("  ✓ Accounting matches!")
    logger.info("  ============================")

    # Print final summary for this file
    logger.info(
        f"\n  Final: {stats['total_rows']} total rows "
        f"({stats['inserted']} inserted, {stats['duplicate_hash']} dupes, "
        f"{stats['skipped_errors']} skipped)"
    )

    return stats


def _insert_batch(conn, batch: list[dict[str, Any]]) -> int:
    """Insert a batch of rows into staging_reactions."""
    if not batch:
        return 0

    columns = [
        "source_file",
        "line_no",
        "id_in_csv",
        "dataset",
        "date_raw",
        "reaction_smiles",
        "reaction_smiles_clean",
        "yield_raw",
        "badmolecules",
        "reactant_size",
        "product_size",
        "confidence",
        "mapped_rxn",
        "raw_hash",
    ]

    values = [tuple(row[col] for col in columns) for row in batch]

    insert_sql = sql.SQL("""
        INSERT INTO staging_reactions ({})
        VALUES %s
        ON CONFLICT (raw_hash) DO NOTHING
        RETURNING raw_hash
    """).format(sql.SQL(", ").join(map(sql.Identifier, columns)))

    with conn.cursor() as cur:
        # Use page_size equal to batch size to avoid multiple statements
        # and fetch=True to get all inserted rows (needed for accurate count with ON CONFLICT)
        inserted_rows = execute_values(
            cur, insert_sql, values, page_size=len(batch), fetch=True
        )
        actual_inserted = len(inserted_rows) if inserted_rows else 0

        # Log if there's a discrepancy (ON CONFLICT caused some to be skipped)
        if actual_inserted < len(batch):
            logger.warning(
                f"      ⚠️  _insert_batch: Tried to insert {len(batch)} rows, but only {actual_inserted} were inserted"
            )
            logger.warning(
                f"         (ON CONFLICT skipped {len(batch) - actual_inserted} rows with duplicate hashes)"
            )

        return actual_inserted


def print_statistics(conn):
    """Print summary statistics from the staging table."""
    with conn.cursor() as cur:
        # Total rows
        cur.execute("SELECT COUNT(*) FROM staging_reactions")
        total = cur.fetchone()[0]

        # By dataset
        cur.execute("""
            SELECT dataset, COUNT(*)
            FROM staging_reactions
            GROUP BY dataset
            ORDER BY COUNT(*) DESC
        """)
        by_dataset = cur.fetchall()

    logger.info("\n" + "=" * 70)
    logger.info("STAGING TABLE STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total rows in staging:     {total:,}")
    logger.info()
    logger.info("Rows by dataset:")
    for dataset, count in by_dataset:
        logger.info(f"  {dataset or '(null)':30s} {count:,}")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("Phase A — Staging & Normalization")
    logger.info("=" * 70)

    # Check files exist
    uspto_path = Path(USPTO_CSV)
    ord_path = Path(ORD_CSV)

    if not uspto_path.exists():
        logger.error(f"ERROR: {USPTO_CSV} not found")
        sys.exit(1)

    if not ord_path.exists():
        logger.error(f"ERROR: {ORD_CSV} not found")
        sys.exit(1)

    # Connect to database
    logger.info(
        f"\nConnecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}..."
    )
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding("UTF8")  # Ensure proper Unicode handling
        conn.autocommit = False
        logger.info("✓ Connected successfully")
    except Exception as e:
        logger.error(f"ERROR: Could not connect to database: {e}")
        logger.error(
            "\nPlease update DB_CONFIG in the script with your PostgreSQL credentials."
        )
        sys.exit(1)

    try:
        # Create tables
        create_tables(conn)

        # Load USPTO data
        uspto_stats = load_csv_to_staging(
            conn, str(uspto_path), "uspto_data_mapped.csv"
        )
        logger.info(
            f"\n✓ USPTO loaded: {uspto_stats['inserted']:,} inserted, "
            f"{uspto_stats['duplicate_hash']:,} duplicates, "
            f"{uspto_stats['skipped_errors']:,} skipped, "
            f"{uspto_stats['total_rows']:,} total"
        )

        # Load ORD data
        ord_stats = load_csv_to_staging(conn, str(ord_path), "ord_data_mapped.csv")
        logger.info(
            f"✓ ORD loaded: {ord_stats['inserted']:,} inserted, "
            f"{ord_stats['duplicate_hash']:,} duplicates, "
            f"{ord_stats['skipped_errors']:,} skipped, "
            f"{ord_stats['total_rows']:,} total"
        )

        # Print overall statistics
        print_statistics(conn)

        logger.info("\n✓ Phase A complete!")
        logger.info("\nNext steps:")
        logger.info("  - Review staging data: SELECT * FROM staging_reactions;")
        logger.info("  - Proceed to Phase B to compute chemistry artifacts")

    except Exception as e:
        logger.error(f"\nERROR: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
