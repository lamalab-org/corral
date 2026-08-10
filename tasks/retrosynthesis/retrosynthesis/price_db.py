"""Read-only access to the frozen local buyables database."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from rdkit import Chem

DEFAULT_DB_PATH = Path(__file__).parent / "data" / "buyables.sqlite"
DB_PATH = Path(os.environ.get("RETRO_PRICE_DB_PATH", DEFAULT_DB_PATH))
SQLITE_PARAMETER_LIMIT = 900


def canonicalize_smiles(smiles: str) -> str | None:
    """Return an atom-map-free canonical SMILES, or ``None`` if invalid."""
    if not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _database_path(db_path: str | Path | None) -> Path:
    path = Path(db_path) if db_path is not None else DB_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Buyables database not found at {path}. "
            "Build it with scripts/build_buyables.py or set RETRO_PRICE_DB_PATH."
        )
    return path


def lookup_prices(
    smiles_list: list[str], db_path: str | Path | None = None
) -> dict[str, dict[str, Any] | None]:
    """Look up frozen USD/g prices by exact canonical SMILES."""
    canonical = {smiles: canonicalize_smiles(smiles) for smiles in smiles_list}
    unique = sorted({value for value in canonical.values() if value is not None})

    if not unique:
        return dict.fromkeys(smiles_list)

    rows: list[tuple[str, float, str]] = []
    with sqlite3.connect(_database_path(db_path)) as conn:
        for offset in range(0, len(unique), SQLITE_PARAMETER_LIMIT):
            chunk = unique[offset : offset + SQLITE_PARAMETER_LIMIT]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                conn.execute(
                    f"""
                    SELECT canonical_smiles, price_usd_per_g, sources
                    FROM buyables
                    WHERE canonical_smiles IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()
            )

    by_canonical = {
        smiles: {
            "canonical_smiles": smiles,
            "price_usd_per_g": float(price),
            "sources": json.loads(sources),
        }
        for smiles, price, sources in rows
    }
    return {
        original: by_canonical.get(canonical_smiles)
        for original, canonical_smiles in canonical.items()
    }
