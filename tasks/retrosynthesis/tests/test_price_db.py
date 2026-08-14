"""Tests for the frozen local buyables database lookup."""

import json
import sqlite3

import pytest
from retrosynthesis.price_db import canonicalize_smiles, lookup_prices


@pytest.fixture
def price_database(tmp_path):
    path = tmp_path / "buyables.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE buyables (
                canonical_smiles TEXT NOT NULL,
                price_usd_per_g REAL NOT NULL,
                median_price_usd_per_g REAL NOT NULL,
                n_observations INTEGER NOT NULL,
                sources TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO buyables VALUES (?, ?, ?, ?, ?)",
            [
                ("CCO", 3.2, 4.1, 2, json.dumps(["a", "b"])),
                ("c1ccccc1", 1.5, 1.5, 1, json.dumps(["a"])),
            ],
        )
    return path


def test_canonicalize_removes_atom_mapping_and_preserves_stereochemistry():
    assert canonicalize_smiles("[CH3:1][CH2:2][OH:3]") == "CCO"
    assert canonicalize_smiles("C[C@H](O)Cl") != canonicalize_smiles("C[C@@H](O)Cl")


def test_canonicalize_rejects_invalid_and_empty_smiles():
    assert canonicalize_smiles("not-smiles") is None
    assert canonicalize_smiles("") is None


def test_lookup_matches_canonical_smiles_and_keeps_original_keys(price_database):
    hits = lookup_prices(
        ["C1=CC=CC=C1", "[CH3:1][CH2:2][OH:3]", "CCCC"],
        db_path=price_database,
    )

    assert hits["C1=CC=CC=C1"] == {
        "canonical_smiles": "c1ccccc1",
        "price_usd_per_g": 1.5,
        "sources": ["a"],
    }
    assert hits["[CH3:1][CH2:2][OH:3]"]["price_usd_per_g"] == 3.2
    assert hits["CCCC"] is None


def test_lookup_invalid_only_does_not_require_database():
    assert lookup_prices(["not-smiles"], db_path="missing.sqlite") == {
        "not-smiles": None
    }


def test_lookup_reports_missing_database_for_valid_smiles(tmp_path):
    with pytest.raises(FileNotFoundError, match="Build it with"):
        lookup_prices(["CCO"], db_path=tmp_path / "missing.sqlite")
