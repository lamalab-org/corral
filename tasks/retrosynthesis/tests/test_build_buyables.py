"""Tests for the frozen buyables database builder."""

import gzip
import json
import runpy
import sqlite3
from pathlib import Path

builder = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "build_buyables.py")
)
build_database = builder["build_database"]
load_askcos = builder["load_askcos"]
load_chemcost = builder["load_chemcost"]
load_coprinet = builder["load_coprinet"]
normalize_entry = builder["normalize_entry"]


def test_normalize_entry_filters_bad_records_and_removes_atom_maps():
    assert normalize_entry("[CH3:1][OH:2]", "2.5", "test") == (
        "CO",
        2.5,
        "test",
    )
    assert normalize_entry("not-smiles", 2.5, "test") is None
    assert normalize_entry("CCO", 0, "test") is None
    assert normalize_entry("CCO", "nan", "test") is None


def test_build_database_aggregates_prices_and_sources(tmp_path):
    output = tmp_path / "buyables.sqlite"
    rows = [
        ("CCO", 5.0, "source_b"),
        ("CCO", 2.0, "source_a"),
        ("CCO", 8.0, "source_a"),
        ("CO", 1.5, "source_a"),
    ]

    assert build_database(rows, output) == (4, 2)

    with sqlite3.connect(output) as conn:
        row = conn.execute(
            "SELECT * FROM buyables WHERE canonical_smiles = 'CCO'"
        ).fetchone()
        indexes = conn.execute("PRAGMA index_list(buyables)").fetchall()

    assert row == ("CCO", 2.0, 5.0, 3, '["source_a", "source_b"]')
    assert any(index[1] == "idx_buyables_smiles" for index in indexes)


def test_source_loaders_accept_documented_formats(tmp_path):
    coprinet = tmp_path / "coprinet.csv"
    coprinet.write_text("SMILES,price\nCCO,3.2\n", encoding="utf-8")

    chemcost = tmp_path / "chemcost.jsonl"
    chemcost.write_text(
        json.dumps(
            {
                "components": [
                    {"smiles": "CO", "price_per_gram_usd": 1.1},
                    {"smiles": "CC", "price_range": {"median": 2.2}},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    askcos = tmp_path / "buyables.json.gz"
    with gzip.open(askcos, "wt", encoding="utf-8") as handle:
        json.dump([{"smiles": "CCC", "ppg": 4.4, "source": "MC"}], handle)

    assert list(load_coprinet(coprinet)) == [("CCO", 3.2, "coprinet_mcule")]
    assert list(load_chemcost(chemcost)) == [
        ("CO", 1.1, "chemcost"),
        ("CC", 2.2, "chemcost"),
    ]
    assert list(load_askcos(askcos)) == [("CCC", 4.4, "askcos:MC")]
