"""Build the frozen retrosynthesis buyables SQLite database."""

# ruff: noqa: T201

import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any, TextIO

from rdkit import Chem

PriceRow = tuple[str, float, str]
SOURCE_METADATA = {
    "coprinet": {
        "url": "https://github.com/oxpig/CoPriNet/blob/main/data/testData/test_set_PC.csv",
        "license": "MIT repository",
    },
    "chemcost": {
        "url": "https://huggingface.co/datasets/nips2026-chemcost/chemcost-bench/blob/main/chemcost.jsonl",
        "license": "CC BY 4.0",
    },
    "askcos": {
        "url": "https://askcos-docs.mit.edu/guide/3-Advanced-Usage/3.5-Utilities.html",
        "license": "Verify the provenance of the supplied snapshot before redistribution",
    },
}
MANUAL_SOURCE = "corral_manual"
MANUAL_SOURCE_METADATA = {
    "license": "Repository-authored benchmark data",
    "provenance": (
        "Manually assigned frozen USD/g estimates for reference-route leaves; "
        "these are benchmark values, not live vendor quotes"
    ),
}
MANUAL_PRICES = {
    "C/C(=C/CCCC(=O)C(F)(F)F)c1ccccc1": 150.0,
    "COC(=O)C1=CC(=O)C1": 40.0,
    "COC(=O)C(=O)CCc1ccccc1": 60.0,
    "COc1ccc(C(=O)c2ccc(OC)cc2)cc1": 15.0,
    "[Li]C[Si](C)(C)C": 100.0,
    "COC#CC(C)P(=O)(OCC(F)(F)F)OCC(F)(F)F": 450.0,
    "CCOC(=O)C[C@H]1C[C@H](O)[C@@H](C)[C@@H](C=O)O1": 500.0,
    "OC1CN(Cc2ccccc2)C1": 200.0,
    "O=Cc1ccc(S(=O)(=O)Cl)cc1": 60.0,
    "O=C1CCC1": 30.0,
    "CC(=O)c1cc([N+](=O)[O-])ccc1O": 25.0,
}


def canonicalize_smiles(smiles: Any) -> str | None:
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


def normalize_entry(smiles: Any, price: Any, source: str) -> PriceRow | None:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None

    try:
        numeric_price = float(price)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_price) or numeric_price <= 0:
        return None

    return canonical, numeric_price, source


def load_coprinet(path: Path) -> Iterator[PriceRow]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"SMILES", "price"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain SMILES and price columns")

        for row in reader:
            entry = normalize_entry(row["SMILES"], row["price"], "coprinet_mcule")
            if entry is not None:
                yield entry


def load_manual_prices() -> Iterator[PriceRow]:
    """Yield the benchmark's built-in manual SMILES/price rows."""
    for smiles, price in MANUAL_PRICES.items():
        entry = normalize_entry(smiles, price, MANUAL_SOURCE)
        if entry is None:
            raise ValueError(f"Invalid built-in manual price row: {smiles}")
        yield entry


def walk_chemcost(obj: Any) -> Iterator[tuple[Any, Any]]:
    """Recursively find ChemCost objects containing a SMILES and USD/g price."""
    if isinstance(obj, dict):
        if "smiles" in obj:
            price = obj.get("price_per_gram_usd")
            if price is None:
                price_range = obj.get("price_range") or {}
                if isinstance(price_range, dict):
                    price = price_range.get("median")
            if price is not None:
                yield obj["smiles"], price

        for value in obj.values():
            yield from walk_chemcost(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_chemcost(value)


def load_chemcost(path: Path) -> Iterator[PriceRow]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error

            for smiles, price in walk_chemcost(record):
                entry = normalize_entry(smiles, price, "chemcost")
                if entry is not None:
                    yield entry


def _open_askcos(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_askcos(path: Path) -> Iterator[PriceRow]:
    with _open_askcos(path) as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")

    for item in data:
        if not isinstance(item, dict):
            continue
        source = f"askcos:{item.get('source', 'unknown')}"
        entry = normalize_entry(item.get("smiles"), item.get("ppg"), source)
        if entry is not None:
            yield entry


def build_database(rows: Iterable[PriceRow], output: Path) -> tuple[int, int]:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    raw_records = 0
    for smiles, price, source in rows:
        grouped[smiles].append((price, source))
        raw_records += 1

    if not grouped:
        raise ValueError("No valid priced molecules were found in the input files")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary_output = Path(handle.name)

    try:
        with sqlite3.connect(temporary_output) as conn:
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
                """
                INSERT INTO buyables (
                    canonical_smiles,
                    price_usd_per_g,
                    median_price_usd_per_g,
                    n_observations,
                    sources
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        smiles,
                        min(price for price, _source in observations),
                        statistics.median(price for price, _source in observations),
                        len(observations),
                        json.dumps(sorted({source for _price, source in observations})),
                    )
                    for smiles, observations in sorted(grouped.items())
                ),
            )
            conn.execute(
                "CREATE UNIQUE INDEX idx_buyables_smiles ON buyables(canonical_smiles)"
            )
            conn.execute("PRAGMA optimize")
        temporary_output.replace(output)
        output.chmod(0o644)
    finally:
        temporary_output.unlink(missing_ok=True)

    return raw_records, len(grouped)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manual_prices_sha256() -> str:
    payload = json.dumps(MANUAL_PRICES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_metadata(
    metadata_path: Path,
    output: Path,
    inputs: dict[str, Path],
    raw_records: int,
    final_molecules: int,
) -> None:
    metadata = {
        "schema_version": 1,
        "price_definition": "Frozen estimated cost in USD for 1 g",
        "aggregation": "minimum USD/g per canonical isomeric SMILES",
        "raw_records": raw_records,
        "final_molecules": final_molecules,
        "database": {"file": output.name, "sha256": sha256(output)},
        "embedded_sources": {
            MANUAL_SOURCE: {
                **MANUAL_SOURCE_METADATA,
                "records": len(MANUAL_PRICES),
                "sha256": manual_prices_sha256(),
            }
        },
        "inputs": {
            source: {
                "file": path.name,
                "sha256": sha256(path),
                **SOURCE_METADATA[source],
            }
            for source, path in sorted(inputs.items())
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coprinet", type=Path)
    parser.add_argument("--chemcost", type=Path)
    parser.add_argument("--askcos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata JSON path (defaults to <output>.metadata.json)",
    )
    args = parser.parse_args()

    loaders = {
        "coprinet": (args.coprinet, load_coprinet),
        "chemcost": (args.chemcost, load_chemcost),
        "askcos": (args.askcos, load_askcos),
    }
    inputs = {name: path for name, (path, _loader) in loaders.items() if path}
    if not inputs:
        parser.error("at least one input dataset is required")

    dataset_rows = (
        row
        for path, loader in loaders.values()
        if path is not None
        for row in loader(path)
    )
    rows = chain(dataset_rows, load_manual_prices())
    raw_records, final_molecules = build_database(rows, args.output)
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    write_metadata(metadata_path, args.output, inputs, raw_records, final_molecules)
    print(f"Raw records: {raw_records:,}")
    print(f"Final molecules: {final_molecules:,}")
    print(f"Database: {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
