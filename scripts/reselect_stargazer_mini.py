"""Audit Stargazer mini-v2 selection from local reports; --write updates the release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from corral_mini_common import (
    load_report_strata,
    rank_cost_tie_counts,
    row_spearman,
    select,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = ROOT / "src/corral/subsets/corral_mini_v2.json"
    manifest = json.loads(manifest_path.read_text())
    trials_csv = ROOT / "reports/stargazer/trials.csv"
    data = pd.read_csv(trials_csv)
    agents = sorted((data.agent + "__" + data.model).unique())
    strata = load_report_strata(
        trials_csv,
        ROOT / "corral_mini/prices.json",
        agents,
        manifest["score_metric"],
        "stargazer",
    )
    rows, ties = [], {}
    for (_, level), stratum in strata.items():
        indexes = np.arange(len(agents))
        chosen = select(
            stratum,
            manifest["tasks_per_level"],
            manifest["method"],
            indexes,
            exclude_saturated=manifest["exclude_saturated"],
            seed=manifest["seed"],
        )
        old = manifest["tasks_by_environment"]["stargazer"][f"level_{level}"]
        new = stratum.tasks[chosen].tolist()
        audit = rank_cost_tie_counts(
            stratum, manifest["tasks_per_level"], indexes, manifest["exclude_saturated"]
        )
        rows.append(
            {
                "level": level,
                "previous_tasks": old,
                "selected_tasks": new,
                "changed": old != new,
                "spearman": float(
                    row_spearman(
                        stratum.passes[chosen].mean(axis=0)[None],
                        stratum.passes.mean(axis=0),
                    )[0]
                ),
                "full_pass_hat_5": float(stratum.passes.mean()),
                "mini_pass_hat_5": float(stratum.passes[chosen].mean()),
                **audit,
            }
        )
        ties[level] = audit
        manifest["tasks_by_environment"]["stargazer"][f"level_{level}"] = new
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "mini-selection.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    sys.stdout.write(json.dumps(rows, indent=2) + "\n")
    if args.write:
        source = {}
        for name in ("runs", "trials"):
            path = trials_csv.with_name(f"{name}.csv")
            source[f"{name}_csv"] = str(path.relative_to(ROOT))
            source[f"{name}_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.setdefault("local_sources", {})["stargazer"] = source
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        tie_path = ROOT / "corral_mini/rank_cost_pass_hat_5_ties.csv"
        with tie_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, records = reader.fieldnames, list(reader)
        for row in records:
            if row["environment"] == "stargazer":
                row.update(ties[int(row["level"])])
        with tie_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
