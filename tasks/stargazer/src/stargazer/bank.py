"""Generate, calibrate, freeze, and verify controller-only Stargazer banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from dataclasses import asdict
from pathlib import Path

import numpy as np

from stargazer.identifiability import AUDIT_VERSION, BankRules, audit_identifiability
from stargazer.models import (
    PlanetParams,
    load_task,
    mass_from_semi_amplitude,
    simulate_keplerian_rv,
)

GENERATOR_VERSION = "stargazer-rv-bank-v1"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_record(seed: int, task_id: str, difficulty: int) -> dict:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 3) if difficulty <= 7 else rng.integers(2, 4))
    mass = float(rng.uniform(0.7, 1.3))
    periods = np.exp(
        np.linspace(np.log(8), np.log(85), count) + rng.uniform(-0.12, 0.12, count)
    )
    planets = []
    for period in periods:
        eccentricity = float(rng.uniform(0.01, 0.2))
        amplitude = float(rng.uniform(2.5, 9))
        planets.append(
            PlanetParams(
                P_days=float(period),
                m_sin_i_mjup=mass_from_semi_amplitude(
                    amplitude, float(period), eccentricity, mass
                ),
                e=eccentricity,
                omega_rad=float(rng.uniform(0, 2 * np.pi)),
                l_rad=float(rng.uniform(0, 2 * np.pi)),
            )
        )
    span = float(max(periods) * rng.uniform(7, 10))
    times = np.sort(rng.uniform(0, span, 220))
    labels = rng.choice(["instA", "instB"], len(times))
    sigma = rng.uniform(0.7, 1.0, len(times))
    offsets = {"instA": float(rng.normal(0, 3)), "instB": float(rng.normal(0, 3))}
    clean = simulate_keplerian_rv(planets, times, mass) + np.array(
        [offsets[label] for label in labels]
    )
    observed = clean + rng.normal(0, sigma)
    withheld = np.sort(rng.uniform(times[-1], times[-1] + span * 0.3, 40))
    withheld_labels = rng.choice(["instA", "instB"], len(withheld))
    withheld_clean = simulate_keplerian_rv(
        planets, np.concatenate((times, withheld)), mass
    )[len(times) :]
    withheld_rv = (
        withheld_clean
        + np.array([offsets[label] for label in withheld_labels])
        + rng.normal(0, 0.85, len(withheld))
    )
    return {
        "task_id": task_id,
        "truth_difficulty": difficulty,
        "config": {
            "star": {"M_star_sun": mass},
            "planets": [
                {k: v for k, v in asdict(p).items() if v is not None} for p in planets
            ],
            "noise": {"sigma_jitter_ms": 0.0},
            "instruments": [
                {"label": key, "gamma_ms": value} for key, value in offsets.items()
            ],
            "los_axis": "x",
            "integrator_preference": "whfast",
        },
        "observations": {
            "times_days": times.tolist(),
            "rvs_ms": observed.tolist(),
            "sigmas_ms": sigma.tolist(),
            "instruments": labels.tolist(),
        },
        "meta": {
            "rv_semantics": "rv_only",
            "generator_version": GENERATOR_VERSION,
            "withheld_observations": {
                "times_days": withheld.tolist(),
                "rvs_ms": withheld_rv.tolist(),
                "sigmas_ms": [0.85] * len(withheld),
                "instruments": withheld_labels.tolist(),
            },
        },
    }


def verify_bank(root: str | Path, *, purpose: str | None = None) -> dict:
    root = Path(root)
    manifest = json.loads((root / "private-manifest.json").read_text())
    if manifest.get("generator_version") != GENERATOR_VERSION or not manifest.get(
        "frozen"
    ):
        raise ValueError("Unfrozen or incompatible private bank")
    if purpose is not None and manifest["purpose"] != purpose:
        raise ValueError(f"Expected a {purpose} bank")
    expected = manifest["files"]
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "private-manifest.json"
    }
    if actual != set(expected):
        raise ValueError("Private bank membership changed")
    for name, digest in expected.items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or _hash(path) != digest
        ):
            raise ValueError("Private bank content hash mismatch")
    bank_hash = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "bank_hash"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if bank_hash != manifest["bank_hash"]:
        raise ValueError("Private bank manifest hash mismatch")
    if manifest["purpose"] == "evaluation" and not manifest.get("calibration_hash"):
        raise ValueError("Evaluation bank requires an independent calibration record")
    return manifest


def generate_bank(
    root: str | Path,
    *,
    per_level: int = 10,
    calibration: str | Path | None = None,
    rules: BankRules | None = None,
    max_attempts: int = 200,
    levels: tuple[int, ...] = (1, 2),
) -> dict:
    """Freeze fresh seeds, IDs, accepted systems and rejected-system diagnostics.

    Without a calibration bank this creates a calibration set. Evaluation
    accepts only the rules already frozen in the supplied calibration manifest.
    """
    if not levels or len(set(levels)) != len(levels) or not set(levels) <= {1, 2}:
        raise ValueError("Select unique levels from 1 and 2")
    if per_level < 1 or max_attempts < len(levels) * per_level:
        raise ValueError("Invalid bank size or attempt budget")
    root = Path(root)
    if root.exists():
        raise FileExistsError(
            "Bank destination must be new; frozen banks are never overwritten"
        )
    calibration_manifest = (
        verify_bank(calibration, purpose="calibration") if calibration else None
    )
    if calibration_manifest:
        if calibration_manifest.get("audit_version") != AUDIT_VERSION:
            raise ValueError("Calibration requires the current recovery audit")
        if rules is not None:
            raise ValueError("Evaluation rules must come from calibration")
        rules = BankRules(**calibration_manifest["rules"])
        if not calibration_manifest["calibration_accepted"]:
            raise ValueError(
                "Calibration did not meet the declared acceptance requirements"
            )
    rules = rules or BankRules()
    root.mkdir(parents=True, mode=0o700)
    memberships = {level: [] for level in levels}
    records = []
    rejected = []
    attempts = 0
    # Calibration and evaluation use independent 256-bit seeds. Opaque IDs are
    # independently generated and carry neither seed nor difficulty information.
    for level in levels:
        while len(memberships[level]) < per_level:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Bank generation exhausted attempts; partial bank remains unfrozen"
                )
            attempts += 1
            seed = secrets.randbits(256)
            task_id = secrets.token_hex(16)
            difficulty = (5 if level == 1 else 8) + len(memberships[level]) % 3
            raw = generate_record(seed, task_id, difficulty)
            scratch = root / "pending.json"
            _write(scratch, raw)
            task = load_task(scratch)
            audit = audit_identifiability(task, rules)
            load_task.cache_clear()
            scratch.unlink()
            record = {
                "task_id": task_id,
                "seed": str(seed),
                "difficulty": difficulty,
                "level": level,
                "audit": audit,
            }
            # Retain evidence even if the attempt budget is later exhausted.
            _write(root / "attempts" / f"{attempts:04d}.json", record)
            if not audit["accepted"]:
                rejected.append(record)
                continue
            _write(root / "synthetic" / f"{task_id}.json", raw)
            memberships[level].append(task_id)
            records.append(record)
    for level, ids in memberships.items():
        _write(
            root / "selectors" / f"level_{level}.json",
            [{"source": "synthetic", "task_ids": ids}],
        )
    _write(
        root / "private-audit.json",
        {"accepted": records, "ambiguous_or_rejected": rejected},
    )
    files = {
        path.relative_to(root).as_posix(): _hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "audit_version": AUDIT_VERSION,
        "criteria": asdict(rules.criteria()),
        "frozen": True,
        "purpose": "evaluation" if calibration else "calibration",
        "rules": asdict(rules),
        "files": files,
        "bank_hash": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "calibration_hash": calibration_manifest["bank_hash"]
        if calibration_manifest
        else None,
        "calibration_accepted": len(records) == len(levels) * per_level,
        "membership": memberships,
        "attempted": attempts,
        "accepted": len(records),
        "rejected": len(rejected),
    }
    manifest["bank_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "bank_hash"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    _write(root / "private-manifest.json", manifest)
    return verify_bank(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "generate", "verify"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--per-level", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--level", type=int, choices=(1, 2), action="append")
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_bank(args.directory)
    else:
        if args.command == "generate" and args.calibration is None:
            parser.error("generate requires --calibration")
        manifest = generate_bank(
            args.directory,
            per_level=args.per_level,
            calibration=args.calibration if args.command == "generate" else None,
            max_attempts=args.max_attempts,
            levels=tuple(args.level) if args.level else (1, 2),
        )
    print(  # noqa: T201 - CLI summary
        json.dumps(
            {
                key: manifest[key]
                for key in ("purpose", "bank_hash", "accepted", "rejected")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
