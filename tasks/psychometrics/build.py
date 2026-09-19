#!/usr/bin/env python3
"""Build and check the psychometrics task environment.

Generated artifacts and task definitions come from the scripts in
``generators/``. A rebuild should reproduce them from source.

    python build.py                  # regenerate every task
    python build.py --verify         # check every task is solvable as intended
    python build.py --naive          # check the default analysis fails on each
    python build.py --check          # build, verify, naive, then the scoring tests
    python build.py --tasks 3 7      # limit any of the above to these tasks
    python build.py --level 2        # limit it to one level instead
    python build.py --clean          # delete generated files, then build

Dependencies come from the calling interpreter:

    uv run --with numpy --with pandas --with scipy --with semopy \
      --with factor_analyzer python build.py --check
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERATED = ["artifacts", "environments"]
NAME = re.compile(r"gen_l(?P<level>\d+)_t(?P<task>\d+)_(?P<slug>.+)\.py")


def generators(tasks=None, level=None):
    """Every generator script in level and task order.

    tasks  keep only these task numbers; all of them by default
    level  keep only this level; all of them by default
    """
    found = []
    for path in sorted(ROOT.glob("generators/level_*/gen_l*_t*.py")):
        match = NAME.match(path.name)
        if not match:
            continue
        if tasks is not None and int(match["task"]) not in tasks:
            continue
        if level is not None and int(match["level"]) != level:
            continue
        found.append((int(match["level"]), int(match["task"]), match["slug"], path))
    return sorted(found)


def run(path, flag=None):
    """Run one generator. Returns (succeeded, seconds, last line of output)."""
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(path)] + ([flag] if flag else []),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    tail = lines[-1].strip() if lines else (result.stderr.strip().splitlines() or ["no output"])[-1]
    return result.returncode == 0, time.monotonic() - started, tail


def clean():
    """Delete everything the generators produce."""
    for name in GENERATED:
        target = ROOT / name
        if target.exists():
            shutil.rmtree(target)
            print(f"  removed {name}/")


def stage(label, scripts, flag):
    """Run one flag across every generator and print a table. Returns failures."""
    print(f"\n{label}")
    print(f"  {'task':34s} {'time':>7s}  result")
    failures = []
    for level, task, slug, path in scripts:
        ok, seconds, tail = run(path, flag)
        if not ok:
            failures.append(f"l{level}t{task:02d} {slug}")
        print(
            f"  l{level}t{task:02d} {slug:28s} {seconds:6.1f}s  "
            f"{'ok' if ok else 'FAILED'}  {'' if ok else tail[:70]}"
        )
    return failures


def scoring_tests():
    """Run the scorer's test suite. Returns failures."""
    print("\nScoring tests")
    result = subprocess.run(
        [sys.executable, "tests/test_scoring.py"], capture_output=True, text=True, cwd=ROOT
    )
    tail = [ln for ln in result.stdout.splitlines() if ln.strip()]
    print(f"  {tail[-1] if tail else 'no output'}")
    return [] if result.returncode == 0 else ["scoring tests"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--verify", action="store_true", help="check the intended answer wins on every task"
    )
    parser.add_argument(
        "--naive", action="store_true", help="check the default analysis fails on every task"
    )
    parser.add_argument(
        "--check", action="store_true", help="build, then verify, naive and the scoring tests"
    )
    parser.add_argument(
        "--clean", action="store_true", help="delete generated files before building"
    )
    parser.add_argument(
        "--tasks", type=int, nargs="+", metavar="N", help="limit to these task numbers"
    )
    parser.add_argument("--level", type=int, metavar="N", help="limit to this level")
    args = parser.parse_args()

    scripts = generators(set(args.tasks) if args.tasks else None, args.level)
    if not scripts:
        print("no generators matched")
        return 1

    failures = []
    if args.clean:
        print("Cleaning")
        clean()

    # The scoring tests read the built datasets, so --check builds first.
    if args.check or not (args.verify or args.naive):
        failures += stage("Building", scripts, None)
    if args.check or args.verify:
        failures += stage("Verifying (the intended answer wins)", scripts, "--verify")
    if args.check or args.naive:
        failures += stage("Checking the obvious analysis fails", scripts, "--naive")
    if args.check:
        failures += scoring_tests()

    print()
    if failures:
        for item in failures:
            print(f"FAILED  {item}")
        return 1
    print(f"{len(scripts)} task(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
