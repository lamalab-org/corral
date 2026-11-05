"""Download every environment's trials.csv from the corral GitHub repository.

The benchmark reports live on ``main``. This fetches
``reports/<environment>/trials.csv`` for each environment there into a local
folder with the same layout, so ``build_dashboard.py --reports <folder>`` can
build the dashboard on a branch that does not carry the reports.

Run from the repository root:
    python reports/dashboard/fetch_trials.py --dest _data/reports
    python reports/dashboard/fetch_trials.py --dest _data/reports --ref <branch>

Uses only the standard library. Set GITHUB_TOKEN to raise the API rate limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "lamalab-org/corral"


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "corral-dashboard"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def environments(repo: str, ref: str) -> list[str]:
    listing = json.loads(
        get(f"https://api.github.com/repos/{repo}/contents/reports?ref={ref}")
    )
    return sorted(
        entry["name"]
        for entry in listing
        if entry["type"] == "dir" and not entry["name"].startswith(("_", "."))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="folder to write <environment>/trials.csv into",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="branch, tag or commit to read (default: main)",
    )
    parser.add_argument(
        "--repo", default=REPO, help=f"GitHub repository (default: {REPO})"
    )
    args = parser.parse_args()

    fetched = 0
    for env in environments(args.repo, args.ref):
        url = f"https://raw.githubusercontent.com/{args.repo}/{args.ref}/reports/{env}/trials.csv"
        try:
            data = get(url)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue  # a reports/ folder that is not an environment, such as figures/
            raise
        target = args.dest / env / "trials.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        fetched += 1
        trials = data.count(b"\n") - 1
        sys.stdout.write(f"{env}: {trials} trials\n")
    if not fetched:
        sys.exit(f"No trials.csv found under reports/ on {args.repo}@{args.ref}")


if __name__ == "__main__":
    main()
