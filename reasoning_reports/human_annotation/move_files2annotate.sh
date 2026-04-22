#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_BASE="$REPO_ROOT/reasoning_reports"
DEST="$REPO_ROOT/reasoning_reports/human_annotation/files2annotate"

mkdir -p "$DEST"

files=(
    "claude_sonnet_45/afm/level_2/afm_experiment_level_2-8.json"
    "gpt_4o/ml/level_1/ml_oxides-13.json"
    "gpt_4o/wetlab/level_2/qualysis_lvl2_08-70.json"
    "claude_sonnet_45/retrosynthesis/level_3/make_5_lvl3-43.json"
    "claude_sonnet_45/spectra/level_1/10_15227_orgsyn_096_0036-19.json"
    "claude_sonnet_45/ml/level_1/ml_sulphides-10.json"
)

for f in "${files[@]}"; do
    dir="$(dirname "$f")"
    base="$(basename "$f" .json)"
    src="$SRC_BASE/$dir/annotated/${base}.annotated.json"
    if [[ -f "$src" ]]; then
        mv "$src" "$DEST/"
        echo "Moved: $src -> $DEST/"
    else
        echo "WARNING: not found: $src" >&2
    fi
done
