#!/bin/bash
# Flags any running Opus 4.8 job whose run.log has been stale > THRESHOLD seconds
# (likely the recurring aiohttp Bedrock hang). Prints flagged jobs; does NOT kill.
ROOT=/Users/nalampara/n0w0f/research/corral/reports/claude-opus-4.8
THRESHOLD=${1:-720}
now=$(date +%s)
flagged=0
for d in $(find "$ROOT" -name run.py | xargs -n1 dirname | sort); do
  name=${d#$ROOT/}
  grep -q "Benchmark completed" "$d/run.log" 2>/dev/null && continue
  [ -f "$d/run.log" ] || continue
  mt=$(stat -f %m "$d/run.log")
  age=$((now - mt))
  if [ "$age" -gt "$THRESHOLD" ]; then
    echo "STALE ${age}s  $name"
    flagged=$((flagged+1))
  fi
done
[ "$flagged" -eq 0 ] && echo "OK: no stale jobs (threshold ${THRESHOLD}s)"
