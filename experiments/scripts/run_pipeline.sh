#!/usr/bin/env bash
# Run every measurement against one attribution graph, waiting for it to appear if needed.
# Usage: run_pipeline.sh <graph path> <tag> [extra args passed to each script]
set -u
GRAPH="$1"; shift
TAG="$1"; shift
cd ~/qk-attribution || exit 1
PY=./.venv/bin/python
mkdir -p results

for _ in $(seq 1 720); do
  [ -s "$GRAPH" ] && break
  sleep 30
done
if [ ! -s "$GRAPH" ]; then
  echo "[pipeline] gave up waiting for $GRAPH"
  exit 1
fi
echo "[pipeline] graph present: $(ls -l "$GRAPH")"

for script in measure_completeness measure_feature_rank measure_edge_loadings; do
  echo "[pipeline] === $script ==="
  $PY "experiments/scripts/$script.py" --graph "$GRAPH" \
    --out "results/${TAG}-${script}.json" "$@" 2>&1 | grep -viE "^WARNING|it/s\]$"
done
echo "[pipeline] === explain_attention ==="
$PY experiments/scripts/explain_attention.py --graph "$GRAPH" \
  --out "results/${TAG}-explain.json" --pairs 6 "$@" 2>&1 | grep -viE "^WARNING|it/s\]$"
echo "[pipeline] done"
