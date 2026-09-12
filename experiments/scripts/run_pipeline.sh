#!/usr/bin/env bash
# Wait for an attribution graph to appear, then run every measurement against it.
# Usage: run_pipeline.sh <graph path> <tag>
set -u
GRAPH="$1"
TAG="$2"
cd ~/qk-attribution || exit 1
. .venv/bin/activate 2>/dev/null || true
PY=$(command -v python)
[ -x ./.venv/bin/python ] && PY=./.venv/bin/python
mkdir -p results

echo "[pipeline] waiting for $GRAPH"
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
  $PY "experiments/scripts/$script.py" --graph "$GRAPH" --out "results/${TAG}-${script}.json" \
    2>&1 | grep -viE "^WARNING|it/s\]$"
  echo "[pipeline] $script exit=$?"
done
echo "[pipeline] done"
