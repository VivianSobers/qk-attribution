#!/usr/bin/env bash
# Sync the working tree to a GPU worker and run the full upstream check suite there.
# Usage: scripts/sync-and-check.sh [host] [-- pytest args...]
set -euo pipefail

HOST="${1:-10.10.3.83}"
shift || true
[[ "${1:-}" == "--" ]] && shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="~/qk-attribution"

cd "$REPO_ROOT"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude 'spike_out' \
  ./ "ccbd@${HOST}:${REMOTE_DIR}/"

ssh "ccbd@${HOST}" "cd ${REMOTE_DIR} && . .venv/bin/activate && \
  python -m pytest -q ${*:-} && ruff check . && ruff format --check . && pyright"
