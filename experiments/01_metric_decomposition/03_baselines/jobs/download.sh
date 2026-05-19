#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
uv run python3 experiments/00_data_preparation/jobs/download.py "$@"
