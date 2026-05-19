#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate
python experiments/phase1_baselines/jobs/run_baselines.py "$@"
