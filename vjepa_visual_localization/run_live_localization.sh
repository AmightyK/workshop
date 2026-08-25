#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
exec uv run --project "$PROJECT_DIR" python \
  "$PROJECT_DIR/scripts/run_live_localization.py" "$@"
