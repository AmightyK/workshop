#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_PARTITION="${GZ_PARTITION:-warehouse_agv_demo}"
source /opt/ros/jazzy/setup.bash
exec uv run --project "$PROJECT_DIR" python \
  "$PROJECT_DIR/scripts/localization_dashboard.py" "$@"
