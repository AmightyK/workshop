#!/usr/bin/env bash
set -eo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_PARTITION="${GZ_PARTITION:-warehouse_agv_demo}"

exec "$DEMO_DIR/run_vqa_mission.sh" --pick-only "$@"
