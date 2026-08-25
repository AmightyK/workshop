#!/usr/bin/env bash
set -eo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_PARTITION="${GZ_PARTITION:-warehouse_agv_demo}"

source /opt/ros/jazzy/setup.bash
python3 -u "$DEMO_DIR/tools/build_lidar_map.py" "$@"
