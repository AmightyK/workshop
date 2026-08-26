#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
exec python3 -u "$DEMO_DIR/scripts/udp_command_bridge.py" \
  --demo-dir "$DEMO_DIR" "$@"
