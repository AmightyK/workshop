#!/usr/bin/env bash
# Start the UDP command bridge that AIWaiter's voice machine talks to.
#
# Two bridges implement this protocol: AIWaiter's `src/robot_link/bridge.py`,
# which is what the voice machine's sender and `make say` are written against,
# and this repo's older `scripts/udp_command_bridge.py`. Prefer AIWaiter's when
# its checkout is present so a voice demo needs no second terminal; fall back to
# the bundled one so the sim stays usable on a machine without AIWaiter.
#
# Only one may run: both set SO_REUSEADDR, so two bridges bind port 45455
# without any error and the one that bound last silently receives everything.
set -eo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIWAITER_DIR="${AIWAITER_DIR:-$(cd "$DEMO_DIR/.." && pwd)/AIWaiter}"

source /opt/ros/jazzy/setup.bash

if [[ "${WAREHOUSE_PREFER_AIWAITER_BRIDGE:-true}" == "true" \
      && -f "$AIWAITER_DIR/src/robot_link/bridge.py" ]]; then
  echo "[bridge] AIWaiter: $AIWAITER_DIR/src/robot_link/bridge.py"
  cd "$AIWAITER_DIR"
  exec python3 -u -m src.robot_link.bridge --demo-dir "$DEMO_DIR" "$@"
fi

echo "[bridge] built-in: $DEMO_DIR/scripts/udp_command_bridge.py"
exec python3 -u "$DEMO_DIR/scripts/udp_command_bridge.py" \
  --demo-dir "$DEMO_DIR" "$@"
