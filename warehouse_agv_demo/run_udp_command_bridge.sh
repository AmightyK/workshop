#!/usr/bin/env bash
# Start the UDP command bridge that AIWaiter's voice machine talks to.
#
# Two bridges implement this protocol: AIWaiter's `src/robot_link/bridge.py`,
# which is what the voice machine's sender and `make say` are written against,
# and this repo's older `scripts/udp_command_bridge.py`. Prefer AIWaiter's when
# its checkout is present so a voice demo needs no second terminal; fall back to
# the bundled one so the sim stays usable on a machine without AIWaiter.
#
# Only one may run. Legacy receivers used SO_REUSEADDR, allowing multiple
# bridges on port 45455 and distributing commands across stale generations.
set -eo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIWAITER_DIR="${AIWAITER_DIR:-$(cd "$DEMO_DIR/.." && pwd)/AIWaiter}"
LOG_DIR="${WAREHOUSE_LOG_DIR:-/tmp/warehouse_agv_demo}"
BRIDGE_LOCK="$LOG_DIR/udp_command_bridge.lock"

source /opt/ros/jazzy/setup.bash

# One UDP receiver must own both the voice command stream and its child
# mission. SO_REUSEADDR previously allowed every demo restart to leave another
# receiver on port 45455, so different bridge generations launched competing
# Nav2 goals. Keep this descriptor locked across exec for the process lifetime.
mkdir -p "$LOG_DIR"
exec 9<>"$BRIDGE_LOCK"
if ! flock -n 9; then
  echo "[bridge] another UDP command bridge already owns $BRIDGE_LOCK" >&2
  exit 2
fi
# Let the integrated launcher replace an orphaned receiver from an earlier
# demo generation. The lock remains the authority; this PID is only a safe,
# validated cleanup hint for run_demo.sh.
printf '%s\n' "$$" >"$BRIDGE_LOCK"

if [[ "${WAREHOUSE_PREFER_AIWAITER_BRIDGE:-true}" == "true" \
      && -f "$AIWAITER_DIR/src/robot_link/bridge.py" ]]; then
  echo "[bridge] AIWaiter: $AIWAITER_DIR/src/robot_link/bridge.py"
  cd "$AIWAITER_DIR"
  exec python3 -u -m src.robot_link.bridge --demo-dir "$DEMO_DIR" "$@"
fi

echo "[bridge] built-in: $DEMO_DIR/scripts/udp_command_bridge.py"
exec python3 -u "$DEMO_DIR/scripts/udp_command_bridge.py" \
  --demo-dir "$DEMO_DIR" "$@"
