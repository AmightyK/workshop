#!/usr/bin/env bash
# Drive the longest cabinet route through both moving-worker crossings.

set -eo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DYNAMIC DEMO] Driving to Storage C through worker crossings at (7,-10) and (-2,-5)."
echo "[DYNAMIC DEMO] LiDAR, Nav2 obstacle layers and collision monitor remain active."
exec "$DEMO_DIR/pick_box.sh" --storage C --route-only "$@"
