#!/usr/bin/env bash
# Backward-compatible entry point. With no flags this picks A/blue and delivers;
# --storage/--area and --color now expose the generalized picker.

set -eo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DEMO_DIR/pick_box.sh" "$@"
