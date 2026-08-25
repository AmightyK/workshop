#!/usr/bin/env bash
# Run one or two fixed Nav2 routes while run_demo owns the live VL-JEPA UI.

set -Eeo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTE="both"

if [[ -t 1 ]]; then
  BOLD=$'\033[1m' CYAN=$'\033[36m' GREEN=$'\033[32m'
  YELLOW=$'\033[33m' RESET=$'\033[0m'
else
  BOLD="" CYAN="" GREEN="" YELLOW="" RESET=""
fi

usage() {
  cat <<'EOF'
Usage: ./run_vljepa_showcase.sh [--route short|long|both]

  short   Dock -> Storage A (one worker crossing)
  long    Dock -> Storage C (two worker crossings)
  both    Run the short route, return through the dock corridor, then run long

Start ./run_demo.sh first. Its streaming window continuously rotates the eight
prepared questions and draws the live V-JEPA latent projection during motion.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --route)
      [[ $# -ge 2 ]] || { echo "Missing value for --route" >&2; exit 2; }
      ROUTE="${2,,}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$ROUTE" in
  short) STORAGES=(A) ;;
  long) STORAGES=(C) ;;
  both) STORAGES=(A C) ;;
  *) echo "--route must be short, long or both" >&2; exit 2 ;;
esac

printf '%s\n' "${CYAN}${BOLD}╭──────────────────────────────────────────────────────────╮${RESET}"
printf '%s\n' "${CYAN}${BOLD}│        VL-JEPA · FIXED ROUTE STREAMING SHOWCASE          │${RESET}"
printf '%s\n' "${CYAN}${BOLD}╰──────────────────────────────────────────────────────────╯${RESET}"
printf '%s\n' "  ${GREEN}●${RESET} Route preset : $ROUTE (${#STORAGES[@]} traversal(s))"
printf '%s\n' "  ${GREEN}●${RESET} Questions    : 8 prepared queries, rotated every 3.5 s"
printf '%s\n' "  ${GREEN}●${RESET} Visualization: live camera + real latent PCA + warehouse map"
printf '%s\n' "  ${GREEN}●${RESET} Avoidance    : A* + LiDAR + moving workers"

for index in "${!STORAGES[@]}"; do
  storage="${STORAGES[$index]}"
  printf '\n%s\n' "${YELLOW}${BOLD}▶ TRAVERSAL $((index + 1))/${#STORAGES[@]} · fixed route to Storage $storage${RESET}"
  "$DEMO_DIR/pick_box.sh" --storage "$storage" --route-only
done

printf '\n%s\n' "${GREEN}${BOLD}✓ Showcase route complete. No box was moved.${RESET}"
