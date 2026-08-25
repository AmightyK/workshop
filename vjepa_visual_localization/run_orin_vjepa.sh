#!/usr/bin/env bash
# Run this on Jetson Orin after the latent map and repository have been copied once.
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VJEPA_CONFIG="${ORIN_VJEPA_CONFIG:-$PROJECT_DIR/configs/warehouse_experiment.yaml}"
VJEPA_MAP="${ORIN_VJEPA_MAP:-$PROJECT_DIR/outputs/autonomous_map_dense}"
VJEPA_IMAGE_TOPIC="${WAREHOUSE_VJEPA_IMAGE_TOPIC:-/vjepa/camera/image_raw}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

exec "$PROJECT_DIR/run_live_localization.sh" \
  --config "$VJEPA_CONFIG" \
  --map "$VJEPA_MAP" \
  --camera-topic "$VJEPA_IMAGE_TOPIC" \
  "$@" \
  --ros-args -p use_sim_time:=true
