#!/usr/bin/env bash
set -eo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_PARTITION="${GZ_PARTITION:-warehouse_agv_demo}"

source /opt/ros/jazzy/setup.bash

# The integrated launcher starts this wrapper before `gz sim`. Subscribing the
# bridge before Gazebo creates the model plugins can miss the one initial odom
# / TF sample while the AGV is stationary, leaving Nav2 without an `odom`
# frame. Wait for the actual Gazebo publishers before constructing bridges.
REQUIRED_GZ_TOPICS=(
  /clock
  /model/warehouse_agv/odometry
  /model/warehouse_agv/tf
  /warehouse_agv/scan
)
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  topics="$(gz topic -l 2>/dev/null || true)"
  ready=true
  for topic in "${REQUIRED_GZ_TOPICS[@]}"; do
    if ! grep -Fxq "$topic" <<<"$topics"; then
      ready=false
      break
    fi
  done
  if "$ready"; then
    break
  fi
  sleep 0.1
done
if ! "$ready"; then
  echo "Gazebo bridge startup timed out waiting for model topics" >&2
  exit 1
fi

ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:="$DEMO_DIR/config/bridge.yaml"
