#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash

# ROS 2 topics below are transported by the selected DDS RMW implementation.
# Fast DDS is the deployment default on both the Gazebo PC and Jetson Orin.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
exec /usr/bin/python3 -u "$PROJECT_DIR/scripts/vjepa_image_relay.py" "$@"
