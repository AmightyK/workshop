#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$DEMO_DIR/.." && pwd)"
VJEPA_DIR="$WORKSPACE_DIR/vjepa_visual_localization"
VJEPA_MAP="${WAREHOUSE_VJEPA_MAP:-$VJEPA_DIR/outputs/autonomous_map_dense}"
VJEPA_CONFIG="${WAREHOUSE_VJEPA_CONFIG:-$VJEPA_DIR/configs/warehouse_experiment.yaml}"
NAV_LOCALIZATION_SOURCE="${WAREHOUSE_NAV_LOCALIZATION_SOURCE:-ground_truth}"
# At 1.0 m/s the short differential base cuts the tight obstacle curve shown
# on the planning dashboard. 0.85 m/s remains brisk while leaving angular
# authority to stay on the NavFn path through that corner.
NAV_MAX_LINEAR_SPEED="${WAREHOUSE_NAV_MAX_LINEAR_SPEED:-1.35}"
PEOPLE_SPEED_SCALE="${WAREHOUSE_PEOPLE_SPEED_SCALE:-0.70}"
LOG_DIR="${WAREHOUSE_LOG_DIR:-/tmp/warehouse_agv_demo}"
DASHBOARD_REFRESH_HZ="${WAREHOUSE_DASHBOARD_REFRESH_HZ:-32}"
DASHBOARD_MAP_REFRESH_HZ="${WAREHOUSE_DASHBOARD_MAP_REFRESH_HZ:-5}"
DASHBOARD_ODOM_PROJECTION="${WAREHOUSE_VJEPA_ODOM_PROJECTION:-false}"
VJEPA_IMAGE_TOPIC="${WAREHOUSE_VJEPA_IMAGE_TOPIC:-/vjepa/camera/image_raw}"
DASHBOARD_CAMERA_TOPIC="${WAREHOUSE_DASHBOARD_CAMERA_TOPIC:-/camera}"
VJEPA_IMAGE_FPS="${WAREHOUSE_VJEPA_IMAGE_FPS:-32.0}"
VJEPA_LOCALIZER_ENABLED="${WAREHOUSE_VJEPA_LOCALIZER:-true}"

# Keep this demo isolated from stale Gazebo discovery sessions. The bridge uses
# the same default partition, while still allowing an explicit override.
export GZ_PARTITION="${GZ_PARTITION:-warehouse_agv_demo}"
export GZ_SIM_RESOURCE_PATH="$DEMO_DIR/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
mkdir -p "$LOG_DIR"

# Two world_demo servers in one partition publish conflicting /clock and TF.
# Refuse to stack a new simulation on a stale server: this otherwise makes the
# dashboard show the previous robot pose and V-JEPA repeatedly reset its clip.
if gz topic -l 2>/dev/null | grep -Fx "/world/world_demo/pose/info" >/dev/null; then
  echo "[STARTUP ERROR] world_demo already exists in GZ_PARTITION=$GZ_PARTITION" >&2
  echo "Stop the old warehouse demo before running ./run_demo.sh again." >&2
  exit 2
fi

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'
  CYAN=$'\033[36m'
  GREEN=$'\033[32m'
  YELLOW=$'\033[33m'
  RESET=$'\033[0m'
else
  BOLD="" CYAN="" GREEN="" YELLOW="" RESET=""
fi

printf '%s\n' "${CYAN}${BOLD}╭────────────────────────────────────────────────────────────╮${RESET}"
printf '%s\n' "${CYAN}${BOLD}│          WAREHOUSE AGV · NAV2 · V-JEPA LIVE               │${RESET}"
printf '%s\n' "${CYAN}${BOLD}╰────────────────────────────────────────────────────────────╯${RESET}"
printf '%s\n' "  ${GREEN}●${RESET} Gazebo partition : $GZ_PARTITION"
printf '%s\n' "  ${GREEN}●${RESET} Dynamic workers  : 5 (2 cross the pick routes)"
printf '%s\n' "  ${GREEN}●${RESET} Motion speeds    : AGV ${NAV_MAX_LINEAR_SPEED} m/s · workers scale ${PEOPLE_SPEED_SCALE}x"
printf '%s\n' "  ${GREEN}●${RESET} Camera            : 640×360 · 16:9 · 32 FPS"
printf '%s\n' "  ${GREEN}●${RESET} V-JEPA clip       : rolling 1.0 s · 4 FPS · 4 newest frames"
printf '%s\n' "  ${GREEN}●${RESET} DDS image topic   : ${VJEPA_IMAGE_TOPIC} · ${VJEPA_IMAGE_FPS} FPS"
printf '%s\n' "  ${GREEN}●${RESET} DDS Orin return  : /vjepa_pose · /vjepa_latent · /vjepa_localization/debug"
printf '%s\n' "  ${GREEN}●${RESET} DDS domain/RMW    : ${ROS_DOMAIN_ID:-0} · ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
printf '%s\n' "  ${GREEN}●${RESET} Global planner     : NavFn A* once/goal + LiDAR stop/resume"
if is_enabled "$DASHBOARD_ODOM_PROJECTION"; then
  printf '%s\n' "  ${YELLOW}●${RESET} Odom projection    : telemetry only (explicitly enabled)"
else
  printf '%s\n' "  ${GREEN}●${RESET} Odom projection    : off · raw camera-only V-JEPA"
fi
if [[ "$NAV_LOCALIZATION_SOURCE" == "vjepa" ]]; then
  printf '%s\n' "  ${YELLOW}●${RESET} V-JEPA role        : experimental Nav2 localization control"
else
  printf '%s\n' "  ${GREEN}●${RESET} V-JEPA role        : live shadow comparison against truth"
fi

CHILD_PIDS=()
PID_FILES=()
PARTITION_KEY="${GZ_PARTITION//[^a-zA-Z0-9_.-]/_}"

start_component() {
  local name="$1"
  local pid_file="$2"
  shift 2
  # Each component owns a process group so launch/uv wrapper descendants are
  # stopped together. This prevents stale Nav2 lifecycle nodes surviving a
  # demo restart and racing the next bt_navigator startup.
  setsid "$@" >"$LOG_DIR/$name.log" 2>&1 &
  local process_pid=$!
  CHILD_PIDS+=("$process_pid")
  if [[ -n "$pid_file" ]]; then
    printf '%s\n' "$process_pid" > "$pid_file"
    PID_FILES+=("$pid_file")
  fi
  printf '%s\n' "  ${GREEN}●${RESET} $name started (log: $LOG_DIR/$name.log)"
}

# Random LiDAR-visible workers are a separate process so the world remains
# simple and the behavior can be paused while rebuilding the static map.
start_component random_people "" \
  python3 -u "$DEMO_DIR/scripts/random_people.py" \
  --speed-scale "$PEOPLE_SPEED_SCALE"

# run_demo is now the integrated entry point. The bridge starts here so the
# camera reaches V-JEPA immediately; pick_box only starts a fallback component
# if this integrated stack was explicitly disabled.
if is_enabled "${WAREHOUSE_AUTOSTART_BRIDGE:-true}"; then
  start_component ros_bridge "" "$DEMO_DIR/run_bridge.sh"
fi

# A latest-frame worker republishes the Gazebo image through a dedicated ROS 2
# DDS topic. It stays active when the local GPU process is disabled, allowing a
# Jetson Orin connected by Ethernet to own V-JEPA inference.
if is_enabled "${WAREHOUSE_VJEPA_IMAGE_RELAY:-true}"; then
  start_component vjepa_image_dds_relay "" \
    "$VJEPA_DIR/run_vjepa_image_relay.sh" \
    --input-topic /camera \
    --output-topic "$VJEPA_IMAGE_TOPIC" \
    --fps "$VJEPA_IMAGE_FPS" \
    --ros-args -p use_sim_time:=true
fi

VJEPA_PID_FILE="$LOG_DIR/${PARTITION_KEY}_vjepa.pid"
DASHBOARD_PID_FILE="$LOG_DIR/${PARTITION_KEY}_dashboard.pid"
VJEPA_READY=false
if is_enabled "${WAREHOUSE_VJEPA_ENABLED:-true}"; then
  if [[ -f "$VJEPA_MAP/global_embeddings.npy" && -f "$VJEPA_MAP/poses.npy" ]]; then
    if is_enabled "$VJEPA_LOCALIZER_ENABLED"; then
      start_component vjepa_temporal "$VJEPA_PID_FILE" \
        "$VJEPA_DIR/run_live_localization.sh" \
        --config "$VJEPA_CONFIG" --map "$VJEPA_MAP" \
        --camera-topic "$VJEPA_IMAGE_TOPIC" \
        --ros-args -p use_sim_time:=true
    else
      printf '%s\n' "  ${YELLOW}●${RESET} V-JEPA localizer  : off · waiting for Orin DDS output"
    fi
    VJEPA_READY=true
    if is_enabled "${WAREHOUSE_VJEPA_DASHBOARD:-true}" && [[ -n "${DISPLAY:-}" ]]; then
      DASHBOARD_ODOM_ARGS=()
      if is_enabled "$DASHBOARD_ODOM_PROJECTION"; then
        DASHBOARD_ODOM_ARGS+=(--odom-projection)
      fi
      start_component vjepa_dashboard "$DASHBOARD_PID_FILE" \
        "$VJEPA_DIR/run_localization_dashboard.sh" \
        --latent-map "$VJEPA_MAP" \
        --camera-topic "$DASHBOARD_CAMERA_TOPIC" \
        --refresh-hz "$DASHBOARD_REFRESH_HZ" \
        --map-refresh-hz "$DASHBOARD_MAP_REFRESH_HZ" \
        "${DASHBOARD_ODOM_ARGS[@]}" \
        --ros-args -p use_sim_time:=true
      printf '%s\n' "  ${GREEN}●${RESET} V-JEPA windows    : camera/QA/latent + warehouse map"
    else
      printf '%s\n' "  ${YELLOW}●${RESET} V-JEPA windows    : disabled or DISPLAY unavailable"
    fi
  else
    printf '%s\n' "  ${YELLOW}●${RESET} V-JEPA            : latent map missing at $VJEPA_MAP"
  fi
fi

# Ground truth remains the default Nav2 localization reference for repeatable
# pickup. V-JEPA owns map->odom only in the explicitly experimental mode;
# dashboard odometry projection is a separate, optional telemetry feature.
if is_enabled "${WAREHOUSE_AUTOSTART_NAV2:-true}"; then
  if [[ "$NAV_LOCALIZATION_SOURCE" == "vjepa" ]] && ! "$VJEPA_READY"; then
    printf '%s\n' "  ${YELLOW}●${RESET} Nav2              : not started (V-JEPA latent map unavailable)"
  else
    NAV2_RVIZ="${WAREHOUSE_NAV2_RVIZ:-true}"
    if [[ -z "${DISPLAY:-}" ]]; then
      NAV2_RVIZ=false
    fi
    start_component "nav2_${NAV_LOCALIZATION_SOURCE}" "" \
      "$DEMO_DIR/run_nav2.sh" \
      "use_rviz:=$NAV2_RVIZ" \
      "localization_source:=$NAV_LOCALIZATION_SOURCE" \
      "max_linear_speed:=$NAV_MAX_LINEAR_SPEED"
    if [[ "$NAV_LOCALIZATION_SOURCE" == "ground_truth" ]]; then
      printf '%s\n' "  ${GREEN}●${RESET} Nav localization  : Gazebo truth reference"
      printf '%s\n' "  ${GREEN}●${RESET} V-JEPA output      : shadow compare only (no steering)"
    else
      printf '%s\n' "  ${GREEN}●${RESET} Nav localization  : V-JEPA → map→odom (control active)"
    fi
  fi
fi

cleanup() {
  local process_pid
  for process_pid in "${CHILD_PIDS[@]}"; do
    kill -TERM -- "-$process_pid" 2>/dev/null \
      || kill -TERM "$process_pid" 2>/dev/null \
      || true
  done
  for process_pid in "${CHILD_PIDS[@]}"; do
    wait "$process_pid" 2>/dev/null || true
  done
  local pid_file
  for pid_file in "${PID_FILES[@]}"; do
    rm -f -- "$pid_file"
  done
}
trap cleanup EXIT INT TERM

# DetachableJoint starts attached by design. Keep physics paused until every
# selectable carton is detached, otherwise startup can drag boxes off shelves.
(
  TASK_BOXES=(
    special_blue_box distractor_red_A02 distractor_green_A03
    distractor_blue_B01 special_red_box distractor_green_B03
    distractor_blue_C01 distractor_red_C02 special_green_box
  )
  for _ in $(seq 1 100); do
    ALL_READY=true
    for MODEL_NAME in "${TASK_BOXES[@]}"; do
      if ! gz topic -l 2>/dev/null | grep -Fx "/warehouse_agv/gripper/$MODEL_NAME/detach" >/dev/null; then
        ALL_READY=false
        break
      fi
    done
    if "$ALL_READY"; then
      # A DetachableJoint creates its fixed constraint during the first world
      # update.  A single detach message sent while physics is paused can race
      # that creation and leave the fork constrained to a carton on a shelf.
      # Repeat the command around small paused world steps so every plugin has
      # both created and removed its initial constraint before normal physics.
      for _DETACH_ROUND in $(seq 1 4); do
        DETACH_PIDS=()
        for MODEL_NAME in "${TASK_BOXES[@]}"; do
          gz topic -t "/warehouse_agv/gripper/$MODEL_NAME/detach" \
            -m gz.msgs.Empty -p "unused: true" &
          DETACH_PIDS+=("$!")
        done
        for _DETACH_PID in "${DETACH_PIDS[@]}"; do
          wait "$_DETACH_PID"
        done
        gz service -s /world/world_demo/control \
          --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
          --timeout 3000 --req 'pause: true, multi_step: 2' >/dev/null
        sleep 0.05
      done
      # Resume only after the final detach command has crossed Transport and
      # been consumed by the systems on a paused simulation update.
      gz service -s /world/world_demo/control \
        --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
        --timeout 3000 --req 'pause: false' >/dev/null
      exit 0
    fi
    sleep 0.1
  done
  echo "Failed to detach all selectable cartons before starting physics" >&2
) &

# Official warehouse layout with the custom camera-equipped AMR and task layer.
gz sim "$DEMO_DIR/worlds/tugbot_warehouse_custom.sdf" "$@"
