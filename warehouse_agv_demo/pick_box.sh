#!/usr/bin/env bash
# One-command physical pickup for any RGB carton in Storage A, B or C.

set -Eeo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$DEMO_DIR/.." && pwd)"
VJEPA_DIR="$WORKSPACE_DIR/vjepa_visual_localization"
VJEPA_MAP="${WAREHOUSE_VJEPA_MAP:-$VJEPA_DIR/outputs/autonomous_map_dense}"
VJEPA_CONFIG="${WAREHOUSE_VJEPA_CONFIG:-$VJEPA_DIR/configs/warehouse_experiment.yaml}"
LOG_DIR="${WAREHOUSE_LOG_DIR:-/tmp/warehouse_agv_demo}"
DASHBOARD_REFRESH_HZ="${WAREHOUSE_DASHBOARD_REFRESH_HZ:-32}"
DASHBOARD_MAP_REFRESH_HZ="${WAREHOUSE_DASHBOARD_MAP_REFRESH_HZ:-5}"
VJEPA_IMAGE_TOPIC="${WAREHOUSE_VJEPA_IMAGE_TOPIC:-/vjepa/camera/image_raw}"
DASHBOARD_CAMERA_TOPIC="${WAREHOUSE_DASHBOARD_CAMERA_TOPIC:-/camera}"
VJEPA_IMAGE_FPS="${WAREHOUSE_VJEPA_IMAGE_FPS:-32.0}"
DASHBOARD_ODOM_ARGS=()
case "${WAREHOUSE_VJEPA_ODOM_PROJECTION:-false}" in
  1|true|TRUE|yes|YES|on|ON) DASHBOARD_ODOM_ARGS+=(--odom-projection) ;;
esac
export GZ_PARTITION="${GZ_PARTITION:-warehouse_agv_demo}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
source /opt/ros/jazzy/setup.bash
set -u
mkdir -p "$LOG_DIR"
# A pick invocation is one auditable mission. Planner and state publishers may
# already be alive under run_demo, so truncate only append-oriented JSON logs;
# the continuously open trajectory stream is copied separately per trial.
truncate -s 0 "$LOG_DIR/mission_states.jsonl"
truncate -s 0 "$LOG_DIR/behavior_decisions.jsonl"

STORAGE="A"
COLOR="blue"
DELIVER=true
ROUTE_ONLY=false
RESUME_DELIVERY=false
DRY_RUN=false
BRIDGE_PID=""
IMAGE_RELAY_PID=""
NAV2_PID=""
VJEPA_PID=""
PREDICTION_PID=""
DASHBOARD_PID=""
VJEPA_EXPECTED=false

if [[ -t 1 ]]; then
  BOLD=$'\033[1m' CYAN=$'\033[36m' GREEN=$'\033[32m'
  YELLOW=$'\033[33m' RED=$'\033[31m' RESET=$'\033[0m'
else
  BOLD="" CYAN="" GREEN="" YELLOW="" RED="" RESET=""
fi

usage() {
  cat <<'EOF'
Usage: ./pick_box.sh [--storage A|B|C] [--color red|blue|green] [--pick-only] [--resume-delivery] [--route-only] [--dry-run]

Options:
  --storage, --area, -s   Cabinet / area A, B or C (default: A)
  --color, -c             Box color red, blue or green (default: blue)
  --deliver               Pick and return to Packing Station (default)
  --pick-only             Stop with the box attached on the lowered tray
  --resume-delivery       Box is already attached; return and drop without repicking
  --route-only            Follow the recorded outbound corridor; do not pick
  --dry-run               Validate storage/color and print the route without ROS
  -h, --help              Show this help

The AGV follows checkpoints sampled from the saved mapping traversal. The color
is resolved at cabinet staging, then it continues the same latent corridor to
the slot, then follows the recorded return leg to Packing by default.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --storage|--area|-s)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      STORAGE="$2"
      shift 2
      ;;
    --color|-c)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      COLOR="$2"
      shift 2
      ;;
    --deliver)
      DELIVER=true
      shift
      ;;
    --pick-only)
      DELIVER=false
      shift
      ;;
    --resume-delivery)
      RESUME_DELIVERY=true
      DELIVER=true
      shift
      ;;
    --route-only)
      ROUTE_ONLY=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
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

STORAGE="${STORAGE^^}"
COLOR="${COLOR,,}"
case "$STORAGE" in A|B|C) ;; *) echo "--storage must be A, B or C" >&2; exit 2 ;; esac
case "$COLOR" in red|blue|green) ;; *) echo "--color must be red, blue or green" >&2; exit 2 ;; esac
if "$RESUME_DELIVERY" && "$ROUTE_ONLY"; then
  echo "--resume-delivery cannot be combined with --route-only" >&2
  exit 2
fi

stop_process_group() {
  local leader_pid="$1"
  [[ -z "$leader_pid" ]] && return 0
  kill -TERM -- "-$leader_pid" 2>/dev/null || true
  wait "$leader_pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT
  stop_process_group "$DASHBOARD_PID"
  stop_process_group "$PREDICTION_PID"
  stop_process_group "$VJEPA_PID"
  stop_process_group "$NAV2_PID"
  stop_process_group "$IMAGE_RELAY_PID"
  stop_process_group "$BRIDGE_PID"
  if (( status == 130 )); then
    echo "${YELLOW}[PICK BOX] Mission cancelled by user.${RESET}" >&2
  elif (( status != 0 )); then
    echo "${RED}[PICK BOX] Failed. Check $LOG_DIR and /tmp/warehouse_pick_box_*.log${RESET}" >&2
  elif [[ -n "$BRIDGE_PID$NAV2_PID" ]]; then
    echo "[PICK BOX] Temporary bridge/Nav2 stopped; Gazebo remains open."
  fi
  exit "$status"
}
trap cleanup EXIT

wait_for_ros_name() {
  local kind="$1"
  local name="$2"
  local timeout_s="$3"
  local started=$SECONDS
  while (( SECONDS - started < timeout_s )); do
    if [[ "$kind" == "topic" ]]; then
      # Consume the complete ros2 output. With pipefail, grep -q can close the
      # pipe early and make ros2 raise BrokenPipe even though the name exists.
      ros2 topic list 2>/dev/null | grep -Fx "$name" >/dev/null && return 0
    else
      ros2 action list 2>/dev/null | grep -Fx "$name" >/dev/null && return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for ROS $kind: $name" >&2
  return 1
}

wait_for_lifecycle_active() {
  local node_name="$1"
  local timeout_s="$2"
  local started=$SECONDS
  while (( SECONDS - started < timeout_s )); do
    if ros2 lifecycle get "$node_name" 2>/dev/null | grep -q '^active ' \
      || timeout 2 ros2 service call "$node_name/get_state" \
        lifecycle_msgs/srv/GetState '{}' 2>/dev/null | grep -q "label='active'"; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for active lifecycle node: $node_name" >&2
  return 1
}

pid_file_is_alive() {
  local pid_file="$1"
  local recorded_pid=""
  [[ -r "$pid_file" ]] || return 1
  read -r recorded_pid < "$pid_file" || return 1
  [[ "$recorded_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$recorded_pid" 2>/dev/null
}

printf '%s\n' "${CYAN}${BOLD}╭──────────────────────────────────────────────╮${RESET}"
printf '%s\n' "${CYAN}${BOLD}│          WAREHOUSE RGB PICK MISSION          │${RESET}"
printf '%s\n' "${CYAN}${BOLD}╰──────────────────────────────────────────────╯${RESET}"
printf '%s\n' "  ${GREEN}●${RESET} Storage : $STORAGE"
printf '%s\n' "  ${GREEN}●${RESET} Color   : $COLOR"
printf '%s\n' "  ${GREEN}●${RESET} Deliver : $DELIVER"
printf '%s\n' "  ${GREEN}●${RESET} Resume  : $RESUME_DELIVERY"

if "$DRY_RUN"; then
  MISSION_ARGS=(--storage "$STORAGE" --color "$COLOR" --dry-run)
  if "$DELIVER"; then
    MISSION_ARGS+=(--deliver)
  else
    MISSION_ARGS+=(--pick-only)
  fi
  if "$ROUTE_ONLY"; then
    MISSION_ARGS+=(--route-only)
  fi
  if "$RESUME_DELIVERY"; then
    MISSION_ARGS+=(--resume-delivery)
  fi
  exec "$DEMO_DIR/run_storage_pick.sh" "${MISSION_ARGS[@]}"
fi

if ! gz topic -l 2>/dev/null | grep -Fx "/warehouse_agv/camera" >/dev/null; then
  echo "[PICK BOX] Gazebo is not running. Start ./run_demo.sh first." >&2
  exit 1
fi

# Workers patrol continuously across missions.  Resetting their physical poses
# here used to queue a Gazebo service request while cmd_vel was already live;
# the late request then snapped workers backward and made them appear frozen.
PEOPLE_RESET_TOPIC="/warehouse/random_people/reset"
PEOPLE_MISSION_TOPIC="/warehouse/random_people/new_mission"
if gz topic -l 2>/dev/null | grep -Fx "$PEOPLE_MISSION_TOPIC" >/dev/null; then
  # A single short-lived gz publisher can exit before Transport discovery has
  # delivered its sample. Send a small burst with one shared unique ID;
  # random_people.py treats every copy as the same idempotent mission.
  PEOPLE_MISSION_ID="$(date +%s%N)"
  for _MISSION_TRIGGER_COPY in 1 2 3 4 5; do
    gz topic -t "$PEOPLE_MISSION_TOPIC" -m gz.msgs.Int64 \
      -p "data: $PEOPLE_MISSION_ID"
    sleep 0.10
  done
  echo "${GREEN}[WORKERS] Crossing direction armed for this mission; poses retained.${RESET}"
fi
if [[ "${WAREHOUSE_RESET_PEOPLE_ON_MISSION:-false}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  gz topic -t "$PEOPLE_RESET_TOPIC" -m gz.msgs.Boolean -p 'data: true'
  echo "${YELLOW}[WORKERS] Physical patrol reset requested by environment override.${RESET}"
else
  echo "${GREEN}[WORKERS] Continuous patrol retained; no mission-time pose reset.${RESET}"
fi

# Fast path for the normal interactive workflow: run_demo.sh already started
# every dependency. The old fallback below intentionally remains for starting
# pick_box.sh on a cold machine, but repeating its topic/action discovery on
# every manual pickup was the source of the visible ~20 s delay.
if [[ "${WAREHOUSE_PICK_FAST_START:-true}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  FAST_START_READY=true
  # One ROS CLI process is noticeably cheaper than five separate topic/action
  # discovery calls (each can take 1–2 s on a busy DDS graph).  The action list
  # is the useful readiness signal here: both Nav2 servers exist only after
  # bt_navigator has created its active action endpoints.  The mission still
  # performs its lifecycle check in Python before sending the goal.
  FAST_ACTIONS="$(ros2 action list 2>/dev/null || true)"
  grep -Fx "/navigate_to_pose" <<<"$FAST_ACTIONS" >/dev/null || FAST_START_READY=false
  grep -Fx "/navigate_through_poses" <<<"$FAST_ACTIONS" >/dev/null || FAST_START_READY=false
  if "$FAST_START_READY"; then
    FAST_MISSION_ARGS=(--storage "$STORAGE" --color "$COLOR")
    if "$DELIVER"; then
      FAST_MISSION_ARGS+=(--deliver)
    else
      FAST_MISSION_ARGS+=(--pick-only)
    fi
    if "$ROUTE_ONLY"; then
      FAST_MISSION_ARGS+=(--route-only)
    fi
    if "$RESUME_DELIVERY"; then
      FAST_MISSION_ARGS+=(--resume-delivery)
    fi
    echo "${GREEN}[PICK BOX] Fast start: Gazebo/Nav2 đã sẵn sàng → chạy mission ngay.${RESET}"
    exec "$DEMO_DIR/run_storage_pick.sh" "${FAST_MISSION_ARGS[@]}"
  fi
fi

BRIDGE_READY=false
for _ in $(seq 1 16); do
  if ros2 topic list 2>/dev/null | grep -Fx "/camera" >/dev/null; then
    BRIDGE_READY=true
    break
  fi
  sleep 0.5
done
if ! "$BRIDGE_READY"; then
  echo "[PICK BOX] Starting Gazebo <-> ROS 2 bridge..."
  setsid "$DEMO_DIR/run_bridge.sh" > /tmp/warehouse_pick_box_bridge.log 2>&1 &
  BRIDGE_PID=$!
fi
wait_for_ros_name topic /camera 30
wait_for_ros_name topic /scan 30

# Start Nav2 as soon as the Gazebo bridge exposes the robot sensors.  V-JEPA
# and the dashboard can warm in parallel; starting them first made a cold
# manual `pick_box.sh` invocation wait through several independent startup
# sequences before Nav2 could accept the first goal.
if ! ros2 action list 2>/dev/null | grep -Fx "/navigate_to_pose" >/dev/null; then
  echo "[PICK BOX] Starting Nav2 while camera/V-JEPA dependencies warm..."
  setsid "$DEMO_DIR/run_nav2.sh" use_rviz:=True \
    > /tmp/warehouse_pick_box_nav2.log 2>&1 &
  NAV2_PID=$!
fi

if ! ros2 topic list 2>/dev/null | grep -Fx "$VJEPA_IMAGE_TOPIC" >/dev/null; then
  echo "[PICK BOX] Starting threaded ROS 2 DDS image relay..."
  setsid "$VJEPA_DIR/run_vjepa_image_relay.sh" \
    --input-topic /camera \
    --output-topic "$VJEPA_IMAGE_TOPIC" \
    --fps "$VJEPA_IMAGE_FPS" \
    --ros-args -p use_sim_time:=true > /tmp/warehouse_pick_box_image_relay.log 2>&1 &
  IMAGE_RELAY_PID=$!
fi
wait_for_ros_name topic "$VJEPA_IMAGE_TOPIC" 30

# run_demo normally owns these two processes. This fallback keeps pick_box
# self-contained when V-JEPA autostart was disabled or an older demo is used.
PARTITION_KEY="${GZ_PARTITION//[^a-zA-Z0-9_.-]/_}"
VJEPA_PID_FILE="$LOG_DIR/${PARTITION_KEY}_vjepa.pid"
DASHBOARD_PID_FILE="$LOG_DIR/${PARTITION_KEY}_dashboard.pid"
if [[ -f "$VJEPA_MAP/global_embeddings.npy" && -f "$VJEPA_MAP/poses.npy" ]]; then
  VJEPA_EXPECTED=true
  if ! pid_file_is_alive "$VJEPA_PID_FILE" \
    && ! ros2 node list 2>/dev/null | grep -Fx "/vjepa_visual_localizer" >/dev/null; then
    echo "${YELLOW}[V-JEPA] Starting temporal localizer for this pick mission...${RESET}"
    setsid "$VJEPA_DIR/run_live_localization.sh" \
      --config "$VJEPA_CONFIG" --map "$VJEPA_MAP" \
      --camera-topic "$VJEPA_IMAGE_TOPIC" \
      --ros-args -p use_sim_time:=true > "$LOG_DIR/pick_vjepa_temporal.log" 2>&1 &
    VJEPA_PID=$!
  else
    echo "${GREEN}[V-JEPA] Temporal localizer is active.${RESET}"
  fi
  if ! ros2 node list 2>/dev/null \
    | grep -Fx "/vjepa_latent_prediction_monitor" >/dev/null; then
    echo "${YELLOW}[V-JEPA] Starting asynchronous z(t+1..3) evaluator...${RESET}"
    setsid "$VJEPA_DIR/run_latent_prediction_monitor.sh" \
      --config "$VJEPA_CONFIG" \
      --ros-args -p use_sim_time:=true \
      > "$LOG_DIR/pick_vjepa_latent_prediction.log" 2>&1 &
    PREDICTION_PID=$!
  else
    echo "${GREEN}[V-JEPA] Latent prediction logger is active.${RESET}"
  fi
  if [[ -n "${DISPLAY:-}" ]] \
    && ! pid_file_is_alive "$DASHBOARD_PID_FILE" \
    && ! ros2 node list 2>/dev/null | grep -Fx "/vjepa_localization_dashboard" >/dev/null; then
    echo "${YELLOW}[V-JEPA] Opening camera/QA/latent and warehouse-map windows...${RESET}"
    setsid "$VJEPA_DIR/run_localization_dashboard.sh" \
      --latent-map "$VJEPA_MAP" \
      --camera-topic "$DASHBOARD_CAMERA_TOPIC" \
      --refresh-hz "$DASHBOARD_REFRESH_HZ" \
      --map-refresh-hz "$DASHBOARD_MAP_REFRESH_HZ" \
      "${DASHBOARD_ODOM_ARGS[@]}" \
      --ros-args -p use_sim_time:=true > "$LOG_DIR/pick_vjepa_dashboard.log" 2>&1 &
    DASHBOARD_PID=$!
  fi
else
  echo "${YELLOW}[V-JEPA] Latent map missing: $VJEPA_MAP${RESET}"
fi

# Loading V-JEPA2 on the GPU takes noticeably longer than bringing up Gazebo.
# Do not let a pick route leave the known dock before the first camera-only
# estimate has established its route-start prior.
if "$VJEPA_EXPECTED" && ! "$RESUME_DELIVERY"; then
  NAV_NODES="$(ros2 node list 2>/dev/null || true)"
  if [[ "$NAV_NODES" == *"/gazebo_ground_truth_localizer"* ]]; then
    # V-JEPA is dashboard-only in the stable demo. Do not consume 15–20 s of
    # the 90-second pick show waiting for a shadow signal that never steers the
    # robot; it can finish warming while Nav2 starts along the retained route.
    echo "${CYAN}[V-JEPA] Shadow model warming in parallel; Nav2 may depart now.${RESET}"
  else
    echo "${CYAN}[V-JEPA] Warming model and anchoring the first latent at the dock...${RESET}"
    wait_for_ros_name topic /vjepa_pose 90
    # Topic discovery happens as soon as the publisher is created, while the
    # first V-JEPA2 CUDA warm-up may still need close to a minute. Wait for an
    # actual pose sample only when V-JEPA owns map->odom.
    if ! timeout 100 ros2 topic echo /vjepa_pose --once >/dev/null 2>&1; then
      echo "[V-JEPA] No initial pose was published; refusing to start an invalid control run." >&2
      exit 1
    fi
    echo "${GREEN}[V-JEPA] Initial temporal pose is ready; starting the recorded corridor.${RESET}"
  fi
fi

wait_for_ros_name action /navigate_to_pose 60
wait_for_lifecycle_active /bt_navigator 60
wait_for_ros_name topic /nav/localization_status 60
NAV_NODES="$(ros2 node list 2>/dev/null || true)"
NAV_STATUS="$(timeout 6 ros2 topic echo /nav/localization_status --once --field data 2>/dev/null || true)"
if [[ "$NAV_NODES" == *"/gazebo_ground_truth_localizer"* ]] \
    || [[ "$NAV_STATUS" == *"GAZEBO_TRUTH_REFERENCE"* ]]; then
  export WAREHOUSE_NAV_LOCALIZATION_SOURCE=ground_truth
  echo "${GREEN}[NAV] Control: Gazebo truth reference | Planner: NavFn A*${RESET}"
  echo "${CYAN}[V-JEPA] Camera-only shadow mode: đang so sánh, không điều khiển xe.${RESET}"
elif [[ "$NAV_NODES" == *"/vjepa_nav_localizer"* ]] \
    || [[ "$NAV_STATUS" == *'"source": "VJEPA'* ]]; then
  export WAREHOUSE_NAV_LOCALIZATION_SOURCE=vjepa
  echo "${GREEN}[NAV] Control: V-JEPA camera + odom | Planner: NavFn A*${RESET}"
  echo "${CYAN}[TRUTH] Gazebo GPS chỉ dùng trong dashboard để đối chiếu.${RESET}"
else
  echo "[PICK BOX] Cannot identify the active Nav2 localization source." >&2
  exit 1
fi

MISSION_ARGS=(--storage "$STORAGE" --color "$COLOR")
if "$DELIVER"; then
  MISSION_ARGS+=(--deliver)
else
  MISSION_ARGS+=(--pick-only)
fi
if "$ROUTE_ONLY"; then
  MISSION_ARGS+=(--route-only)
fi
if "$RESUME_DELIVERY"; then
  MISSION_ARGS+=(--resume-delivery)
fi
if "$RESUME_DELIVERY"; then
  echo "${CYAN}[DELIVERY RESUME] Payload already attached → recorded return corridor → Packing Station${RESET}"
elif "$ROUTE_ONLY"; then
  echo "${CYAN}[DYNAMIC DEMO] Storage=$STORAGE, route-only=true${RESET}"
else
  echo "${CYAN}[PICK BOX] Route first → resolve $COLOR at Storage $STORAGE → camera pick${RESET}"
fi
"$DEMO_DIR/run_storage_pick.sh" "${MISSION_ARGS[@]}"
if "$RESUME_DELIVERY"; then
  echo "${GREEN}${BOLD}[DELIVERY] Attached $COLOR box delivered from Storage $STORAGE to Packing Station.${RESET}"
elif "$ROUTE_ONLY"; then
  echo "${GREEN}[DYNAMIC DEMO] Route to Storage $STORAGE complete; no box was moved.${RESET}"
else
  if "$DELIVER"; then
    echo "${GREEN}${BOLD}[DELIVERY] Complete: $COLOR box from Storage $STORAGE is at Packing Station.${RESET}"
  else
    echo "${GREEN}${BOLD}[PICK BOX] Complete: $COLOR box remains attached on the AGV tray.${RESET}"
  fi
fi
