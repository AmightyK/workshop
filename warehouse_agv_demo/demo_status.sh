#!/usr/bin/env bash
# Report what the running demo is actually doing, component by component.
#
# run_demo.sh can only describe the moment it spawned each process. Components
# that pull dependencies on first run stay alive for minutes before they open a
# window, and a component that dies later leaves the startup banner stale. This
# reads live state instead: pid liveness, the phase implied by each log tail,
# the UDP bridge socket, and any error the logs recorded.
set -eo pipefail

LOG_DIR="${WAREHOUSE_LOG_DIR:-/tmp/warehouse_agv_demo}"
UDP_PORT="${WAREHOUSE_UDP_PORT:-45455}"

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
  RED=$'\033[31m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; RESET=""
fi

if [[ ! -d "$LOG_DIR" ]]; then
  printf '%s\n' "${RED}No $LOG_DIR - the demo has never been started.${RESET}"
  exit 1
fi

printf '%s\n' "${CYAN}${BOLD}Demo status${RESET}  ($LOG_DIR)"
printf '%s\n' "──────────────────────────────────────────────────────────────"

# Resolve a component's pid from the kernel rather than from a pid file: the
# process holding this log open on stdout is the component itself.
find_pid_by_log() {
  local target="$1" fd owner
  for fd in /proc/[0-9]*/fd/1; do
    owner="$(readlink "$fd" 2>/dev/null || true)"
    if [[ "$owner" == "$target" ]]; then
      printf '%s\n' "$fd" | cut -d/ -f3
      return 0
    fi
  done
  return 0
}

alive=0; installing=0; dead=0; errored=0

for log in "$LOG_DIR"/*.log; do
  [[ -e "$log" ]] || continue
  name="$(basename "$log" .log)"
  pid_file="$LOG_DIR/$name.pid"
  pid=""
  [[ -f "$pid_file" ]] && pid="$(cat "$pid_file" 2>/dev/null || true)"

  running=false
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    running=true
  else
    # A demo started before run_demo.sh wrote per-component pid files, or one
    # whose leader exited while its children kept working, would otherwise be
    # reported dead. Every component has its stdout redirected to its own log,
    # so the kernel already knows which processes belong to it.
    pid="$(find_pid_by_log "$log")"
    if [[ -n "$pid" ]]; then
      running=true
    fi
  fi

  last="$(tail -n 1 "$log" 2>/dev/null || true)"

  # uv streams its download and install progress before the node's own output
  # appears, so a log tail still in that shape means dependencies, not a hang.
  if [[ "$last" =~ (Downloading|Downloaded|Building|Installing|Prepared|Resolved|Updating|Audited) ]]; then
    done_n="$(grep -c 'Downloaded' "$log" 2>/dev/null || true)"
    if "$running"; then
      printf '  %s⏳%s %-26s installing deps (%s downloaded) - %s\n' \
        "$YELLOW" "$RESET" "$name" "$done_n" "${last:0:44}"
      installing=$((installing + 1))
      continue
    fi
  fi

  # Only a fatal line counts as failure; ROS logs warnings constantly and the
  # word "not found" shows up in benign parameter-lookup warnings.
  fatal="$(grep -cE '\[ERROR\]|Traceback|command not found|unbound variable|ModuleNotFoundError' "$log" 2>/dev/null || true)"

  if "$running"; then
    if [[ "$fatal" -gt 0 ]]; then
      printf '  %s●%s %-26s running, but %s error line(s) in log\n' \
        "$YELLOW" "$RESET" "$name" "$fatal"
      errored=$((errored + 1))
    else
      printf '  %s●%s %-26s running (pid %s)\n' "$GREEN" "$RESET" "$name" "$pid"
      alive=$((alive + 1))
    fi
  else
    printf '  %s✘%s %-26s DEAD - %s\n' "$RED" "$RESET" "$name" "${last:0:44}"
    dead=$((dead + 1))
  fi
done

printf '%s\n' "──────────────────────────────────────────────────────────────"

# The bridge reports itself listening in its log before the socket is actually
# bound, so read the kernel's table rather than trusting that line. ss and
# netstat are both absent from this image; /proc always has the answer.
port_hex="$(printf '%04X' "$UDP_PORT")"
if awk -v h=":$port_hex" 'NR>1 && index($2, h)' /proc/net/udp /proc/net/udp6 2>/dev/null | grep -q .; then
  printf '  %s●%s UDP %s: listening\n' "$GREEN" "$RESET" "$UDP_PORT"
else
  printf '  %s✘%s UDP %s: nothing listening\n' "$RED" "$RESET" "$UDP_PORT"
fi

printf '\n  %s%s running · %s installing · %s with errors · %s dead%s\n' \
  "$BOLD" "$alive" "$installing" "$errored" "$dead" "$RESET"

if [[ "$installing" -gt 0 ]]; then
  printf '  %sStill installing - the V-JEPA windows appear once it finishes.%s\n' "$YELLOW" "$RESET"
elif [[ "$dead" -eq 0 && "$errored" -eq 0 ]]; then
  printf '  %sAll components ready.%s\n' "$GREEN" "$RESET"
fi
