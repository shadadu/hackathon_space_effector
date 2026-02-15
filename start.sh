#!/usr/bin/env bash
set -euo pipefail

############################################
# User-configurable paths / names
############################################
MAIN_WS="${MAIN_WS:-$PWD}"                 # run script from main workspace root
MOVEIT_BUILD_DIR="${MOVEIT_BUILD_DIR:-$MAIN_WS/moveit_build}"

NET_NAME="${NET_NAME:-rosnet}"

ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"

ROS_MASTER_IMAGE="${ROS_MASTER_IMAGE:-ros:noetic-ros-core}"

# Your images:
ASTROBEE_IMAGE="${ASTROBEE_IMAGE:-astrobee_grasp:noetic}"
MOVEIT_IMAGE="${MOVEIT_IMAGE:-moveit_image:noetic}"

# Launch commands inside containers:
ASTROBEE_LAUNCH="${ASTROBEE_LAUNCH:-roslaunch astrobee_grasp perception.launch}"
MOVEIT_LAUNCH="${MOVEIT_LAUNCH:-roslaunch panda_benchmark_moveit demo.launch rviz:=false}"

# Healthcheck timeouts (seconds)
MASTER_TIMEOUT="${MASTER_TIMEOUT:-20}"
ASTROBEE_TIMEOUT="${ASTROBEE_TIMEOUT:-1500}"
MOVEIT_TIMEOUT="${MOVEIT_TIMEOUT:-1500}"

############################################
# Helpers
############################################
log()   { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[PASS]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m $*"; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"; }

docker_rm_if_exists() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    log "Removing existing container: $name"
    docker rm -f "$name" >/dev/null || true
  fi
}

ensure_network() {
  if ! docker network ls --format '{{.Name}}' | grep -qx "$NET_NAME"; then
    log "Creating docker network: $NET_NAME"
    docker network create "$NET_NAME" >/dev/null
  else
    ok "Docker network exists: $NET_NAME"
  fi
}

container_ip() {
  local name="$1"
  docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$name"
}

wait_for_master() {
  log "Waiting for ROS master to respond..."
  local start
  start="$(date +%s)"

  counter=1
  max=10
  while [ $counter -le $max ]; do
    counter=$((counter + 1))
    # test TCP connectivity from inside ros_master container itself
    if docker exec "$ROS_MASTER_NAME" bash -lc "python3 - <<'PY'
import socket
s=socket.socket(); s.settimeout(1)
try:
  s.connect(('127.0.0.1',11311))
  print('ok')
except Exception as e:
  raise SystemExit(1)
finally:
  s.close()
PY" >/dev/null 2>&1; then
      ok "roscore is listening on :11311"
      return 0
    fi

    local now
    now="$(date +%s)"
    if (( now - start > MASTER_TIMEOUT )); then
      fail "Timed out waiting for ROS master"
    fi
    sleep 1
  done
}

wait_for_topic_publisher() {
  local container="$1"
  local topic="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  log "Waiting for publisher on topic: $topic (container=$container)"
  counter=1
  max=10
  while [ $counter -le $max ]; do
    counter=$((counter + 1))
    if docker exec "$container" bash -lc "rostopic info $topic 2>/dev/null | grep -q '^Publishers:.*None'"; then
      : # no publisher yet
    elif docker exec "$container" bash -lc "rostopic info $topic 2>/dev/null | grep -q '^Publishers:'"; then
      # ensure not None
      if ! docker exec "$container" bash -lc "rostopic info $topic 2>/dev/null | grep -q '^Publishers: *None'"; then
        ok "Publisher detected for $topic"
        return 0
      fi
    fi

    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      fail "Timed out waiting for publisher on $topic"
    fi
    sleep 1
  done
}

wait_for_rostopic_echo_once() {
  local container="$1"
  local topic="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  log "Waiting for a message on: $topic (container=$container)"
  counter=1
  max=10
  while [ $counter -le $max ]; do
    counter=$((counter + 1))
    if docker exec "$container" bash -lc "timeout 2 rostopic echo -n 1 $topic >/dev/null 2>&1"; then
      ok "Received at least one message on $topic"
      return 0
    fi

    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      fail "Timed out waiting for a message on $topic"
    fi
    sleep 1
  done
}

wait_for_service() {
  local container="$1"
  local srv="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  log "Waiting for service: $srv (container=$container)"
  counter=1
  max=10
  while [ $counter -le $max ]; do
    counter=$((counter + 1))
    if docker exec "$container" bash -lc "rosservice list 2>/dev/null | grep -qx '$srv'"; then
      ok "Service available: $srv"
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      fail "Timed out waiting for service $srv"
    fi
    sleep 1
  done
}

check_param() {
  local container="$1"
  local param="$2"
  log "Checking param exists: $param"
  if docker exec "$container" bash -lc "rosparam get '$param' >/dev/null 2>&1"; then
    ok "Param exists: $param"
  else
    fail "Param missing: $param"
  fi
}

############################################
# Main
############################################
require_cmd docker

log "Workspace: $MAIN_WS"
cd "$MAIN_WS"

ensure_network

############################################
# Start ROS master
############################################
docker_rm_if_exists "$ROS_MASTER_NAME"

log "Starting ROS master container: $ROS_MASTER_NAME"
# Map port to host for debugging; inside rosnet it’s still reachable as ros_master:11311
docker run -d --name "$ROS_MASTER_NAME" --network "$NET_NAME" -p 11311:11311 \
  "$ROS_MASTER_IMAGE" roscore >/dev/null

ok "Started $ROS_MASTER_NAME (IP=$(container_ip "$ROS_MASTER_NAME"))"
wait_for_master

############################################
# Start Astrobee container + launch perception
############################################
docker_rm_if_exists "$ASTROBEE_NAME"

log "Starting Astrobee container: $ASTROBEE_NAME"
docker run -d --name "$ASTROBEE_NAME" --network "$NET_NAME" \
  -e ROS_MASTER_URI="http://$ROS_MASTER_NAME:11311" \
  "$ASTROBEE_IMAGE" bash -lc "tail -f /dev/null" >/dev/null

ok "Started $ASTROBEE_NAME (IP=$(container_ip "$ASTROBEE_NAME"))"

log "Launching Astrobee perception: $ASTROBEE_LAUNCH"
docker exec -d "$ASTROBEE_NAME" bash -lc "
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash || true
source /root/catkin_ws/devel/setup.bash || true
$ASTROBEE_LAUNCH
" >/dev/null

# Checks: object/state should have publisher and produce a message
wait_for_topic_publisher "$ASTROBEE_NAME" "/object/state" "$ASTROBEE_TIMEOUT"
wait_for_rostopic_echo_once "$ASTROBEE_NAME" "/object/state" "$ASTROBEE_TIMEOUT"
ok "Astrobee perception appears to be publishing /object/state"

############################################
# Start MoveIt container + launch demo
############################################
log "Switching to MoveIt build dir: $MOVEIT_BUILD_DIR"
cd "$MOVEIT_BUILD_DIR" || fail "MoveIt build dir not found: $MOVEIT_BUILD_DIR"

docker_rm_if_exists "$MOVEIT_NAME"

log "Starting MoveIt container: $MOVEIT_NAME"
docker run -d --name "$MOVEIT_NAME" --network "$NET_NAME" \
  -e ROS_MASTER_URI="http://$ROS_MASTER_NAME:11311" \
  "$MOVEIT_IMAGE" bash -lc "tail -f /dev/null" >/dev/null

ok "Started $MOVEIT_NAME (IP=$(container_ip "$MOVEIT_NAME"))"

log "Launching MoveIt demo: $MOVEIT_LAUNCH"
docker exec -d "$MOVEIT_NAME" bash -lc "
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
$MOVEIT_LAUNCH
" >/dev/null

# Checks: MoveIt sees object topic from Astrobee
wait_for_topic_publisher "$ASTROBEE_NAME" "/object/state" "$MOVEIT_TIMEOUT"
wait_for_rostopic_echo_once "$MOVEIT_NAME" "/object/state" "$MOVEIT_TIMEOUT"
ok "MoveIt container can see /object/state from Astrobee"

# Checks: MoveGroup services / DGM service
wait_for_service "$MOVEIT_NAME" "/compute_ik" "$MOVEIT_TIMEOUT" || true
# OMPL planning service name can vary; check both common names
if docker exec "$MOVEIT_NAME" bash -lc "rosservice list | grep -qx '/plan_kinematic_path'"; then
  ok "OMPL planning service available: /plan_kinematic_path"
elif docker exec "$MOVEIT_NAME" bash -lc "rosservice list | grep -qx '/move_group/plan_kinematic_path'"; then
  ok "OMPL planning service available: /move_group/plan_kinematic_path"
else
  warn "Could not find plan_kinematic_path service yet (may still be starting)"
fi

# DGM service (your node advertises it)
wait_for_service "$MOVEIT_NAME" "/dgm/get_motion_plan" "$MOVEIT_TIMEOUT"

# Controller manager param (must exist under /move_group)
check_param "$MOVEIT_NAME" "/move_group/moveit_controller_manager"

# Optional: collision object topic subscriber check (MoveIt subscribes)
if docker exec "$MOVEIT_NAME" bash -lc "rostopic info /collision_object 2>/dev/null | grep -q 'Subscribers:'"; then
  ok "MoveIt subscribes to /collision_object"
else
  warn "Could not confirm /collision_object subscribers (non-fatal)"
fi

log "Stack is up."
ok "ros_master + astrobee perception + moveit demo are running on $NET_NAME"

cat <<EOF

Next commands (optional):
  # View logs
  docker logs -f $ASTROBEE_NAME
  docker logs -f $MOVEIT_NAME

  # Enter containers
  docker exec -it $ASTROBEE_NAME bash
  docker exec -it $MOVEIT_NAME bash

  # Run intercept planner (inside moveit container shell)
  rosrun object_tracking intercept_planner.py _plan_service:=/dgm/get_motion_plan _ik_service:=/compute_ik _world_frame:=world

EOF
