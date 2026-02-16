#!/usr/bin/env bash
set -euo pipefail

############################################
# User-configurable paths / names
############################################
MAIN_WS="${MAIN_WS:-$PWD}"
MOVEIT_BUILD_DIR="${MOVEIT_BUILD_DIR:-$MAIN_WS/moveit_build}"

NET_NAME="${NET_NAME:-rosnet}"

ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"

ROS_MASTER_IMAGE="${ROS_MASTER_IMAGE:-ros:noetic-ros-core}"
ASTROBEE_IMAGE="${ASTROBEE_IMAGE:-astrobee_grasp:noetic}"
MOVEIT_IMAGE="${MOVEIT_IMAGE:-moveit_image:noetic}"

ASTROBEE_LAUNCH="${ASTROBEE_LAUNCH:-roslaunch astrobee_grasp perception.launch}"
MOVEIT_LAUNCH="${MOVEIT_LAUNCH:-roslaunch panda_benchmark_moveit demo.launch rviz:=false}"

MASTER_TIMEOUT="${MASTER_TIMEOUT:-20}"
ASTROBEE_TIMEOUT="${ASTROBEE_TIMEOUT:-40}"
MOVEIT_TIMEOUT="${MOVEIT_TIMEOUT:-60}"

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

# Run a ROS command inside a container with environment sourced properly
ros_exec() {
  local container="$1"; shift
  local cmd="$*"
  docker exec "$container" bash -lc "
set -e
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
if [ -f /opt/ros/noetic/setup.bash ]; then source /opt/ros/noetic/setup.bash; fi
if [ -f /root/catkin_ws/devel/setup.bash ]; then source /root/catkin_ws/devel/setup.bash; fi
$cmd
"
}

make_scripts_executable() {
  local container="$1"
  shift
  local dirs=("$@")

  log "Ensuring Python scripts are executable in container: $container"

  for d in "${dirs[@]}"; do
    ros_exec "$container" "
if [ -d '$d' ]; then
  find '$d' -type f -name '*.py' -exec chmod +x {} \;
fi
"
  done

  ok "Executable permissions enforced."
}

wait_for_master() {
  log "Waiting for ROS master to respond..."
  local start
  start="$(date +%s)"

  counter=1
  max=10
  while [ $counter -le $max ]; do
    counter=$((counter+1))
    if docker exec "$ROS_MASTER_NAME" bash -lc "python3 - <<'PY'
import socket
s=socket.socket(); s.settimeout(1)
try:
  s.connect(('127.0.0.1',11311))
  print('ok')
except Exception:
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
    counter=$((counter+1))
    if ros_exec "$container" "rostopic info $topic 2>/dev/null | grep -q '^Publishers:'" ; then
      if ! ros_exec "$container" "rostopic info $topic 2>/dev/null | grep -q '^Publishers: *None'" ; then
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
  while true; do
    if ros_exec "$container" "timeout 2 rostopic echo -n 1 $topic >/dev/null 2>&1" ; then
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

wait_for_service_any() {
  local container="$1"
  local timeout="$2"
  shift 2
  local services=("$@")
  local start
  start="$(date +%s)"

  log "Waiting for any service: ${services[*]} (container=$container)"
  while true; do
    for s in "${services[@]}"; do
      if ros_exec "$container" "python3 - <<'PY'
import os, sys
import rosgraph
m = rosgraph.Master('/svc_check')
try:
    m.lookupService('${s}')
    sys.exit(0)
except Exception:
    sys.exit(1)
PY" >/dev/null 2>&1; then
        ok "Service available: $s"
        echo "$s"
        return 0
      fi
    done

    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      fail "Timed out waiting for any of: ${services[*]}"
    fi
    sleep 1
  done
}

check_param() {
  local container="$1"
  local param="$2"
  log "Checking param exists: $param"
  if ros_exec "$container" "rosparam get '$param' >/dev/null 2>&1" ; then
    ok "Param exists: $param"
  else
    fail "Param missing: $param"
  fi
}

start_and_test_jacobian_server() {
  local container="$1"
  local timeout="$2"

  log "== Jacobian server: toolchain check (gcc/g++) =="
  ros_exec "$container" "
command -v gcc >/dev/null || { echo 'gcc missing'; exit 1; }
command -v g++ >/dev/null || { echo 'g++ missing'; exit 1; }
gcc --version | head -n 1
g++ --version | head -n 1
" || fail "C++ toolchain missing in $container (install build-essential in Dockerfile)"

  log "== Jacobian server: catkin build =="
  ros_exec "$container" "
cd /root/catkin_ws
catkin_make
" || fail "catkin_make failed in $container (jacobian_server build)"

  log "== Jacobian server: start node in background =="
  docker exec -d "$container" bash -lc "
set -e
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash || true
source /root/catkin_ws/devel/setup.bash || true
rosrun jacobian_server jacobian_server_node
" >/dev/null

  log "== Jacobian server: wait for /get_jacobian =="
  local start
  start="$(date +%s)"
  while true; do
    if ros_exec "$container" "python3 - <<'PY'
import sys, rosgraph
m = rosgraph.Master('/wait_get_jacobian')
try:
    m.lookupService('/get_jacobian')
    sys.exit(0)
except Exception:
    sys.exit(1)
PY" >/dev/null 2>&1; then
      ok "Service available: /get_jacobian"
      break
    fi
    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      fail "Timed out waiting for /get_jacobian"
    fi
    sleep 1
  done

  log "== Jacobian server: test call (Panda) =="
  ros_exec "$container" "python3 - <<'PY'
import rospy
from geometry_msgs.msg import Point
from jacobian_server.srv import GetJacobian, GetJacobianRequest

rospy.init_node('test_get_jacobian', anonymous=True, disable_signals=True)
rospy.wait_for_service('/get_jacobian', timeout=10.0)
srv = rospy.ServiceProxy('/get_jacobian', GetJacobian)

req = GetJacobianRequest()
req.group_name = 'panda_arm'
req.link_name = 'panda_hand'
req.joint_names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']
req.joint_positions = [0.0,0.0,0.0,0.0,0.0,1.571,0.785]
req.reference_point = Point(0,0,0)

resp = srv(req)
print('message:', resp.message)
print('rows, cols:', resp.rows, resp.cols)
assert resp.message == 'OK'
assert resp.rows == 6 and resp.cols == 7
assert len(resp.jacobian) == resp.rows * resp.cols
print('OK Jacobian length:', len(resp.jacobian))
PY" || fail "Jacobian test call failed"

  ok "Jacobian server built, launched, and validated."
}


############################################
# Main
############################################
require_cmd docker

log "Workspace: $MAIN_WS"
cd "$MAIN_WS"

ensure_network

# ROS master
docker_rm_if_exists "$ROS_MASTER_NAME"
log "Starting ROS master: $ROS_MASTER_NAME"
docker run -d --name "$ROS_MASTER_NAME" --network "$NET_NAME" -p 11311:11311 \
  "$ROS_MASTER_IMAGE" roscore >/dev/null
ok "Started $ROS_MASTER_NAME (IP=$(container_ip "$ROS_MASTER_NAME"))"
wait_for_master

# Astrobee
docker_rm_if_exists "$ASTROBEE_NAME"
log "Starting Astrobee container: $ASTROBEE_NAME"
docker run -d --name "$ASTROBEE_NAME" --network "$NET_NAME" \
  -e ROS_MASTER_URI="http://$ROS_MASTER_NAME:11311" \
  "$ASTROBEE_IMAGE" bash -lc "tail -f /dev/null" >/dev/null
ok "Started $ASTROBEE_NAME (IP=$(container_ip "$ASTROBEE_NAME"))"

make_scripts_executable "$ASTROBEE_NAME" \
  "/root/catkin_ws/src/astrobee_grasp/scripts"

log "Launching Astrobee perception"
docker exec -d "$ASTROBEE_NAME" bash -lc "
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash || true
source /root/catkin_ws/devel/setup.bash || true
$ASTROBEE_LAUNCH
" >/dev/null

wait_for_topic_publisher "$ASTROBEE_NAME" "/object/state" "$ASTROBEE_TIMEOUT"
wait_for_rostopic_echo_once "$ASTROBEE_NAME" "/object/state" "$ASTROBEE_TIMEOUT"
ok "Astrobee publishing /object/state"

# MoveIt
log "Switching to MoveIt build dir: $MOVEIT_BUILD_DIR"
cd "$MOVEIT_BUILD_DIR" || fail "MoveIt build dir not found: $MOVEIT_BUILD_DIR"

docker_rm_if_exists "$MOVEIT_NAME"
log "Starting MoveIt container: $MOVEIT_NAME"
docker run -d --name "$MOVEIT_NAME" --network "$NET_NAME" \
  -e ROS_MASTER_URI="http://$ROS_MASTER_NAME:11311" \
  "$MOVEIT_IMAGE" bash -lc "tail -f /dev/null" >/dev/null
ok "Started $MOVEIT_NAME (IP=$(container_ip "$MOVEIT_NAME"))"

make_scripts_executable "$MOVEIT_NAME" \
  "/root/catkin_ws/src/object_tracking/scripts"

log "Checking C++ toolchain + building MoveIt workspace (for jacobian_server)..."
ros_exec "$MOVEIT_NAME" "
command -v gcc >/dev/null || { echo 'gcc missing'; exit 1; }
command -v g++ >/dev/null || { echo 'g++ missing'; exit 1; }
cd /root/catkin_ws
catkin_make
"
ok "catkin_make succeeded (jacobian_server build ready)"


log "Launching MoveIt demo"
docker exec -d "$MOVEIT_NAME" bash -lc "
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
$MOVEIT_LAUNCH
" >/dev/null

# Start + test Jacobian server (requires robot_description from MoveIt launch)
start_and_test_jacobian_server "$MOVEIT_NAME" "$MOVEIT_TIMEOUT"


# Confirm MoveIt can see the Astrobee topic
wait_for_rostopic_echo_once "$MOVEIT_NAME" "/object/state" "$MOVEIT_TIMEOUT"
ok "MoveIt can receive /object/state"

# Services & params
# Service names can be either global or under /move_group depending on config
IK_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/compute_ik" "/move_group/compute_ik")"
PLAN_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/plan_kinematic_path" "/move_group/plan_kinematic_path")"
DGM_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/dgm/get_motion_plan")"

ok "Found IK service: $IK_SVC"
ok "Found planning service: $PLAN_SVC"
ok "Found DGM service: $DGM_SVC"

# controller manager param can be global or private depending on launch
if ros_exec "$MOVEIT_NAME" "rosparam get /move_group/moveit_controller_manager >/dev/null 2>&1"; then
  ok "Param exists: /move_group/moveit_controller_manager"
elif ros_exec "$MOVEIT_NAME" "rosparam get /moveit_controller_manager >/dev/null 2>&1"; then
  ok "Param exists: /moveit_controller_manager"
else
  fail "Param missing: /move_group/moveit_controller_manager or /moveit_controller_manager"
fi

log "Stack is up and basic checks passed."
ok "ros_master + astrobee + moveit are running on network '$NET_NAME'"
