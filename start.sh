#!/usr/bin/env bash
set -euo pipefail

set -x  # optional: comment out if too verbose

LAST_STEP="start"
trap 'ec=$?; echo -e "\n\033[1;33m[DEBUG]\033[0m start.sh exiting (code=$ec) at step=$LAST_STEP"; exit $ec' EXIT


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
ASTROBEE_TIMEOUT="${ASTROBEE_TIMEOUT:-100}"
MOVEIT_TIMEOUT="${MOVEIT_TIMEOUT:-200}"

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

start_dgm_service() {
  local container="$1"
  log "Starting DGM planner service node manually (container=$container)"

  docker exec -d "$container" bash -lc "
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
nohup rosrun object_tracking dgm_planner_node.py \
  _service_name:=/dgm/get_motion_plan \
  _ik_service:=/compute_ik \
  _group_name:=panda_arm \
  _ee_link:=panda_hand \
  > /root/start_logs/dgm_planner.log 2>&1 &
" >/dev/null
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
import sys, rosgraph
m = rosgraph.Master('/svc_check')
try:
    m.lookupService('${s}')
    sys.exit(0)
except Exception:
    sys.exit(1)
PY" >/dev/null 2>&1; then
        ok "Service available: $s"
#        echo "$s"
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

wait_for_param() {
  local container="$1"
  local param="$2"
  local timeout="$3"
  local start; start="$(date +%s)"

  log "Waiting for param: $param (container=$container)"
  while true; do
    if ros_exec "$container" "python3 - <<'PY'
import sys, rosgraph
m = rosgraph.Master('/wait_param')
try:
    val = m.getParam('${param}')
    sys.exit(0)
except Exception:
    sys.exit(1)
PY" >/dev/null 2>&1; then
      ok "Param available: $param"
      return 0
    fi

    # If roslaunch crashed, show logs immediately
    if ! docker exec "$container" bash -lc "test -f /root/start_logs/moveit_demo.pid && ps -p \$(cat /root/start_logs/moveit_demo.pid) >/dev/null 2>&1"; then
      warn "MoveIt launch process is not running (crashed or never started). Showing last 120 log lines:"
      docker exec "$container" bash -lc "tail -n 120 /root/start_logs/moveit_demo.log 2>/dev/null || true"
      fail "MoveIt launch not running; cannot get $param"
      docker exec "$container" bash -lc "echo '--- moveit_demo.log ---'; cat /root/start_logs/moveit_demo.log 2>/dev/null || true"
    fi

    local now; now="$(date +%s)"
    if (( now - start > timeout )); then
      warn "Timed out waiting for $param. Diagnostics:"
      ros_exec "$container" "rosnode list | head -n 50 || true" || true
      docker exec "$container" bash -lc "tail -n 200 /root/start_logs/moveit_demo.log 2>/dev/null || true"
      fail "Timed out waiting for $param"
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
  local timeout="${2:-120}"

  log "== Jacobian server: toolchain check (gcc/g++) =="
  ros_exec "$container" "
command -v gcc >/dev/null || { echo 'gcc missing'; exit 1; }
command -v g++ >/dev/null || { echo 'g++ missing'; exit 1; }
" || fail "C++ toolchain missing in $container (install build-essential in Dockerfile)"

  log "== Jacobian server: verify package exists in workspace =="
  ros_exec "$container" "rospack find jacobian_server >/dev/null" \
    || fail "jacobian_server package not found in $container. Ensure MoveIt Dockerfile COPYs jacobian_server into /root/catkin_ws/src and rebuild image."

  log "== Jacobian server: wait for /robot_description (MoveIt must be up) =="
  local start
  start="$(date +%s)"
  while true; do
    if ros_exec "$container" \
      "rosparam get /robot_description >/dev/null 2>&1"; then
      ok "/robot_description present"
      break
    fi
    local now; now="$(date +%s)"
    if (( now - start > timeout )); then
      fail "Timed out waiting for /robot_description (is move_group.launch actually running?)"
    fi
    sleep 1
  done

  log "== Jacobian server: build package only (fast) =="
  ros_exec "$container" "
cd /root/catkin_ws
catkin_make --pkg jacobian_server
" || fail "catkin_make --pkg jacobian_server failed"

  # Kill any prior instance (prevents duplicate nodes / stale ports)
  log "== Jacobian server: kill any existing node =="
  ros_exec "$container" "
if rosnode list 2>/dev/null | grep -qx '/jacobian_server_node'; then
  rosnode kill /jacobian_server_node || true
fi
" || true

  log "== Jacobian server: start node in background =="
  docker exec -d "$container" bash -lc "
set -e
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
# run with a known name so we can detect it
rosrun jacobian_server jacobian_server_node __name:=jacobian_server_node
" >/dev/null

  log "== Jacobian server: wait for /jacobian_server_node to appear =="
  start="$(date +%s)"
  while true; do
    if ros_exec "$container" "rosnode list 2>/dev/null | grep -qx '/jacobian_server_node'"; then
      ok "Node is running: /jacobian_server_node"
      break
    fi
    local now; now="$(date +%s)"
    if (( now - start > 15 )); then
      warn "jacobian_server_node did not appear. Printing recent logs:"
      ros_exec "$container" "ls -lt ~/.ros/log/latest | head -n 30 || true"
      ros_exec "$container" "tail -n 200 ~/.ros/log/latest/*.log 2>/dev/null | tail -n 200 || true"
      fail "jacobian_server_node failed to start"
    fi
    sleep 1
  done

  log "== Jacobian server: wait for /get_jacobian service =="
  start="$(date +%s)"
  while true; do
    # if node died, fail early and show logs
    if ! ros_exec "$container" "rosnode ping -c 1 /jacobian_server_node >/dev/null 2>&1"; then
      warn "jacobian_server_node died while waiting for service. Recent logs:"
      ros_exec "$container" "ls -lt ~/.ros/log/latest | head -n 30 || true"
      ros_exec "$container" "tail -n 200 ~/.ros/log/latest/*.log 2>/dev/null | tail -n 200 || true"
      fail "jacobian_server_node crashed"
    fi

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

    local now; now="$(date +%s)"
    if (( now - start > timeout )); then
      warn "Timed out waiting for /get_jacobian. Dumping diagnostics:"
      ros_exec "$container" "rosnode list | grep jacob || true"
      ros_exec "$container" "rosservice list | grep jacob || true"
      ros_exec "$container" "tail -n 200 ~/.ros/log/latest/*.log 2>/dev/null | tail -n 200 || true"
      fail "Timed out waiting for /get_jacobian"
    fi
    sleep 1
  done

  ok "Jacobian server built + running + service up."
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
log "Building Astrobee container: $ASTROBEE_NAME"
docker build -t "$ASTROBEE_IMAGE" .
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
log "Building MoveIt Container: $MOVEIT_NAME"
docker build -t "$MOVEIT_IMAGE" .
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

log "Launching MoveIt demo with robust logging"
docker exec -d "$MOVEIT_NAME" bash -lc "
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

mkdir -p /root/start_logs
echo '[bringup] starting moveit launch...' > /root/start_logs/moveit_demo.log
echo '[bringup] ROS_MASTER_URI='\"\$ROS_MASTER_URI\" >> /root/start_logs/moveit_demo.log
echo '[bringup] ROS_IP='\"\$ROS_IP\" >> /root/start_logs/moveit_demo.log

# Preflight: prove rospack can find package
rospack find panda_benchmark_moveit >> /root/start_logs/moveit_demo.log 2>&1 || true
rospack find moveit_resources_panda_description >> /root/start_logs/moveit_demo.log 2>&1 || true
rospack find moveit_resources_panda_moveit_config >> /root/start_logs/moveit_demo.log 2>&1 || true

# Launch in background, always append logs
nohup roslaunch panda_benchmark_moveit demo.launch rviz:=false >> /root/start_logs/moveit_demo.log 2>&1 &
echo \$! > /root/start_logs/moveit_demo.pid
" >/dev/null




## Start + test Jacobian server (requires robot_description from MoveIt launch)
#start_and_test_jacobian_server "$MOVEIT_NAME" "$MOVEIT_TIMEOUT"
#
#jacobian_service_test "$MOVEIT_NAME" "$MOVEIT_TIMEOUT"


# Confirm MoveIt can see the Astrobee topic
wait_for_rostopic_echo_once "$MOVEIT_NAME" "/object/state" "$MOVEIT_TIMEOUT"
ok "MoveIt can receive /object/state"

wait_for_param "$MOVEIT_NAME" "/robot_description" "$MOVEIT_TIMEOUT"
wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/compute_ik" "/move_group/compute_ik" >/dev/null
ok "MoveIt core services are up"


## Start + test Jacobian server (requires robot_description from MoveIt launch)
start_and_test_jacobian_server "$MOVEIT_NAME" "$MOVEIT_TIMEOUT"
#
#jacobian_service_test "$MOVEIT_NAME" "$MOVEIT_TIMEOUT"

# Services & params
# Service names can be either global or under /move_group depending on config
LAST_STEP="wait_services"
log "Discovering IK / Plan / DGM / Jacobian services"

IK_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/compute_ik" "/move_group/compute_ik" || true)"
PLAN_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/plan_kinematic_path" "/move_group/plan_kinematic_path" || true)"
start_dgm_service "$MOVEIT_NAME"
DGM_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/dgm/get_motion_plan" || true)"
#DGM_SVC="$(wait_for_service_any "$MOVEIT_NAME" 20 "/dgm/get_motion_plan" || true)"

#if [[ -z "${DGM_SVC:-}" ]]; then
#  warn "DGM service not found; starting dgm_planner_node manually..."
#  start_dgm_service "$MOVEIT_NAME"
#  DGM_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/dgm/get_motion_plan")"
#fi
#ok "Found DGM service: $DGM_SVC"

JAC_SVC="$(wait_for_service_any "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/get_jacobian" "/jacobian_server/get_jacobian" || true)"

# If any are empty, dump diagnostics and fail explicitly
if [[ -z "${IK_SVC:-}" || -z "${PLAN_SVC:-}" || -z "${DGM_SVC:-}" || -z "${JAC_SVC:-}" ]]; then
  warn "One or more services not found."
  ros_exec "$MOVEIT_NAME" "rosservice list | head -n 200" || true
  fail "Service discovery failed (IK='$IK_SVC' PLAN='$PLAN_SVC' DGM='$DGM_SVC' JAC='$JAC_SVC')"
fi

ok "Found IK service: $IK_SVC"
ok "Found planning service: $PLAN_SVC"
ok "Found DGM service: $DGM_SVC"
ok "Found Jacobian service: $JAC_SVC"



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
LAST_STEP="end"
log "Reached end-of-script final checks block"
