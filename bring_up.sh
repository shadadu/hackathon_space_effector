#!/usr/bin/env bash
set -Eeuo pipefail

############################################
# Config
############################################
MAIN_WS="${MAIN_WS:-$PWD}"
MOVEIT_BUILD_DIR="${MOVEIT_BUILD_DIR:-$MAIN_WS/moveit_build}"

NET_NAME="${NET_NAME:-rosnet}"

ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"
DGM_MODELS_VOLUME="${DGM_MODELS_VOLUME:-moveit_dgm_models}"

ROS_MASTER_IMAGE="${ROS_MASTER_IMAGE:-ros:noetic-ros-core}"
ASTROBEE_IMAGE="${ASTROBEE_IMAGE:-astrobee_grasp:noetic}"
MOVEIT_IMAGE="${MOVEIT_IMAGE:-moveit_image:noetic}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

ASTROBEE_LAUNCH="${ASTROBEE_LAUNCH:-roslaunch astrobee_grasp perception.launch}"
MOVEIT_LAUNCH="${MOVEIT_LAUNCH:-roslaunch panda_benchmark_moveit demo.launch rviz:=false start_dgm_planner:=false}"
ASTROBEE_ENABLE_X11="${ASTROBEE_ENABLE_X11:-false}"

MASTER_TIMEOUT="${MASTER_TIMEOUT:-20}"
ASTROBEE_TIMEOUT="${ASTROBEE_TIMEOUT:-120}"
MOVEIT_TIMEOUT="${MOVEIT_TIMEOUT:-300}"

DGM_MODEL_PATH="${DGM_MODEL_PATH:-/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pkl}"
DGM_NODE_NAME="${DGM_NODE_NAME:-dgm_planner_node}"
DGM_SERVICE_NAME="${DGM_SERVICE_NAME:-/dgm/get_motion_plan}"

LAST_STEP="init"

############################################
# Logging
############################################
log()  { printf "\n\033[1;34m[INFO]\033[0m %s\n" "$*" >&2; }
ok()   { printf "\033[1;32m[PASS]\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*" >&2; }
fail() { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*" >&2; exit 1; }

trap 'ec=$?; printf "\n\033[1;33m[DEBUG]\033[0m start.sh exiting code=%s step=%s\n" "$ec" "$LAST_STEP" >&2; exit "$ec"' EXIT

############################################
# Helpers
############################################
require_cmd() { command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"; }

docker_rm_if_exists() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    log "Removing existing container: $name"
    docker rm -f "$name" >/dev/null 2>&1 || true
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
  docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$1"
}

ros_exec() {
  local container="$1"; shift
  local cmd="$*"

  docker exec "$container" bash -lc "
set -Eeuo pipefail
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash
[ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash
$cmd
"
}

ros_bg_exec() {
  local container="$1"
  local log_file="$2"
  shift 2
  local cmd="$*"

  docker exec -d "$container" bash -lc "
set -Eeuo pipefail
mkdir -p /root/start_logs
export ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311
unset ROS_HOSTNAME
export ROS_IP=\$(hostname -i | awk '{print \$1}')
[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash
[ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash
nohup bash -lc '$cmd' >> '$log_file' 2>&1 &
echo \$! > '${log_file}.pid'
"
}

make_scripts_executable() {
  local container="$1"; shift
  local dirs=("$@")
  for d in "${dirs[@]}"; do
    ros_exec "$container" "
if [ -d '$d' ]; then
  find '$d' -type f -name '*.py' -exec chmod +x {} \;
fi
"
  done
  ok "Executable permissions enforced in $container"
}

wait_for_master() {
  local start
  start="$(date +%s)"
  log "Waiting for ROS master..."
  while true; do
    if docker exec "$ROS_MASTER_NAME" bash -lc "python3 - <<'PY'
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', 11311))
    print('ok')
finally:
    s.close()
PY" >/dev/null 2>&1; then
      ok "roscore is listening on :11311"
      return 0
    fi
    if (( $(date +%s) - start > MASTER_TIMEOUT )); then
      fail "Timed out waiting for ROS master"
    fi
    sleep 1
  done
}

wait_for_topic_message() {
  local container="$1"
  local topic="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  log "Waiting for message on topic: $topic (container=$container)"
  while true; do
    if ros_exec "$container" "timeout 3 rostopic echo -n 1 '$topic' >/dev/null 2>&1"; then
      ok "Received message on $topic"
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      fail "Timed out waiting for message on $topic"
    fi
    sleep 1
  done
}

wait_for_param() {
  local container="$1"
  local param="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  log "Waiting for param: $param (container=$container)"
  while true; do
    if ros_exec "$container" "rosparam get '$param' >/dev/null 2>&1"; then
      ok "Param available: $param"
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      warn "Timed out waiting for param: $param"
      ros_exec "$container" "rosnode list | head -n 100 || true" || true
      fail "Missing param: $param"
    fi
    sleep 1
  done
}

find_first_service() {
  local container="$1"
  local timeout="$2"
  shift 2
  local services=("$@")
  local start
  start="$(date +%s)"

  log "Waiting for any service: ${services[*]} (container=$container)"
  while true; do
    for s in "${services[@]}"; do
      if ros_exec "$container" "rosservice info '$s' >/dev/null 2>&1"; then
        printf "%s\n" "$s"
        return 0
      fi
    done

    if (( $(date +%s) - start > timeout )); then
      return 1
    fi
    sleep 1
  done
}

wait_for_node() {
  local container="$1"
  local node="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"

  log "Waiting for node: $node (container=$container)"
  while true; do
    if ros_exec "$container" "rosnode list | grep -qx '$node'"; then
      ok "Node available: $node"
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      fail "Timed out waiting for node: $node"
    fi
    sleep 1
  done
}

build_image() {
  local image="$1"
  local dir="$2"
  log "Building image $image from $dir"
  docker build --platform "$DOCKER_PLATFORM" --build-arg ROS_PLATFORM="$DOCKER_PLATFORM" -t "$image" "$dir"
  ok "Built image: $image"
}

start_idle_container() {
  local name="$1"
  local image="$2"
  docker_rm_if_exists "$name"

  log "Starting container: $name"
  local docker_args=(
    docker run -d
    --platform "$DOCKER_PLATFORM"
    --name "$name"
    --network "$NET_NAME"
    -e "ROS_MASTER_URI=http://$ROS_MASTER_NAME:11311"
  )

  if [[ "$name" == "$ASTROBEE_NAME" ]]; then
    if [[ "$ASTROBEE_ENABLE_X11" == "true" || "$ASTROBEE_ENABLE_X11" == "1" ]]; then
      docker_args+=(
        -e "DISPLAY=${DISPLAY:-:0}"
        -e "QT_X11_NO_MITSHM=1"
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
      )
      if [[ -d /dev/dri ]]; then
        docker_args+=(--device /dev/dri)
      fi
    fi
  elif [[ "$name" == "$MOVEIT_NAME" ]]; then
    docker_args+=(
      -v "$DGM_MODELS_VOLUME:/root/catkin_ws/src/object_tracking/models"
    )
  fi

  "${docker_args[@]}" "$image" bash -lc "tail -f /dev/null" >/dev/null

  ok "Started $name (IP=$(container_ip "$name"))"
}

start_dgm_service() {
  local container="$1"
  local ik_service="$2"
  local model_path="$3"

  LAST_STEP="start_dgm_service"

  log "Ensuring DGM model exists: $model_path"
  if ! ros_exec "$container" "[ -f '$model_path' ]"; then
    warn "DGM model missing. Running short pretrain..."
    ros_bg_exec "$container" "/root/start_logs/dgm_pretrain.log" \
      "rosrun object_tracking dgm_pretrain.py _T:=2.0 _iters:=20 _batch:=192"

    local start
    start="$(date +%s)"
    while true; do
      if ros_exec "$container" "[ -f '$model_path' ]"; then
        ok "DGM model created: $model_path"
        break
      fi
      if (( $(date +%s) - start > 280 )); then
        ros_exec "$container" "tail -n 200 /root/start_logs/dgm_pretrain.log || true" || true
        fail "Timed out waiting for DGM model file"
      fi
      sleep 2
    done
  else
    ok "DGM model exists"
  fi

  log "Stopping old DGM node, if any"
  ros_exec "$container" "
if rosnode list 2>/dev/null | grep -qx '/$DGM_NODE_NAME'; then
  rosnode kill '/$DGM_NODE_NAME' || true
fi
" || true

  log "Starting DGM planner node"
  ros_bg_exec "$container" "/root/start_logs/dgm_planner.log" \
    "rosrun object_tracking dgm_planner_node.py \
      __name:=$DGM_NODE_NAME \
      _service_name:=$DGM_SERVICE_NAME \
      _ik_service:=$ik_service \
      _group_name:=panda_arm \
      _ee_link:=panda_hand \
      _model_path:=$model_path"

  wait_for_node "$container" "/$DGM_NODE_NAME" 60

  local dgm_svc
  dgm_svc="$(find_first_service "$container" 60 "$DGM_SERVICE_NAME")" || {
    warn "DGM planner log:"
    ros_exec "$container" "tail -n 200 /root/start_logs/dgm_planner.log || true" || true
    fail "DGM service failed to start: $DGM_SERVICE_NAME"
  }

  ok "DGM service available: $dgm_svc"
}

start_and_test_jacobian_server() {
  local container="$1"
  local timeout="${2:-120}"

  LAST_STEP="jacobian_server"

  log "Checking toolchain and jacobian_server package"
  ros_exec "$container" "
command -v gcc >/dev/null
command -v g++ >/dev/null
rospack find jacobian_server >/dev/null
" || fail "jacobian_server prerequisites missing in $container"

  wait_for_param "$container" "/robot_description" "$timeout"

  ros_exec "$container" "
if rosnode list 2>/dev/null | grep -qx '/jacobian_server_node'; then
  rosnode kill /jacobian_server_node || true
fi
" || true

  log "Starting jacobian_server"
  ros_bg_exec "$container" "/root/start_logs/jacobian_server.log" \
    "rosrun jacobian_server jacobian_server_node __name:=jacobian_server_node"

  wait_for_node "$container" "/jacobian_server_node" 20

  local jac_svc
  jac_svc="$(find_first_service "$container" "$timeout" "/get_jacobian" "/jacobian_server/get_jacobian")" || {
    ros_exec "$container" "tail -n 200 /root/start_logs/jacobian_server.log || true" || true
    fail "Timed out waiting for Jacobian service"
  }

  ok "Jacobian service available: $jac_svc"
}

############################################
# Main
############################################
require_cmd docker
LAST_STEP="main"

log "Workspace: $MAIN_WS"
cd "$MAIN_WS"

ensure_network

############################################
# ROS master
############################################
LAST_STEP="ros_master"
docker_rm_if_exists "$ROS_MASTER_NAME"
log "Starting ROS master"
docker run -d --platform "$DOCKER_PLATFORM" --name "$ROS_MASTER_NAME" --network "$NET_NAME" -p 11311:11311 \
  "$ROS_MASTER_IMAGE" roscore >/dev/null
ok "Started $ROS_MASTER_NAME (IP=$(container_ip "$ROS_MASTER_NAME"))"
wait_for_master

############################################
# Astrobee
############################################
LAST_STEP="astrobee_build"
build_image "$ASTROBEE_IMAGE" "$MAIN_WS"

LAST_STEP="astrobee_start"
start_idle_container "$ASTROBEE_NAME" "$ASTROBEE_IMAGE"
make_scripts_executable "$ASTROBEE_NAME" "/root/catkin_ws/src/astrobee_grasp/scripts"

log "Launching Astrobee perception"
ros_bg_exec "$ASTROBEE_NAME" "/root/start_logs/astrobee_perception.log" "$ASTROBEE_LAUNCH"

wait_for_topic_message "$ASTROBEE_NAME" "/object/state" "$ASTROBEE_TIMEOUT"
ok "Astrobee is publishing /object/state"

############################################
# MoveIt
############################################
LAST_STEP="moveit_build"
build_image "$MOVEIT_IMAGE" "$MOVEIT_BUILD_DIR"

LAST_STEP="moveit_start"
start_idle_container "$MOVEIT_NAME" "$MOVEIT_IMAGE"
make_scripts_executable "$MOVEIT_NAME" "/root/catkin_ws/src/object_tracking/src"

log "Verifying prebuilt MoveIt workspace"
ros_exec "$MOVEIT_NAME" "
command -v gcc >/dev/null
command -v g++ >/dev/null
test -f /root/catkin_ws/devel/setup.bash
test -x /root/catkin_ws/devel/lib/jacobian_server/jacobian_server_node
" || fail "MoveIt image is missing prebuilt workspace artifacts"

log "Launching MoveIt demo"
ros_bg_exec "$MOVEIT_NAME" "/root/start_logs/moveit_demo.log" "$MOVEIT_LAUNCH"

wait_for_topic_message "$MOVEIT_NAME" "/object/state" "$MOVEIT_TIMEOUT"
wait_for_param "$MOVEIT_NAME" "/robot_description" "$MOVEIT_TIMEOUT"

IK_SVC="$(find_first_service "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/compute_ik" "/move_group/compute_ik")" \
  || fail "No IK service found"
PLAN_SVC="$(find_first_service "$MOVEIT_NAME" "$MOVEIT_TIMEOUT" "/plan_kinematic_path" "/move_group/plan_kinematic_path")" \
  || fail "No planning service found"

ok "Found IK service: $IK_SVC"
ok "Found planning service: $PLAN_SVC"

############################################
# Jacobian server
############################################
start_and_test_jacobian_server "$MOVEIT_NAME" "$MOVEIT_TIMEOUT"

############################################
# DGM planner
############################################
start_dgm_service "$MOVEIT_NAME" "$IK_SVC" "$DGM_MODEL_PATH"

############################################
# Controller manager param
############################################
LAST_STEP="controller_param"
if ros_exec "$MOVEIT_NAME" "rosparam get /move_group/moveit_controller_manager >/dev/null 2>&1"; then
  ok "Param exists: /move_group/moveit_controller_manager"
elif ros_exec "$MOVEIT_NAME" "rosparam get /moveit_controller_manager >/dev/null 2>&1"; then
  ok "Param exists: /moveit_controller_manager"
else
  fail "Missing moveit_controller_manager param"
fi

LAST_STEP="done"
log "Stack is up and checks passed."
ok "ROS master + Astrobee + MoveIt + Jacobian + DGM are running on network '$NET_NAME'"
