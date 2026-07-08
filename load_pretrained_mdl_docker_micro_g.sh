#!/usr/bin/env bash
set -euo pipefail

NET_NAME="${NET_NAME:-rosnet}"
ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"

MODEL_PATH="${MODEL_PATH:-/root/catkin_ws/src/object_tracking/models/micro_g_dgm_v1.pkl}"

MOVEIT_STATE=$(docker inspect -f '{{.State.Running}}' "$MOVEIT_NAME")

if [ "$MOVEIT_STATE" == "true" ]; then
    # shellcheck disable=SC2086
    echo $MOVEIT_STATE "$MOVEIT_NAME" "container is running"
    RESULT_MODEL="${RESULT_DIR:-/home/shad/Documents/results/micro_g_dgm_v1.pkl}"
    docker exec "$MOVEIT_NAME" mkdir -p "$(dirname "$MODEL_PATH")"
    docker cp "$RESULT_MODEL" "$MOVEIT_NAME:$MODEL_PATH"
    echo "Copied $RESULT_MODEL to $MOVEIT_NAME:$MODEL_PATH"
else
    echo "$MOVEIT_STATE" "$MOVEIT_NAME" "container is NOT running"
fi


log() { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[PASS]\033[0m $*"; }
