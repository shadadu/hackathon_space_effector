#!/usr/bin/env bash
set -euo pipefail

NET_NAME="${NET_NAME:-rosnet}"
ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"

MOVEIT_STATE=$(docker inspect -f '{{.State.Running}}' moveit)

if [ "$MOVEIT_STATE" == "true" ]; then
    # shellcheck disable=SC2086
    echo $MOVEIT_STATE "$MOVEIT_NAME" "container is running"
    RESULT_MODEL="${RESULT_DIR:-/home/shad/Documents/results/panda_dgm_v1.pkl}"
    docker cp "$RESULT_MODEL" "$MOVEIT_NAME":'$model_path'
else
    echo "$MOVEIT_STATE" "$MOVEIT_NAME" "container is NOT running"
fi


log() { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[PASS]\033[0m $*"; }




