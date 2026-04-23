#!/usr/bin/env bash
set -euo pipefail

NET_NAME="${NET_NAME:-rosnet}"
ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"
MODEL_NAME="${MODEL_NAME:-panda_dgm_v1.pth}"

RESULT_MODEL="${RESULT_DIR:-/Users/rckyi/Documents/results/panda_dgm_v1.pth}"
TRAIN_PERF_DATA="${RESULT_DIR:-/Users/rckyi/Documents/results/training_perf_data.csv}"

log() { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[PASS]\033[0m $*"; }

#docker cp moveit:'$model_path' "$RESULT_MODEL"
docker cp moveit:'/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pth' "$RESULT_MODEL"
log "Model copied to $RESULT_MODEL"

docker cp moveit:'/root/catkin_ws/src/object_tracking/models/train_perf_data.csv' "$TRAIN_PERF_DATA"
log "Training perf data copied to $TRAIN_PERF_DATA"

ok "Done: training results copied to persistent storage"