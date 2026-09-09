#!/usr/bin/env bash
set -euo pipefail

NET_NAME="${NET_NAME:-rosnet}"
ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"
MODEL_NAME="${MODEL_NAME:-micro_g_dgm_v1.pth}"

RESULT_STEPS_PATH="${RESULT_DIR:-/home/shad/Documents/results/micro_g_dgm_v1.pkl}"
TRIALS_DATA="${RESULT_DIR:-/home/shad/Documents/results/micro_g_dgm_inference_trials.csv}"

log() { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[PASS]\033[0m $*"; }

#docker cp moveit:'$model_path' "$RESULT_MODEL"
docker cp moveit:'/root/catkin_ws/src/object_tracking/models/results/micro_g_dgm_inference_step_path.csv' "$RESULT_STEPS_PATH"
log "Steps path copied to $RESULT_STEPS_PATH"

docker cp moveit:'/root/catkin_ws/src/object_tracking/models/results/micro_g_dgm_inference_trials.csv' "$TRIALS_DATA"
log "Trials data copied to $TRIALS_DATA"

ok "Done: Inference trials results copied to persistent storage"