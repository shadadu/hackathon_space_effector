#!/usr/bin/env bash
set -euo pipefail

NET_NAME="${NET_NAME:-rosnet}"
ROS_MASTER_NAME="${ROS_MASTER_NAME:-ros_master}"
ASTROBEE_NAME="${ASTROBEE_NAME:-astrobee}"
MOVEIT_NAME="${MOVEIT_NAME:-moveit}"

log() { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[PASS]\033[0m $*"; }

rm_if_exists() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    log "Removing container: $name"
    docker rm -f "$name" >/dev/null || true
  else
    ok "Container not present: $name"
  fi
}

log "Tearing down stack..."
rm_if_exists "$MOVEIT_NAME"
rm_if_exists "$ASTROBEE_NAME"
rm_if_exists "$ROS_MASTER_NAME"

if docker network ls --format '{{.Name}}' | grep -qx "$NET_NAME"; then
  log "Removing network: $NET_NAME"
  docker network rm "$NET_NAME" >/dev/null || true
  ok "Removed network: $NET_NAME"
else
  ok "Network not present: $NET_NAME"
fi

log "Freeing memory space/volumes"
docker system prune --volumes --force

ok "Teardown complete."
