#!/usr/bin/env python3
import os
import csv
import time
import json
import random
import statistics
from typing import Dict, List, Tuple, Optional

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints, PositionConstraint
from moveit_msgs.msg import MotionPlanRequest
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from moveit_msgs.msg import PlanningSceneComponents
from shape_msgs.msg import SolidPrimitive


# ----------------- utilities -----------------

def make_robot_state_from_joint_dict(joint_dict: Dict[str, float]) -> RobotState:
    js = JointState()
    js.name = list(joint_dict.keys())
    js.position = [joint_dict[n] for n in js.name]
    rs = RobotState()
    rs.joint_state = js
    return rs


def panda_extended_open_start_state() -> RobotState:
    # A deterministic, collision-free-ish default for panda resources
    joints = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        "panda_finger_joint1": 0.035,
        "panda_finger_joint2": 0.035,
    }
    return make_robot_state_from_joint_dict(joints)


def get_planning_scene(service_name: str = "/get_planning_scene"):
    rospy.wait_for_service(service_name, timeout=20.0)
    srv = rospy.ServiceProxy(service_name, GetPlanningScene)

    req = GetPlanningSceneRequest()
    req.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.SCENE_SETTINGS
    )
    return srv(req).scene


def start_state_from_scene_or_default(scene) -> RobotState:
    """
    Prefer the planning scene robot_state (more complete in headless setups),
    but ensure gripper joints exist.
    """
    if scene and scene.robot_state and scene.robot_state.joint_state.name:
        rs = scene.robot_state
        names = list(rs.joint_state.name)
        pos = list(rs.joint_state.position)

        if "panda_finger_joint1" not in names:
            names += ["panda_finger_joint1", "panda_finger_joint2"]
            pos += [0.035, 0.035]

        rs.joint_state.name = names
        rs.joint_state.position = pos
        return rs

    return panda_extended_open_start_state()


def sample_goal(bounds: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        random.uniform(bounds["x_min"], bounds["x_max"]),
        random.uniform(bounds["y_min"], bounds["y_max"]),
        random.uniform(bounds["z_min"], bounds["z_max"]),
    )


def make_goal_pose(world_frame: str, x: float, y: float, z: float) -> PoseStamped:
    g = PoseStamped()
    g.header.frame_id = world_frame
    g.header.stamp = rospy.Time.now()
    g.pose.position.x = x
    g.pose.position.y = y
    g.pose.position.z = z
    g.pose.orientation.w = 1.0
    return g


def make_position_only_constraints(goal: PoseStamped, link_name: str, pos_tol: float = 0.02) -> Constraints:
    c = Constraints()
    c.name = "pos_only_goal"

    pc = PositionConstraint()
    pc.header = goal.header
    pc.link_name = link_name
    pc.weight = 1.0

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [pos_tol, pos_tol, pos_tol]  # a small box

    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal.pose)

    c.position_constraints.append(pc)
    return c


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    v = sorted(vals)
    return {
        "n": len(v),
        "mean": float(statistics.mean(v)),
        "median": float(statistics.median(v)),
        "p90": float(v[int(0.90 * (len(v) - 1))]),
        "min": float(v[0]),
        "max": float(v[-1]),
    }


def safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def safe_mean_int(vals: List[int]) -> float:
    return float(sum(vals) / max(1, len(vals)))


# ----------------- planning call -----------------

def call_plan(service_proxy: rospy.ServiceProxy, mpr: MotionPlanRequest) -> Tuple[int, float, int, float]:
    """
    Calls GetMotionPlan and returns:
      (error_code, planning_time, num_points, wall_time)

    This is the *only* place planners are invoked.
    """
    req = GetMotionPlanRequest()
    req.motion_plan_request = mpr

    t0 = time.time()
    resp = service_proxy(req).motion_plan_response
    wall = time.time() - t0

    code = int(resp.error_code.val)
    ptime = float(resp.planning_time)

    npts = 0
    try:
        npts = len(resp.trajectory.joint_trajectory.points)
    except Exception:
        npts = 0

    rospy.loginfo("Trajectory planning time=%s, code=%s", ptime, code)

    return code, ptime, npts, wall


# ----------------- main benchmark -----------------

def main():
    rospy.init_node("benchmark_100_trials", anonymous=True)
    rospy.sleep(1.0)

    # Publisher: latched summary (works without custom msg)
    pub_summary = rospy.Publisher("/benchmark/summary", String, queue_size=1, latch=True)

    # Services
    ompl_service = rospy.get_param("~ompl_service", "/plan_kinematic_path")
    dgm_service = rospy.get_param("~dgm_service", "/dgm/get_motion_plan")

    rospy.loginfo("Waiting for services: OMPL=%s DGM=%s", ompl_service, dgm_service)
    rospy.wait_for_service(ompl_service, timeout=60.0)
    rospy.wait_for_service(dgm_service, timeout=60.0)

    ompl = rospy.ServiceProxy(ompl_service, GetMotionPlan, persistent=True)
    dgm = rospy.ServiceProxy(dgm_service, GetMotionPlan, persistent=True)

    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")

    # Trials / bounds
    trials = int(rospy.get_param("~trials", 100))
    seed = int(rospy.get_param("~seed", 7))
    random.seed(seed)

    bounds = {
        "x_min": float(rospy.get_param("~workspace_x_min", 0.25)),
        "x_max": float(rospy.get_param("~workspace_x_max", 0.65)),
        "y_min": float(rospy.get_param("~workspace_y_min", -0.35)),
        "y_max": float(rospy.get_param("~workspace_y_max", 0.35)),
        "z_min": float(rospy.get_param("~workspace_z_min", 0.10)),
        "z_max": float(rospy.get_param("~workspace_z_max", 0.60)),
    }

    pos_tol = float(rospy.get_param("~pos_tol", 0.03))

    # Planning params
    allowed_planning_time = float(rospy.get_param("~allowed_planning_time", 2.0))
    num_planning_attempts = int(rospy.get_param("~num_planning_attempts", 3))
    vel_scale = float(rospy.get_param("~vel_scale", 0.3))
    acc_scale = float(rospy.get_param("~acc_scale", 0.3))

    run_id = rospy.get_param("~run_id", f"run_{int(time.time())}")
    out_csv = rospy.get_param("~out_csv", f"/root/catkin_ws/src/object_tracking/results/bench_{run_id}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Start state
    scene = None
    try:
        scene = get_planning_scene("/get_planning_scene")
    except Exception as e:
        rospy.logwarn("Could not fetch planning scene: %s (using default start state)", str(e))
    # start_state = start_state_from_scene_or_default(scene)
    start_state = panda_extended_open_start_state()
    scene.robot_state.joint_state.position = start_state.joint_state.position

    rospy.loginfo("Benchmark config: run_id=%s trials=%d seed=%d", run_id, trials, seed)
    rospy.loginfo("Bounds: %s pos_tol=%.3f", bounds, pos_tol)
    rospy.loginfo("Planning params: allowed_time=%.2f attempts=%d vel=%.2f acc=%.2f",
                  allowed_planning_time, num_planning_attempts, vel_scale, acc_scale)

    rows: List[Dict] = []
    ompl_times_ok: List[float] = []
    dgm_times_ok: List[float] = []
    ompl_pts_ok: List[int] = []
    dgm_pts_ok: List[int] = []
    ompl_success = 0
    dgm_success = 0

    # Per-code histograms
    ompl_code_hist: Dict[int, int] = {}
    dgm_code_hist: Dict[int, int] = {}
    rospy.loginfo("Scene Robot Joint names =%s, Start position = %s",
                  scene.robot_state.joint_state.name,
                  scene.robot_state.joint_state.position)
    rospy.loginfo("Start state Robot State names =%s, Start position = %s",
                  start_state.joint_state.name,
                  start_state.joint_state.position)
    for k in range(trials):
        gx, gy, gz = sample_goal(bounds)
        rospy.loginfo("Benchmark trials, goal: gx= %s, gy=%s, gz=%s", gx, gy, gz)
        goal = make_goal_pose(world_frame, gx, gy, gz)

        # Build MotionPlanRequest once and re-use for both planners
        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        mpr.start_state = start_state
        mpr.goal_constraints = [make_position_only_constraints(goal, ee_link, pos_tol=pos_tol)]
        mpr.allowed_planning_time = allowed_planning_time
        mpr.num_planning_attempts = num_planning_attempts
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = acc_scale

        # ---- OMPL call (uses call_plan) ----
        rospy.loginfo("OMPL call try ===========================")
        try:
            ompl_code, ompl_ptime, ompl_npts, ompl_wall = call_plan(ompl, mpr)
            rospy.loginfo("OMPL call success ===================")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "OMPL call failed: %s", str(e))
            ompl_code, ompl_ptime, ompl_npts, ompl_wall = (-999, 0.0, 0, 0.0)

        ompl_code_hist[ompl_code] = ompl_code_hist.get(ompl_code, 0) + 1
        ompl_ok = (ompl_code == 1) and (ompl_npts > 0)
        if ompl_ok:
            ompl_success += 1
            ompl_times_ok.append(ompl_ptime)
            ompl_pts_ok.append(ompl_npts)

        # ---- DGM call (uses call_plan) ----
        rospy.loginfo("DGM call try **************************")
        try:
            dgm_code, dgm_ptime, dgm_npts, dgm_wall = call_plan(dgm, mpr)
            rospy.loginfo("DGM call success ****************** ")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "DGM call failed: %s", str(e))
            dgm_code, dgm_ptime, dgm_npts, dgm_wall = (-999, 0.0, 0, 0.0)

        dgm_code_hist[dgm_code] = dgm_code_hist.get(dgm_code, 0) + 1
        dgm_ok = (dgm_code == 1) and (dgm_npts > 0)
        if dgm_ok:
            dgm_success += 1
            dgm_times_ok.append(dgm_ptime)
            dgm_pts_ok.append(dgm_npts)

        rospy.loginfo("k=%s, Scene Robot Joint names =%s, Start position = %s",
                      k,
                      scene.robot_state.joint_state.name,
                      scene.robot_state.joint_state.position)
        rospy.loginfo("Start state Robot State names =%s, Start position = %s",
                      start_state.joint_state.name,
                      start_state.joint_state.position)

        rows.append({
            "trial": k,
            # "start_x":
            "goal_x": gx, "goal_y": gy, "goal_z": gz,

            "ompl_code": ompl_code,
            "ompl_planning_time": ompl_ptime,
            "ompl_wall_time": ompl_wall,
            "ompl_points": ompl_npts,

            "dgm_code": dgm_code,
            "dgm_planning_time": dgm_ptime,
            "dgm_wall_time": dgm_wall,
            "dgm_points": dgm_npts,
        })

        if (k + 1) % 10 == 0:
            rospy.loginfo("Progress %d/%d | OMPL succ=%d DGM succ=%d",
                          k + 1, trials, ompl_success, dgm_success)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Stats
    ompl_stats = summarize(ompl_times_ok)
    dgm_stats = summarize(dgm_times_ok)

    summary = {
        "run_id": run_id,
        "trials": trials,
        "seed": seed,
        "bounds": bounds,
        "pos_tol": pos_tol,
        "planning_params": {
            "allowed_planning_time": allowed_planning_time,
            "num_planning_attempts": num_planning_attempts,
            "vel_scale": vel_scale,
            "acc_scale": acc_scale,
        },
        "services": {
            "ompl": ompl_service,
            "dgm": dgm_service,
        },
        "results": {
            "ompl_success": ompl_success,
            "dgm_success": dgm_success,
            "ompl_success_rate": float(ompl_success) / float(trials),
            "dgm_success_rate": float(dgm_success) / float(trials),
            "ompl_time_stats": ompl_stats,
            "dgm_time_stats": dgm_stats,
            "ompl_avg_points": safe_mean_int(ompl_pts_ok),
            "dgm_avg_points": safe_mean_int(dgm_pts_ok),
            "ompl_code_hist": {str(k): int(v) for k, v in sorted(ompl_code_hist.items())},
            "dgm_code_hist": {str(k): int(v) for k, v in sorted(dgm_code_hist.items())},
        },
        "csv": out_csv,
    }

    rospy.loginfo("============================================")
    rospy.loginfo("Benchmark complete. CSV: %s", out_csv)
    rospy.loginfo("OMPL success: %d/%d (%.1f%%)", ompl_success, trials, 100.0 * ompl_success / trials)
    rospy.loginfo("DGM  success: %d/%d (%.1f%%)", dgm_success, trials, 100.0 * dgm_success / trials)
    rospy.loginfo("OMPL codes: %s", json.dumps(summary["results"]["ompl_code_hist"]))
    rospy.loginfo("DGM  codes: %s", json.dumps(summary["results"]["dgm_code_hist"]))
    rospy.loginfo("============================================")

    # Publish latched summary
    pub_summary.publish(String(data=json.dumps(summary)))
    rospy.loginfo("Published /benchmark/summary (latched).")


if __name__ == "__main__":
    main()
