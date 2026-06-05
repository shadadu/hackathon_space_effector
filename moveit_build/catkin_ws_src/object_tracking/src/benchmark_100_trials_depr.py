#!/usr/bin/env python3
import os
import csv
import time
import random
import statistics
from typing import Dict, List, Tuple

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints, PositionConstraint
from moveit_msgs.msg import MotionPlanRequest, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from moveit_msgs.msg import PlanningSceneComponents
from shape_msgs.msg import SolidPrimitive

# NEW msg (import test message)
from object_tracking.msg import BenchmarkSummary
# from objecttracking.trajectory_executor_manager import TrajectoryExecutorManager


# ----------------- utilities -----------------
def decode(code):
    for k, v in MoveItErrorCodes.__dict__.items():
        if isinstance(v, int) and v == code:
            return k
    return str(code)
def make_robot_state_from_joint_dict(joint_dict: Dict[str, float]) -> RobotState:
    js = JointState()
    js.name = list(joint_dict.keys())
    js.position = [joint_dict[n] for n in js.name]
    rs = RobotState()
    rs.joint_state = js
    return rs


def panda_extended_open_start_state() -> RobotState:
    joints = {
        "panda_joint1": 0.0,
        "panda_joint2": 0.0,
        "panda_joint3": 0.0,
        "panda_joint4": 0.0,
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
    if scene and scene.robot_state and scene.robot_state.joint_state.name:
        rs = scene.robot_state
        rospy.loginfo("Scene initial robot state %s: %s", rs.joint_state.name, rs.joint_state.position )
        rs = panda_extended_open_start_state()
        rospy.loginfo("Panda extended open start robot state: %s %s", rs.joint_state.name, rs.joint_state.position)
        names = list(rs.joint_state.name)
        pos = list(rs.joint_state.position)

        if "panda_finger_joint1" not in names:
            names += ["panda_finger_joint1", "panda_finger_joint2"]
            pos += [0.035, 0.035]

        rs.joint_state.name = names
        rs.joint_state.position = pos
        rospy.loginfo("Scene: updated robot start state positions %s with finger joints", rs.joint_state.position)
        return rs
    panda_eos = panda_extended_open_start_state()
    rospy.loginfo("Scene: setting start state position to extended open start default %s", panda_eos)
    return panda_eos


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
    box.dimensions = [pos_tol, pos_tol, pos_tol]
    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal.pose)

    c.position_constraints.append(pc)
    return c


def call_plan(service_proxy: rospy.ServiceProxy, mpr: MotionPlanRequest) -> Tuple[int, float, int]:
    """
    Returns: (error_code, planning_time, num_points)
    """
    req = GetMotionPlanRequest()
    req.motion_plan_request = mpr
    resp = service_proxy(req).motion_plan_response
    code = int(resp.error_code.val)
    ptime = float(resp.planning_time)

    npts = 0
    try:
        npts = len(resp.trajectory.joint_trajectory.points)
    except Exception:
        npts = 0
    return code, ptime, npts


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    return {
        "n": n,
        "mean": statistics.mean(vals_sorted),
        "median": statistics.median(vals_sorted),
        "p90": vals_sorted[int(0.90 * (n - 1))],
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


def safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / max(1, len(vals))) if vals else 0.0


# ----------------- main benchmark -----------------

def main():
    rospy.init_node("benchmark_100_trials", anonymous=True)
    rospy.sleep(0.5)

    # Publish summary (latched so it persists after script exits)
    summary_pub = rospy.Publisher("/benchmark/summary", BenchmarkSummary, queue_size=1, latch=True)

    run_id = rospy.get_param("~run_id", f"run_{int(time.time())}")
    planner_a = rospy.get_param("~planner_a", "OMPL")
    planner_b = rospy.get_param("~planner_b", "DGM")

    # Services
    ompl_service = rospy.get_param("~ompl_service", "/plan_kinematic_path")
    dgm_service = rospy.get_param("~dgm_service",  "/dgm/get_motion_plan")

    rospy.loginfo("Waiting for services: OMPL=%s DGM=%s", ompl_service, dgm_service)
    rospy.wait_for_service(ompl_service, timeout=60.0)
    try:
        rospy.wait_for_service(dgm_service, timeout=30.0)
    except rospy.ROSException:
        rospy.logerr("DGM service not available: %s", dgm_service)
        # print the closest matches for debugging
        try:
            import rosservice
            svcs = rosservice.get_service_list()
            cand = [s for s in svcs if "dgm" in s.lower() or "motion_plan" in s.lower()]
            rospy.logerr("Services containing dgm/motion_plan: %s", cand[:50])
        except Exception as e:
            rospy.logerr("Could not list services: %s", e)
        raise

    ompl = rospy.ServiceProxy(ompl_service, GetMotionPlan, persistent=True)
    dgm = rospy.ServiceProxy(dgm_service,  GetMotionPlan, persistent=True)

    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")

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

    out_csv = rospy.get_param("~out_csv", "/root/catkin_ws/src/object_tracking/results/bench_100.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Start state
    scene = get_planning_scene("/get_planning_scene")
    # start_state = start_state_from_scene_or_default(scene) # start state:RobotState
    start_state = panda_extended_open_start_state()

    rows = []
    ompl_times_ok: List[float] = []
    dgm_times_ok: List[float] = []
    ompl_pts_ok: List[int] = []
    dgm_pts_ok: List[int] = []
    ompl_success = 0
    dgm_success = 0

    rospy.loginfo("Benchmark config: run_id=%s trials=%d seed=%d", run_id, trials, seed)
    rospy.loginfo("Bounds: %s pos_tol=%.3f", bounds, pos_tol)
    rospy.loginfo("Planning params: allowed_time=%.2f attempts=%d vel=%.2f acc=%.2f",
                  allowed_planning_time, num_planning_attempts, vel_scale, acc_scale)

    gx, gy, gz = sample_goal(bounds)
    goal = make_goal_pose(world_frame, gx, gy, gz)
    rospy.loginfo("Scene Robot Joint names =%s, Start position = %s",
                  scene.robot_state.joint_state.name,
                  scene.robot_state.joint_state.position)
    rospy.loginfo("Start state Robot State names =%s, Start position = %s",
                  start_state.joint_state.name,
                  start_state.joint_state.position)

    for k in range(trials):
        # gx, gy, gz = sample_goal(bounds)
        # goal = make_goal_pose(world_frame, gx, gy, gz)

        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        mpr.start_state = start_state
        mpr.goal_constraints = [make_position_only_constraints(goal, ee_link, pos_tol=pos_tol)]
        mpr.allowed_planning_time = allowed_planning_time
        mpr.num_planning_attempts = num_planning_attempts
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = acc_scale

        # OMPL call
        t0 = time.time()
        try:
            ompl_code, ompl_ptime, ompl_npts = call_plan(ompl, mpr)
        except Exception:
            ompl_code, ompl_ptime, ompl_npts = (-999, 0.0, 0)
        ompl_wall = time.time() - t0

        # DGM call
        t0 = time.time()
        try:
            dgm_code, dgm_ptime, dgm_npts = call_plan(dgm, mpr)
        except Exception:
            dgm_code, dgm_ptime, dgm_npts = (-999, 0.0, 0)
        dgm_wall = time.time() - t0

        ompl_ok = (ompl_code == 1)
        dgm_ok = (dgm_code == 1)

        if ompl_ok:
            ompl_success += 1
            ompl_times_ok.append(ompl_ptime)
            ompl_pts_ok.append(ompl_npts)

        if dgm_ok:
            dgm_success += 1
            dgm_times_ok.append(dgm_ptime)
            dgm_pts_ok.append(dgm_npts)

        row_data = {
            "trial": k,
            "goal_x": gx, "goal_y": gy, "goal_z": gz,
            "ompl_code": ompl_code, "ompl_code_msg": decode(ompl_code), "ompl_planning_time": ompl_ptime, "ompl_wall_time": ompl_wall, "ompl_points": ompl_npts,
            "dgm_code": dgm_code, "dgm_code_msg": decode(dgm_code), "dgm_planning_time": dgm_ptime, "dgm_wall_time": dgm_wall, "dgm_points": dgm_npts,
        }

        rows.append(row_data)

        rospy.loginfo("Row data @ trial %s %s", k, str(row_data)+"\n")

        if (k + 1) % 10 == 0:
            rospy.loginfo("Progress %d/%d | OMPL succ=%d DGM succ=%d",
                          k + 1, trials, ompl_success, dgm_success)

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # stats
    ompl_stats = summarize(ompl_times_ok)
    dgm_stats = summarize(dgm_times_ok)

    rospy.loginfo("============================================")
    rospy.loginfo("Benchmark complete. CSV: %s", out_csv)
    rospy.loginfo("Trials=%d seed=%d", trials, seed)
    rospy.loginfo("OMPL success: %d/%d (%.1f%%)", ompl_success, trials, 100.0 * ompl_success / trials)
    rospy.loginfo("DGM  success: %d/%d (%.1f%%)", dgm_success, trials, 100.0 * dgm_success / trials)
    rospy.loginfo("OMPL planning_time stats: %s", ompl_stats)
    rospy.loginfo("DGM  planning_time stats: %s", dgm_stats)
    rospy.loginfo("OMPL avg points: %.1f", safe_mean([float(x) for x in ompl_pts_ok]))
    rospy.loginfo("DGM  avg points: %.1f", safe_mean([float(x) for x in dgm_pts_ok]))
    rospy.loginfo("============================================")

    # Publish summary (latched)
    msg = BenchmarkSummary()
    msg.header = Header(stamp=rospy.Time.now(), frame_id=world_frame)
    msg.run_id = run_id
    msg.planner_a = planner_a
    msg.planner_b = planner_b
    msg.n_trials = trials
    msg.success_a = ompl_success
    msg.success_b = dgm_success
    msg.intercept_success_a = 0
    msg.intercept_success_b = 0
    msg.mean_plan_time_a = float(ompl_stats.get("mean", 0.0))
    msg.std_plan_time_a = float(statistics.pstdev(ompl_times_ok) if len(ompl_times_ok) > 1 else 0.0)
    msg.mean_plan_time_b = float(dgm_stats.get("mean", 0.0))
    msg.std_plan_time_b = float(statistics.pstdev(dgm_times_ok) if len(dgm_times_ok) > 1 else 0.0)
    msg.mean_path_length_a = 0.0
    msg.mean_path_length_b = 0.0
    msg.mean_time_to_intercept_a = 0.0
    msg.mean_time_to_intercept_b = 0.0
    msg.notes = f"csv={out_csv} seed={seed} pos_tol={pos_tol}"

    summary_pub.publish(msg)
    rospy.loginfo("Published /benchmark/summary (latched).")


if __name__ == "__main__":
    main()
