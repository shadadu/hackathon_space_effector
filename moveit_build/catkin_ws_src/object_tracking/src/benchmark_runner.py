#!/usr/bin/env python3
import os
import csv
import time
import rospy
import numpy as np

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from moveit_msgs.msg import RobotState
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from moveit_commander import RobotCommander
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from moveit_msgs.msg import PlanningSceneComponents
from sensor_msgs.msg import JointState

# Use a deterministic, collision-free-ish state as fallback
PANDA_EXTENDED = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
}
PANDA_HAND_OPEN = {
    "panda_finger_joint1": 0.035,
    "panda_finger_joint2": 0.035,
}


def make_panda_start_state() -> RobotState:
    js = JointState()
    names = list(PANDA_EXTENDED.keys()) + list(PANDA_HAND_OPEN.keys())
    js.name = names
    js.position = [PANDA_EXTENDED[n] for n in PANDA_EXTENDED] + [PANDA_HAND_OPEN[n] for n in PANDA_HAND_OPEN]
    rs = RobotState()
    rs.joint_state = js
    return rs


def make_goal_constraints_from_pose(goal_pose: PoseStamped,
                                    link_name: str,
                                    pos_tol: float = 0.01,
                                    ang_tol: float = 0.05) -> Constraints:
    c = Constraints()
    c.name = "goal_pose_constraints"

    pc = PositionConstraint()
    pc.header = goal_pose.header
    pc.link_name = link_name
    pc.target_point_offset.x = 0.0
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0
    pc.weight = 1.0

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    # Keep your original “2*tol” sizing
    box.dimensions = [pos_tol * 2, pos_tol * 2, pos_tol * 2]
    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal_pose.pose)

    oc = OrientationConstraint()
    oc.header = goal_pose.header
    oc.link_name = link_name
    oc.orientation = goal_pose.pose.orientation
    oc.absolute_x_axis_tolerance = ang_tol
    oc.absolute_y_axis_tolerance = ang_tol
    oc.absolute_z_axis_tolerance = ang_tol
    oc.weight = 1.0

    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)
    return c


def get_planning_scene(service_name: str = "/get_planning_scene", timeout: float = 10.0):
    rospy.wait_for_service(service_name, timeout=timeout)
    srv = rospy.ServiceProxy(service_name, GetPlanningScene)

    req = GetPlanningSceneRequest()
    req.components.components = (
            PlanningSceneComponents.SCENE_SETTINGS |
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.WORLD_OBJECT_NAMES |
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    )
    return srv(req).scene


def ensure_gripper_joints(state: RobotState) -> RobotState:
    """Make sure finger joints exist in the RobotState for Panda fake controllers."""
    if state is None:
        return make_panda_start_state()

    if not state.joint_state or not state.joint_state.name:
        return make_panda_start_state()

    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)

    # If missing, append open gripper
    if "panda_finger_joint1" not in names:
        names += ["panda_finger_joint1", "panda_finger_joint2"]
        positions += [PANDA_HAND_OPEN["panda_finger_joint1"], PANDA_HAND_OPEN["panda_finger_joint2"]]

    state.joint_state.name = names
    state.joint_state.position = positions
    return state


def traj_metrics(mpr: MotionPlanRequest, traj) -> dict:
    """
    Basic trajectory metrics:
      - points
      - duration
      - joint path length (sum of euclidean joint deltas)
    """
    out = {"points": 0, "duration_s": 0.0, "joint_path_len": 0.0}

    if traj is None or traj.joint_trajectory is None:
        return out

    pts = traj.joint_trajectory.points
    if not pts:
        return out

    out["points"] = len(pts)
    out["duration_s"] = float(pts[-1].time_from_start.to_sec()) if pts[-1].time_from_start else 0.0

    # Path length in joint space (positions only)
    path_len = 0.0
    for i in range(1, len(pts)):
        q0 = np.array(pts[i - 1].positions, dtype=np.float64)
        q1 = np.array(pts[i].positions, dtype=np.float64)
        if q0.size and q1.size and q0.shape == q1.shape:
            path_len += float(np.linalg.norm(q1 - q0))
    out["joint_path_len"] = path_len
    return out


def call_moveit_get_motion_plan(service_name: str, mpr: MotionPlanRequest,
                                wait_timeout: float,
                                call_timeout: float) -> dict:
    """
    Robust service call wrapper that returns a dict of:
      success, error_code, planning_time, wall_time, trajectory, exception
    """
    res = {
        "service": service_name,
        "success": False,
        "error_code": None,
        "planning_time": None,
        "wall_time": None,
        "trajectory": None,
        "exception": None,
    }

    try:
        rospy.wait_for_service(service_name, timeout=wait_timeout)
    except Exception as e:
        res["exception"] = f"wait_for_service failed: {e}"
        return res

    proxy = rospy.ServiceProxy(service_name, GetMotionPlan, persistent=False)

    req = GetMotionPlanRequest()
    req.motion_plan_request = mpr

    t0 = time.time()
    try:
        # IMPORTANT: enforce call timeout so we never deadlock
        resp = proxy.call(req, timeout=call_timeout)
        dt = time.time() - t0

        mpr_resp = resp.motion_plan_response
        res["wall_time"] = float(dt)
        res["error_code"] = int(mpr_resp.error_code.val)
        res["planning_time"] = float(mpr_resp.planning_time)
        res["trajectory"] = mpr_resp.trajectory
        res["success"] = (mpr_resp.error_code.val == 1)
        return res
    except Exception as e:
        dt = time.time() - t0
        res["wall_time"] = float(dt)
        res["exception"] = f"service call failed: {e}"
        return res


def append_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "stamp",
        "planner",
        "service",
        "success",
        "error_code",
        "planning_time",
        "wall_time",
        "traj_points",
        "traj_duration_s",
        "traj_joint_path_len",
        "goal_x",
        "goal_y",
        "goal_z",
        "world_frame",
        "group_name",
        "ee_link",
        "note",
    ]

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    rospy.init_node("benchmark_runner")
    rospy.sleep(0.5)

    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")

    ompl_service = rospy.get_param("~ompl_service", "/plan_kinematic_path")
    dgm_service = rospy.get_param("~dgm_service", "/dgm/get_motion_plan")

    # Goal params
    goal_x = float(rospy.get_param("~goal_x", 0.5))
    goal_y = float(rospy.get_param("~goal_y", 0.0))
    goal_z = float(rospy.get_param("~goal_z", 0.4))

    # Request params
    allowed_planning_time = float(rospy.get_param("~allowed_planning_time", 2.0))
    num_planning_attempts = int(rospy.get_param("~num_planning_attempts", 5))
    vel_scale = float(rospy.get_param("~vel_scale", 0.3))
    acc_scale = float(rospy.get_param("~acc_scale", 0.3))
    pos_tol = float(rospy.get_param("~pos_tol", 0.01))
    ang_tol = float(rospy.get_param("~ang_tol", 0.05))

    # Robustness params
    wait_timeout = float(rospy.get_param("~wait_timeout", 20.0))
    call_timeout = float(rospy.get_param("~call_timeout", 10.0))

    # CSV logging
    csv_path = rospy.get_param("~csv_path", "/tmp/moveit_benchmark.csv")
    note = rospy.get_param("~note", "")

    # Build goal
    goal = PoseStamped()
    goal.header.frame_id = world_frame
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = goal_x
    goal.pose.position.y = goal_y
    goal.pose.position.z = goal_z
    goal.pose.orientation.w = 1.0

    # Build MotionPlanRequest
    _ = RobotCommander()  # ensures robot model loaded in this process

    mpr = MotionPlanRequest()
    mpr.group_name = group_name

    # Prefer planning scene robot state if available (more complete in headless)
    try:
        scene = get_planning_scene("/get_planning_scene", timeout=min(10.0, wait_timeout))
        start_state = ensure_gripper_joints(scene.robot_state if scene else None)
    except Exception as e:
        rospy.logwarn("Planning scene fetch failed, using fixed start state: %s", str(e))
        start_state = make_panda_start_state()

    mpr.start_state = start_state
    mpr.goal_constraints = [make_goal_constraints_from_pose(goal, ee_link, pos_tol=pos_tol, ang_tol=ang_tol)]
    mpr.allowed_planning_time = allowed_planning_time
    mpr.num_planning_attempts = num_planning_attempts
    mpr.max_velocity_scaling_factor = vel_scale
    mpr.max_acceleration_scaling_factor = acc_scale

    # Run OMPL
    rospy.loginfo("Calling OMPL service: %s", ompl_service)
    ompl = call_moveit_get_motion_plan(ompl_service, mpr, wait_timeout=wait_timeout, call_timeout=call_timeout)

    if ompl["exception"]:
        rospy.logwarn("OMPL exception: %s", ompl["exception"])
    else:
        rospy.loginfo("OMPL error_code=%s planning_time=%.3f wall_time=%.3f",
                      str(ompl["error_code"]), float(ompl["planning_time"]), float(ompl["wall_time"]))

    ompl_m = traj_metrics(mpr, ompl["trajectory"]) if ompl["trajectory"] else {"points": 0, "duration_s": 0.0,
                                                                               "joint_path_len": 0.0}
    if ompl["success"]:
        rospy.loginfo("OMPL traj points=%d duration=%.3f joint_path_len=%.3f",
                      ompl_m["points"], ompl_m["duration_s"], ompl_m["joint_path_len"])

    append_csv(csv_path, {
        "stamp": rospy.Time.now().to_sec(),
        "planner": "OMPL",
        "service": ompl_service,
        "success": ompl["success"],
        "error_code": ompl["error_code"],
        "planning_time": ompl["planning_time"],
        "wall_time": ompl["wall_time"],
        "traj_points": ompl_m["points"],
        "traj_duration_s": ompl_m["duration_s"],
        "traj_joint_path_len": ompl_m["joint_path_len"],
        "goal_x": goal_x, "goal_y": goal_y, "goal_z": goal_z,
        "world_frame": world_frame,
        "group_name": group_name,
        "ee_link": ee_link,
        "note": note,
    })

    # Run DGM
    rospy.loginfo("Calling DGM service: %s", dgm_service)
    dgm = call_moveit_get_motion_plan(dgm_service, mpr, wait_timeout=wait_timeout, call_timeout=call_timeout)

    if dgm["exception"]:
        rospy.logwarn("DGM exception: %s", dgm["exception"])
    else:
        rospy.loginfo("DGM error_code=%s planning_time=%.3f wall_time=%.3f",
                      str(dgm["error_code"]), float(dgm["planning_time"]), float(dgm["wall_time"]))

    dgm_m = traj_metrics(mpr, dgm["trajectory"]) if dgm["trajectory"] else {"points": 0, "duration_s": 0.0,
                                                                            "joint_path_len": 0.0}
    if dgm["success"]:
        rospy.loginfo("DGM traj points=%d duration=%.3f joint_path_len=%.3f",
                      dgm_m["points"], dgm_m["duration_s"], dgm_m["joint_path_len"])

    append_csv(csv_path, {
        "stamp": rospy.Time.now().to_sec(),
        "planner": "DGM",
        "service": dgm_service,
        "success": dgm["success"],
        "error_code": dgm["error_code"],
        "planning_time": dgm["planning_time"],
        "wall_time": dgm["wall_time"],
        "traj_points": dgm_m["points"],
        "traj_duration_s": dgm_m["duration_s"],
        "traj_joint_path_len": dgm_m["joint_path_len"],
        "goal_x": goal_x, "goal_y": goal_y, "goal_z": goal_z,
        "world_frame": world_frame,
        "group_name": group_name,
        "ee_link": ee_link,
        "note": note,
    })

    rospy.loginfo("Benchmark done. CSV appended: %s", csv_path)


if __name__ == "__main__":
    main()
