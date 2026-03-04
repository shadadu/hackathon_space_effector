#!/usr/bin/env python3
import os
import time
import pickle
import random
from typing import Dict, List, Tuple, Optional

import rospy
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import RobotCommander, MoveGroupCommander

from jacobian_server.srv import GetJacobian, GetJacobianRequest


def panda_joint_limits() -> Tuple[List[float], List[float]]:
    # Common Panda limits (rad)
    jmin = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    jmax = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
    return jmin, jmax


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


def sample_q0(active_joints: List[str], jmin: List[float], jmax: List[float], margin: float = 0.05) -> Dict[str, float]:
    q = {}
    for i, jn in enumerate(active_joints):
        lo = jmin[i] + margin * (jmax[i] - jmin[i])
        hi = jmax[i] - margin * (jmax[i] - jmin[i])
        q[jn] = random.uniform(lo, hi)
    q["panda_finger_joint1"] = 0.035
    q["panda_finger_joint2"] = 0.035
    return q


def sample_goal_pos(bounds: Dict[str, float]) -> Tuple[float, float, float]:
    x = random.uniform(bounds["x_min"], bounds["x_max"])
    y = random.uniform(bounds["y_min"], bounds["y_max"])
    z = random.uniform(bounds["z_min"], bounds["z_max"])
    return x, y, z


def make_goal_pose(world_frame: str, x: float, y: float, z: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = world_frame
    p.header.stamp = rospy.Time.now()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z
    p.pose.orientation.w = 1.0
    return p


def robot_state_from_active_joint_positions(active_joints: List[str], q: Dict[str, float]) -> RobotState:
    js = JointState()
    js.name = []
    js.position = []
    for jn in active_joints:
        js.name.append(jn)
        js.position.append(float(q[jn]))
    for gj in ("panda_finger_joint1", "panda_finger_joint2"):
        if gj in q:
            js.name.append(gj)
            js.position.append(float(q[gj]))
    rs = RobotState()
    rs.joint_state = js
    return rs


def ik_solve(
    ik_proxy: rospy.ServiceProxy,
    group_name: str,
    ee_link: str,
    goal_pose: PoseStamped,
    seed_state: RobotState,
    timeout_s: float = 0.15,
) -> Tuple[bool, Optional[RobotState], int]:
    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ee_link
    req.ik_request.pose_stamped = goal_pose
    req.ik_request.robot_state = seed_state
    req.ik_request.timeout = rospy.Duration(timeout_s)

    resp = ik_proxy(req)
    ok = (resp.error_code.val == 1)
    return ok, (resp.solution if ok else None), resp.error_code.val


def rollout_joint_line(q0: List[float], q1: List[float], T: float, dt: float) -> Tuple[List[float], List[List[float]]]:
    steps = int(round(T / dt))
    ts = [i * dt for i in range(steps + 1)]
    qs = []
    for i, t in enumerate(ts):
        a = 0.0 if steps == 0 else (i / float(steps))
        q = [(1 - a) * q0[j] + a * q1[j] for j in range(len(q0))]
        qs.append(q)
    return ts, qs


def jacobian_at(
    jac_proxy: rospy.ServiceProxy,
    group_name: str,
    link_name: str,
    joint_names: List[str],
    joint_positions: List[float],
    reference_point: Optional[Point] = None,
) -> Tuple[bool, List[float], int, int, str]:
    req = GetJacobianRequest()
    req.group_name = group_name
    req.link_name = link_name
    req.joint_names = list(joint_names)
    req.joint_positions = list(joint_positions)
    req.reference_point = reference_point if reference_point is not None else Point(0.0, 0.0, 0.0)

    resp = jac_proxy(req)
    ok = (resp.message == "OK") and (resp.rows > 0) and (resp.cols > 0) and (len(resp.jacobian) == resp.rows * resp.cols)
    return ok, list(resp.jacobian), int(resp.rows), int(resp.cols), resp.message


def main():
    rospy.init_node("dgm_dataset_gen", anonymous=True)

    out_path = rospy.get_param("~out", "/root/catkin_ws/src/objecttracking/data/panda_train_withJ.pkl")
    num_samples = int(rospy.get_param("~num_samples", 2000))
    max_attempts = int(rospy.get_param("~max_attempts", num_samples * 30))

    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")
    ik_service = rospy.get_param("~ik_service", "/compute_ik")
    jac_service = rospy.get_param("~jacobian_service", "/get_jacobian")

    ik_timeout = float(rospy.get_param("~ik_timeout", 0.15))

    # rollout params
    T = float(rospy.get_param("~T", 2.0))
    dt = float(rospy.get_param("~dt", 0.02))

    # For position-only first: store only top 3 rows (linear velocity Jacobian)
    store_position_only = bool(rospy.get_param("~store_position_only", True))

    bounds = {
        "x_min": float(rospy.get_param("~workspace_x_min", 0.25)),
        "x_max": float(rospy.get_param("~workspace_x_max", 0.65)),
        "y_min": float(rospy.get_param("~workspace_y_min", -0.35)),
        "y_max": float(rospy.get_param("~workspace_y_max", 0.35)),
        "z_min": float(rospy.get_param("~workspace_z_min", 0.10)),
        "z_max": float(rospy.get_param("~workspace_z_max", 0.60)),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rospy.loginfo("Waiting for IK: %s", ik_service)
    rospy.wait_for_service(ik_service, timeout=60.0)
    ik = rospy.ServiceProxy(ik_service, GetPositionIK)

    rospy.loginfo("Waiting for Jacobian: %s", jac_service)
    rospy.wait_for_service(jac_service, timeout=60.0)
    jac = rospy.ServiceProxy(jac_service, GetJacobian)

    robot = RobotCommander()
    group = MoveGroupCommander(group_name)
    active_joints = group.get_active_joints()
    if len(active_joints) != 7:
        rospy.logwarn("Expected 7 active joints, got %d: %s", len(active_joints), active_joints)

    jmin, jmax = panda_joint_limits()
    margin = float(rospy.get_param("~joint_margin", 0.05))

    dataset: List[Dict] = []
    attempts = 0
    accepted = 0
    t0 = time.time()

    rospy.loginfo("Generating dataset with Jacobians:")
    rospy.loginfo("  out=%s", out_path)
    rospy.loginfo("  samples=%d (max_attempts=%d)", num_samples, max_attempts)
    rospy.loginfo("  rollout T=%.3f dt=%.3f steps=%d", T, dt, int(round(T / dt)) + 1)
    rospy.loginfo("  store_position_only=%s", store_position_only)

    while accepted < num_samples and attempts < max_attempts and not rospy.is_shutdown():
        attempts += 1

        q0_map = sample_q0(active_joints, jmin, jmax, margin=margin)
        start_state = robot_state_from_active_joint_positions(active_joints, q0_map)

        gx, gy, gz = sample_goal_pos(bounds)
        goal_pose = make_goal_pose(world_frame, gx, gy, gz)

        ok_ik, sol_state, _code = ik_solve(
            ik_proxy=ik,
            group_name=group_name,
            ee_link=ee_link,
            goal_pose=goal_pose,
            seed_state=start_state,
            timeout_s=ik_timeout,
        )
        if not ok_ik:
            continue

        sol_map = dict(zip(sol_state.joint_state.name, sol_state.joint_state.position))
        try:
            q_goal = [float(sol_map[jn]) for jn in active_joints]
        except KeyError:
            continue

        q0 = [float(q0_map[jn]) for jn in active_joints]

        # Rollout in joint space (simple for now; replace later with policy rollout)
        ts, qs = rollout_joint_line(q0, q_goal, T=T, dt=dt)

        # Jacobian snapshots along rollout
        Js = []
        rows = cols = None

        for q in qs:
            okJ, Jflat, r, c, msg = jacobian_at(
                jac_proxy=jac,
                group_name=group_name,
                link_name=ee_link,
                joint_names=active_joints,
                joint_positions=q,
                reference_point=Point(0.0, 0.0, 0.0),
            )
            if not okJ:
                Js = []
                break

            if rows is None:
                rows, cols = r, c

            # position-only: keep top 3 rows (vx,vy,vz) => 3x7
            if store_position_only:
                # Jflat is row-major [r x c]
                Jpos = []
                for rr in range(min(3, r)):
                    base = rr * c
                    Jpos.extend(Jflat[base:base + c])
                Js.append(Jpos)
            else:
                Js.append(Jflat)

        if not Js:
            continue

        sample = {
            "q0": q0,
            "goal_pos": [gx, gy, gz],
            "q_goal": q_goal,
            "rollout": {
                "t": ts,
                "q": qs,
                "J": Js,
                "J_rows": (3 if store_position_only else rows),
                "J_cols": cols,
                "store_position_only": store_position_only,
            },
            "meta": {
                "group": group_name,
                "ee_link": ee_link,
                "world_frame": world_frame,
                "created": time.time(),
            },
        }

        dataset.append(sample)
        accepted += 1

        if accepted % 100 == 0:
            elapsed = time.time() - t0
            rospy.loginfo("Accepted %d/%d (attempts=%d) rate=%.1f/s",
                          accepted, num_samples, attempts, accepted / max(elapsed, 1e-9))

    elapsed = time.time() - t0
    rospy.loginfo("Done. accepted=%d attempts=%d elapsed=%.1fs", accepted, attempts, elapsed)

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "meta": {
                    "created": time.time(),
                    "group": group_name,
                    "ee_link": ee_link,
                    "world_frame": world_frame,
                    "bounds": bounds,
                    "num_samples": accepted,
                    "attempts": attempts,
                    "ik_service": ik_service,
                    "jacobian_service": jac_service,
                    "T": T,
                    "dt": dt,
                    "store_position_only": store_position_only,
                },
                "data": dataset,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    rospy.loginfo("Wrote dataset: %s", out_path)


if __name__ == "__main__":
    main()
