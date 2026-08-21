#!/usr/bin/env python3
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import os
import rospy
from geometry_msgs.msg import Point, Vector3
from moveit_msgs.msg import RobotTrajectory
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectoryPoint

from object_tracking.micro_g_dgm_hjb_loss import policy_np


@dataclass
class MicroGRolloutConfig:
    T: float = 2.0
    dt: float = 0.02
    joint_min: np.ndarray = None
    joint_max: np.ndarray = None
    joint_vel_limits: np.ndarray = None
    base_vel_limits: np.ndarray = None
    R_q_diag: np.ndarray = None
    R_b_diag: np.ndarray = None
    base_min: np.ndarray = None
    base_max: np.ndarray = None
    grasp_pos_tol: float = 0.08
    grasp_vel_tol: float = 0.05
    entry_guard_width: float = 0.10
    entry_velocity_weight: float = 10.0
    reach_min: float = 0.20
    reach_max: float = 0.75
    reach_margin: float = 0.02
    require_final_reachable: bool = True
    max_nan_guard: int = 5


_ENTRY_STATS = {
    "rollouts": 0,
    "position_only": 0,
    "grasp_ready": 0,
    "no_position_entry": 0,
}
_ENTRY_STATS_LOCK = Lock()


def record_entry_outcome(outcome: str):
    with _ENTRY_STATS_LOCK:
        _ENTRY_STATS["rollouts"] += 1
        _ENTRY_STATS[outcome] += 1
        total = _ENTRY_STATS["rollouts"]
        position_total = _ENTRY_STATS["position_only"] + _ENTRY_STATS["grasp_ready"]
        position_only_rate = _ENTRY_STATS["position_only"] / float(total)
        grasp_ready_rate = _ENTRY_STATS["grasp_ready"] / float(total)
        conditional_ready_rate = (
            _ENTRY_STATS["grasp_ready"] / float(position_total) if position_total else 0.0
        )
        snapshot = dict(_ENTRY_STATS)
    rospy.loginfo(
        "micro-g entry outcomes: total=%d position_only=%d (%.1f%%) "
        "grasp_ready=%d (%.1f%%) no_position=%d ready_given_position=%.1f%%",
        total,
        snapshot["position_only"],
        100.0 * position_only_rate,
        snapshot["grasp_ready"],
        100.0 * grasp_ready_rate,
        snapshot["no_position_entry"],
        100.0 * conditional_ready_rate,
    )


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def point_to_np(p: Point):
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def vector_to_np(v: Vector3):
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def object_state_from_odom(odom: Odometry):
    p_o = point_to_np(odom.pose.pose.position)
    v_o = vector_to_np(odom.twist.twist.linear)
    stamp = odom.header.stamp if odom.header.stamp != rospy.Time(0) else rospy.Time.now()
    return p_o, v_o, stamp


def predict_object_state(odom: Odometry, dt: float):
    p_o, v_o, _ = object_state_from_odom(odom)
    return p_o + max(0.0, float(dt)) * v_o, v_o


def target_reach_distance(base_position: np.ndarray, object_position: np.ndarray):
    return float(np.linalg.norm(np.asarray(object_position) - np.asarray(base_position)))


def is_target_reachable(distance: float, reach_min: float, reach_max: float, margin: float = 0.0):
    return reach_min - margin <= distance <= reach_max + margin


def get_latest_object_state(
        object_odom: Odometry,
        object_state_provider: Optional[Callable[[], Optional[Odometry]]] = None,
):
    if object_state_provider is None:
        return object_odom
    latest = object_state_provider()
    return latest if latest is not None else object_odom


def jacobian_from_group(group: Any, q: np.ndarray):
    jac = np.asarray(group.get_jacobian_matrix([float(x) for x in q.tolist()]), dtype=np.float64)
    if jac.shape[0] < 3:
        raise RuntimeError(f"Expected at least 3 Jacobian rows, got shape {jac.shape}")
    return jac[:3, :7]

def get_final_joint_state_translation(trajectory):
    rospy.wait_for_service('compute_fk')
    fk_srv = rospy.ServiceProxy('compute_fk', GetPositionFK)

    last_point = trajectory.joint_trajectory.points[-1]
    rospy.loginfo("Joint state last_point = %s", last_point)

    # Build the RobotState message
    robot_state = RobotState()
    robot_state.joint_state.name = trajectory.joint_trajectory.joint_names
    robot_state.joint_state.position = last_point.positions

    # Create the FK Request
    request = GetPositionFKRequest()
    request.fk_link_names = ["panda_hand"]
    request.robot_state = robot_state

    try:
        response = fk_srv(request)
        if response.error_code.val == 1:  # SUCCESS
            translation = response.pose_stamped[0].pose.position
            orientation = response.pose_stamped[0].pose.orientation
            print(f"Joint State EE Position: x={translation.x}, y={translation.y}, z={translation.z}")
            return translation, orientation
    except rospy.ServiceException as e:
        rospy.logerr("FK service call failed: %s" % e)


def rollout_micro_g_dgm_policy(
        model: Any,
        q0: np.ndarray,
        b0: np.ndarray,
        object_odom: Odometry,
        active_joints: List[str],
        group: Any,
        cfg: MicroGRolloutConfig,
        fk_client: Any = None,
        object_state_provider: Optional[Callable[[], Optional[Odometry]]] = None,
) -> Tuple[RobotTrajectory, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out the micro-g value policy with state (q, b, r, v_o, tau).

    If object_state_provider is supplied, the object pose/velocity is re-read at
    each step. That is the hook used by a receding-horizon planner to re-estimate
    p_o and v_o every planning cycle.
    """

    out_path = rospy.get_param(
            "~out_path",
            "/root/catkin_ws/src/object_tracking/models/rollout_micro_g_dgm_policy_path.csv",
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        f.write("dt,o_dt,min_reach_dist,p_o,b,p_ee,p_ee_local\n")

    rospy.loginfo("Rollout micro-g DGM policy with q0=%s, b0=%s, object_odom=%s", q0, b0, object_odom)
    rospy.loginfo("Rollout micro-g DGM policy config: %s", cfg)
    q = np.asarray(q0, dtype=np.float64).reshape(7).copy()
    b = np.asarray(b0, dtype=np.float64).reshape(3).copy()
    p_o, v_o, stamp = object_state_from_odom(object_odom)

    N = int(round(cfg.T / cfg.dt)) + 1
    q_hist = np.zeros((N, 7), dtype=np.float64)
    b_hist = np.zeros((N, 3), dtype=np.float64)
    r_hist = np.zeros((N, 3), dtype=np.float64)

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    last_reach_dist = None
    last_valid_len = 0
    min_ee_dist = None
    nan_hits = 0
    saw_position_goal = False
    last_u_q = None
    last_u_b = None

    for k in range(N):
        t_s = min(k * cfg.dt, cfg.T)
        # tau is remaining time; the timeout boundary used in training is tau=0.
        tau = cfg.T - t_s
        # rospy.loginfo("Rollout step %d/%d: t=%s, tau=%s", k, N, t_s, tau)

        latest_odom = get_latest_object_state(object_odom, object_state_provider)
        p_o = None
        o_dt = 0.0
        msg_age = (rospy.Time.now() - stamp).to_sec()
        o_dt = msg_age
        # p_o, v_o = predict_object_state(object_odom, msg_age + t_s)
        # rospy.loginfo("Predicted object state at t=%s: p_o=%s, v_o=%s", msg_age + t_s, p_o, v_o)
        if latest_odom is not object_odom:
            object_odom = latest_odom
            p_o, v_o, stamp = object_state_from_odom(object_odom)
        else:
            msg_age = (rospy.Time.now() - stamp).to_sec()
            o_dt = msg_age
            p_o, v_o = predict_object_state(object_odom, msg_age + t_s)

        # rospy.loginfo("Rollout step %d/%d: t=%s, tau=%s", k, N, t_s, tau)

        if fk_client is None:
            pose = group.get_current_pose().pose
            p_ee_local = point_to_np(pose.position)
            rospy.loginfo("Rollout step %d/%d: FK from group: p_ee_local=%s", k, N, p_ee_local)
        else:
            p_ee_local = fk_client.ee_position(active_joints, q)
            rospy.loginfo("Rollout step %d/%d: FK from client: p_ee_local=%s", k, N, p_ee_local)
        p_ee = b + p_ee_local
        r = p_ee - p_o

        jac = jacobian_from_group(group, q)

        q_hist[k, :] = q
        b_hist[k, :] = b
        r_hist[k, :] = r

        # At tau=0, use the last applied command to check entry before timeout.
        if tau <= 0.0:
            if last_u_q is not None:
                terminal_v_rel = jac.dot(last_u_q) + last_u_b - v_o
                position_ready = np.linalg.norm(r) <= cfg.grasp_pos_tol
                velocity_ready = np.linalg.norm(terminal_v_rel) <= cfg.grasp_vel_tol
                saw_position_goal = saw_position_goal or position_ready
                if position_ready and velocity_ready:
                    pt = JointTrajectoryPoint()
                    pt.positions = [float(x) for x in q.tolist()]
                    pt.velocities = [float(x) for x in last_u_q.tolist()]
                    pt.time_from_start = rospy.Duration.from_sec(t_s)
                    traj.joint_trajectory.points.append(pt)
                    record_entry_outcome("grasp_ready")
                    return traj, q_hist[:k + 1], b_hist[:k + 1], r_hist[:k + 1]
            last_valid_len = k + 1
            break

        u_q, u_b = policy_np(
            model, q, b, r, v_o, np.array([tau], dtype=np.float64),
            jac, cfg.R_q_diag, cfg.R_b_diag,
            cfg.grasp_pos_tol, cfg.entry_guard_width, cfg.entry_velocity_weight,
        )
        u_q = np.asarray(u_q, dtype=np.float64).reshape(7)
        u_b = np.asarray(u_b, dtype=np.float64).reshape(3)

        if not np.all(np.isfinite(u_q)) or not np.all(np.isfinite(u_b)):
            nan_hits += 1
            if nan_hits > cfg.max_nan_guard:
                raise RuntimeError("micro-g DGM rollout: too many non-finite controls")
            u_q = np.zeros(7, dtype=np.float64)
            u_b = np.zeros(3, dtype=np.float64)

        # rospy.loginfo("Rollout step %d: u_q=%s, u_b=%s", k, u_q, u_b)
        u_q = clamp(u_q, -cfg.joint_vel_limits, cfg.joint_vel_limits)
        u_b = clamp(u_b, -cfg.base_vel_limits, cfg.base_vel_limits)
        # rospy.loginfo("Rollout step clamped %d: u_q=%s, u_b=%s", k, u_q, u_b)
        last_u_q = u_q.copy()
        last_u_b = u_b.copy()

        v_rel = jac.dot(u_q) + u_b - v_o
        position_ready = np.linalg.norm(r) <= cfg.grasp_pos_tol
        velocity_ready = np.linalg.norm(v_rel) <= cfg.grasp_vel_tol
        saw_position_goal = saw_position_goal or position_ready
        rospy.loginfo(
            "Rollout step %d: p_o=%s, b=%s, p_ee=%s, r=%s, v_rel=%s, position_ready=%s, velocity_ready=%s",
            k, p_o, b, p_ee, r, v_rel, position_ready, velocity_ready
        )

        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q.tolist()]
        pt.velocities = [float(x) for x in u_q.tolist()]
        pt.time_from_start = rospy.Duration.from_sec(t_s)
        traj.joint_trajectory.points.append(pt)

        if position_ready and velocity_ready:
            record_entry_outcome("grasp_ready")
            rospy.loginfo("Rollout step %d: position and velocity ready; returning early", k)
            return traj, q_hist[:k + 1], b_hist[:k + 1], r_hist[:k + 1]
        if position_ready:
            rospy.logwarn_throttle(
                1.0,
                "Position tolerance reached but relative speed %.4f exceeds %.4f m/s; continuing rollout",
                np.linalg.norm(v_rel),
                cfg.grasp_vel_tol,
            )

        if k < N - 1:
            q = q + cfg.dt * u_q
            b = b + cfg.dt * u_b
            if cfg.joint_min is not None and cfg.joint_max is not None:
                q = clamp(q, cfg.joint_min, cfg.joint_max)
            if cfg.base_min is not None and cfg.base_max is not None:
                b = clamp(b, cfg.base_min, cfg.base_max)

            if fk_client is None:
                pose = group.get_current_pose().pose
                p_ee_local = point_to_np(pose.position)
            else:
                p_ee_local = fk_client.ee_position(active_joints, q)
            p_ee = b + p_ee_local
            r = p_ee - p_o
            ee_o_dist = np.linalg.norm(r)
            object_base_dist = target_reach_distance(b, p_o)
            last_ee_dist = ee_o_dist
            if min_ee_dist is None or ee_o_dist < min_ee_dist:
                min_ee_dist = ee_o_dist

            # v_rel = jac.dot(u_q) + u_b - v_o
            # position_ready = np.linalg.norm(r) <= cfg.grasp_pos_tol
            # velocity_ready = np.linalg.norm(v_rel) <= cfg.grasp_vel_tol
            # rospy.loginfo("Rollout step %d: r_norm=%.4f, v_rel_norm=%.4f, position_ready=%s, velocity_ready=%s", k, np.linalg.norm(r), np.linalg.norm(v_rel), position_ready, velocity_ready)
            # saw_position_goal = saw_position_goal or position_ready

            # pt = JointTrajectoryPoint()
            # pt.positions = [float(x) for x in q.tolist()]
            # pt.velocities = [float(x) for x in u_q.tolist()]
            # pt.time_from_start = rospy.Duration.from_sec(t_s)
            # traj.joint_trajectory.points.append(pt)

            # if position_ready and velocity_ready:
            #     record_entry_outcome("grasp_ready")
            #     # rospy.loginfo("Rollout step %d: position and velocity ready; returning early", k)
            #     return traj, q_hist[:k + 1], b_hist[:k + 1], r_hist[:k + 1]
            # if position_ready:
            #     rospy.logwarn_throttle(
            #         1.0,
            #         "Position tolerance reached but relative speed %.4f exceeds %.4f m/s; continuing rollout",
            #         np.linalg.norm(v_rel),
            #         cfg.grasp_vel_tol,
            #     )
            

        rospy.loginfo("dt=%.3f, object_dt=%.3f, min_ee_dist=%.3f, last_ee_dist=%.3f, p_o=%s, b=%s, ee_pos=%s, ee_local_pos=%s", cfg.dt, o_dt, min_ee_dist, last_ee_dist, p_o, b, p_ee, p_ee_local)
        rospy.loginfo(
            "Object/base reach distance %.3f m; allowed [%.3f, %.3f] m ; t=%s, tau=%s; last valid rollout length %d",
            object_base_dist, cfg.reach_min, cfg.reach_max, t_s, tau, last_valid_len
        )
        with open(out_path, "a") as f:
                        f.write(
                            f"{cfg.dt},{float(o_dt)},{float(min_ee_dist)},"
                            f"{p_o},{b},{p_ee},{p_ee_local}\n"
                        )
        last_valid_len = k + 1

    f.close()

    if last_valid_len > 0:
        q_hist = q_hist[:last_valid_len]
        b_hist = b_hist[:last_valid_len]
        r_hist = r_hist[:last_valid_len]
    record_entry_outcome("position_only" if saw_position_goal else "no_position_entry")
    if cfg.require_final_reachable and last_reach_dist is not None and not is_target_reachable(
            last_reach_dist,
            cfg.reach_min,
            cfg.reach_max,
            cfg.reach_margin,
    ):
        raise RuntimeError(
            "micro-g DGM rollout: final object/base distance {:.3f} outside reach shell [{:.3f}, {:.3f}]".format(
                last_reach_dist,
                cfg.reach_min,
                cfg.reach_max,
            )
        )
    return traj, q_hist, b_hist, r_hist
