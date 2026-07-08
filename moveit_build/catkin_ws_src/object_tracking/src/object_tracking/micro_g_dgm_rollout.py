#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
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
    max_nan_guard: int = 5


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

    rospy.loginfo("Rollout micro-g DGM policy with q0=%s, b0=%s, object_odom=%s", q0, b0, object_odom)
    q = np.asarray(q0, dtype=np.float64).reshape(7).copy()
    b = np.asarray(b0, dtype=np.float64).reshape(3).copy()
    p_o, v_o, stamp = object_state_from_odom(object_odom)

    batch = 192
    bt = max(64, batch // 3)
    n_samples = 12
    t_np = np.random.uniform(0.0, 1, (batch, 1)).astype(np.float64)
    t_np = np.sort(t_np.flatten()).reshape((batch, 1))
    # t_np[-n_samples:] = np.ones((n_samples, 1), dtype=np.float64)
    
    tT_np = np.ones((bt, 1), dtype=np.float64)
    # ts = t_np
    ts = np.concatenate([t_np, tT_np])

    N = int(round(cfg.T / cfg.dt)) + 1
    N = max(N, len(ts))
    q_hist = np.zeros((N, 7), dtype=np.float64)
    b_hist = np.zeros((N, 3), dtype=np.float64)
    r_hist = np.zeros((N, 3), dtype=np.float64)

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    k = 0
    dt = ts[k]
    nan_hits = 0
    # for k in range(N):
    for t in ts:

        # t = k * cfg.dt
        tau = max(0.0, cfg.T - t)
        rospy.loginfo("Rollout step %d/%d: t=%s, tau=%s", k, N, t, tau)

        latest_odom = get_latest_object_state(object_odom, object_state_provider)
        p_o = None
        if latest_odom is not object_odom:
            object_odom = latest_odom
            p_o, v_o, stamp = object_state_from_odom(object_odom)
        else:
            msg_age = (rospy.Time.now() - stamp).to_sec()
            p_o, v_o = predict_object_state(object_odom, msg_age + t)

        rospy.loginfo("Predicted object state at t=%s: p_o=%s, v_o=%s", t, p_o, v_o)

        if fk_client is None:
            pose = group.get_current_pose().pose
            p_ee_local = point_to_np(pose.position)
        else:
            p_ee_local = fk_client.ee_position(active_joints, q)
        p_ee = b + p_ee_local
        r = p_ee - p_o
        jac = jacobian_from_group(group, q)


        q_hist[k, :] = q
        b_hist[k, :] = b
        r_hist[k, :] = r

        
        u_q, u_b = policy_np(
            model,
            q,
            b,
            r,
            v_o,
            np.array([tau], dtype=np.float64),
            jac,
            cfg.R_q_diag,
            cfg.R_b_diag,
        )
        u_q = np.asarray(u_q, dtype=np.float64).reshape(7)
        u_b = np.asarray(u_b, dtype=np.float64).reshape(3)

        if not np.all(np.isfinite(u_q)) or not np.all(np.isfinite(u_b)):
            nan_hits += 1
            if nan_hits > cfg.max_nan_guard:
                raise RuntimeError("micro-g DGM rollout: too many non-finite controls")
            u_q = np.zeros(7, dtype=np.float64)
            u_b = np.zeros(3, dtype=np.float64)

        rospy.loginfo("Rollout step %d: u_q=%s, u_b=%s", k, u_q, u_b)
        u_q = clamp(u_q, -cfg.joint_vel_limits, cfg.joint_vel_limits)
        u_b = clamp(u_b, -cfg.base_vel_limits, cfg.base_vel_limits)

        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q.tolist()]
        pt.velocities = [float(x) for x in u_q.tolist()]
        pt.time_from_start = rospy.Duration.from_sec(t)
        traj.joint_trajectory.points.append(pt)

        v_rel = jac.dot(u_q) + u_b - v_o
        if np.linalg.norm(r) <= cfg.grasp_pos_tol and np.linalg.norm(v_rel) <= cfg.grasp_vel_tol:
            q_hist = q_hist[:k + 1]
            b_hist = b_hist[:k + 1]
            r_hist = r_hist[:k + 1]
            break

        if k < N - 1:
            # q = q + cfg.dt * u_q
            # b = b + cfg.dt * u_b

            if k == 0:
                dt = float(np.asarray(ts[k]).reshape(-1)[0])
            else:
                del_t = float(np.asarray(ts[k]).reshape(-1)[0] - np.asarray(ts[k - 1]).reshape(-1)[0])
                if del_t > 0: # 
                    dt = max(
                        1e-3,
                        float(np.asarray(ts[k]).reshape(-1)[0] - np.asarray(ts[k - 1]).reshape(-1)[0]),
                    )
                # keep prev dt if del_t is non-positive, to avoid NaNs and instability in rollouts due to bad ts samples.

            q = q + cfg.dt * u_q
            b = b + cfg.dt * u_b
            if cfg.joint_min is not None and cfg.joint_max is not None:
                q = clamp(q, cfg.joint_min, cfg.joint_max)
            if cfg.base_min is not None and cfg.base_max is not None:
                b = clamp(b, cfg.base_min, cfg.base_max)

        
        # ee_pos, _ = get_final_joint_state_translation(traj)
        # proximity_ee = np.linalg.norm(rs)

        rospy.loginfo("p_o  = %s, b = %s, ee_pos = %s", p_o, b, p_ee)
        # rospy.loginfo(f"micro-g DGM rollout: t={t:.3f}, tau={tau:.3f}, r={r}, v_rel={v_rel}, u_q={u_q}, u_b={u_b}")
        k += 1

    return traj, q_hist, b_hist, r_hist
