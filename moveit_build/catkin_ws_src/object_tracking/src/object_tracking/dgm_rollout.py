#!/usr/bin/env python3
import numpy as np
import math
import torch
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from typing import List, Tuple, Optional
import datetime as datetime
from .dgm_model import build_input, DGMValueNet, ValueNet, ValueNet_

from dataclasses import dataclass

from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from moveit_msgs.msg import RobotTrajectory, RobotState


@dataclass
class RolloutConfig:
    T: float = 2.0
    dt: float = 0.02
    vel_limits: np.ndarray = None  # (7,)
    joint_min: np.ndarray = None  # (7,)
    joint_max: np.ndarray = None  # (7,)
    R_diag: np.ndarray = None  # (7,)
    max_nan_guard: int = 5


def euclidean_dist(ee_position, goal):
    d_sq = ((ee_position.x - goal[0]) ** 2 +
            (ee_position.y - goal[1]) ** 2 +
            (ee_position.z - goal[2]) ** 2)

    return math.sqrt(d_sq)


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

def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def rollout_value_policy(model, q0, goal_pos, active_joints,
                         T=2.0, dt=0.02, R_diag=None,
                         vel_limits=None, joint_min=None, joint_max=None,
                         device="cpu"):
    """
    u* = -0.5 R^{-1} grad_q V, qdot=u
    """
    if R_diag is None:
        R_diag = np.array([0.15] * 7, dtype=np.float64)
    if vel_limits is None:
        vel_limits = np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0], dtype=np.float64)

    R_inv = 1.0 / np.maximum(R_diag, 1e-9)

    N = int(round(T / dt)) + 1
    q = np.array(q0, dtype=np.float64).copy()

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    for k in range(N):
        t = k * dt
        # Torch (requires_grad for q)
        qt = torch.tensor(q[None, :], dtype=torch.float32, device=device, requires_grad=True)
        tt = torch.tensor([[t / T]], dtype=torch.float32, device=device, requires_grad=True)
        gt = torch.tensor(goal_pos[None, :], dtype=torch.float32, device=device)

        V = model(build_input(qt, tt, gt))  # (1,)
        grad_q = torch.autograd.grad(V.sum(), qt, create_graph=False)[0].detach().cpu().numpy().reshape(7)

        if not np.all(np.isfinite(grad_q)):
            u = np.zeros(7, dtype=np.float64)
        else:
            u = -0.5 * R_inv * grad_q

        u = clamp(u, -vel_limits, vel_limits)

        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.velocities = u.tolist()
        pt.time_from_start = rospy.Duration.from_sec(t)
        traj.joint_trajectory.points.append(pt)

        if k < N - 1:
            q = q + dt * u
            if joint_min is not None and joint_max is not None:
                q = clamp(q, joint_min, joint_max)

    return traj


def rollout_dgm_joint_policy(
        model: ValueNet,
        q0: np.ndarray,
        goal_pos: np.ndarray,
        active_joints: List[str],
        cfg: RolloutConfig,
        proximity_threshold: float = 0.10,
        device: str = "cpu",
) -> Tuple[RobotTrajectory, np.ndarray]:
    """
    Roll out u* = -0.5 * R^{-1} * grad_q V(q,t,gpos)
    Dynamics: qdot = u

    Returns:
      (RobotTrajectory, q_traj_np)
    """
    assert q0.shape == (7,), f"Expected q0 shape (7,), got {q0.shape}"
    assert goal_pos.shape == (3,), f"Expected goal_pos shape (3,), got {goal_pos.shape}"
    assert cfg.vel_limits is not None and cfg.R_diag is not None
    assert cfg.joint_min is not None and cfg.joint_max is not None

    rospy.loginfo("Executing DGM Value Net rollout ")

    N = int(round(cfg.T / cfg.dt)) + 1
    q = q0.astype(np.float64).copy()

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    q_hist = np.zeros((N, 7), dtype=np.float64)
    nan_hits = 0

    # Precompute R^{-1}
    R_inv = 1.0 / np.maximum(cfg.R_diag.astype(np.float64), 1e-9)

    k = 0
    proximity_ee = float('inf')

    model.apply(enable_dropout)

    model.eval()

    while (k < N) and (proximity_threshold < proximity_ee):
        # for k in range(N):

        t = k * cfg.dt
        q_hist[k, :] = q

        # Add trajectory point at current q (velocities set below)
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.time_from_start = rospy.Duration.from_sec(t)

        # Torch inputs
        qt = torch.tensor(q[None, :], dtype=torch.float32, device=device, requires_grad=True)
        tt = torch.tensor([[t / cfg.T]], dtype=torch.float32, device=device)  # normalize time to [0,1]
        gt = torch.tensor(goal_pos[None, :], dtype=torch.float32, device=device)

        x = build_input(qt, tt, gt)

        model.apply(enable_dropout) # add dropout to enable stochasticity
        # with torch.no_grad():
        V = model(x)  # (1,)

        # grad_q V
        grad_q = torch.autograd.grad(V.sum(), qt, create_graph=False, retain_graph=False)[0]  # (1,7)
        grad_q_np = grad_q.detach().cpu().numpy().reshape(7)

        if not np.all(np.isfinite(grad_q_np)):
            nan_hits += 1
            if nan_hits > cfg.max_nan_guard:
                raise RuntimeError("DGM rollout: too many non-finite gradients; aborting.")
            # fallback: zero velocity
            u = np.zeros(7, dtype=np.float64)
        else:
            # u* = -0.5 R^{-1} grad
            u = -0.5 * R_inv * grad_q_np

        # clamp velocities
        u = clamp(u, -cfg.vel_limits, cfg.vel_limits)
        # rospy.loginfo("Clamped velocities u = \%s", u)

        pt.velocities = u.tolist()
        traj.joint_trajectory.points.append(pt)

        ee_pos, _ = get_final_joint_state_translation(traj)
        # rospy.loginfo("goal_pos  = %s, ee_pos = %s", ee_pos)
        proximity_ee = euclidean_dist(ee_pos, goal_pos)
        rospy.loginfo("goal_pos  = %s, ee_pos = %s, proximity_ee: %s", goal_pos, ee_pos, proximity_ee)

        # integrate forward (except after last point)
        if k < N - 1:
            q = q + cfg.dt * u
            q = clamp(q, cfg.joint_min, cfg.joint_max)

        k += 1

    rospy.loginfo("traj last point = %s", traj.joint_trajectory.points[-1])

    return traj, q_hist


def rollout_dgm_batch_joint_policy(
        model: ValueNet,
        q0: np.ndarray,
        goal_pos: np.ndarray,
        active_joints: List[str],
        cfg: RolloutConfig,
        proximity_threshold: float = 0.10,
        batch: int = 256,
        device: str = "cpu",
) -> Tuple[RobotTrajectory, np.ndarray]:
    """
    Roll out u* = -0.5 * R^{-1} * grad_q V(q,t,gpos)
    Dynamics: qdot = u

    Returns:
      (RobotTrajectory, q_traj_np)
    """
    assert q0.shape == (7,), f"Expected q0 shape (7,), got {q0.shape}"
    assert goal_pos.shape == (3,), f"Expected goal_pos shape (3,), got {goal_pos.shape}"
    assert cfg.vel_limits is not None and cfg.R_diag is not None
    assert cfg.joint_min is not None and cfg.joint_max is not None

    rospy.loginfo("Executing DGM Value Net rollout ")

    N = int(round(cfg.T / cfg.dt)) + 1
    q = q0.astype(np.float64).copy()

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    q_hist = np.zeros((N, 7), dtype=np.float64)
    nan_hits = 0

    # Precompute R^{-1}
    R_inv = 1.0 / np.maximum(cfg.R_diag.astype(np.float64), 1e-9)

    k = 0
    proximity_ee = float('inf')

    model.apply(enable_dropout)

    model.eval()

    n_samples = 8

    t_np = np.random.uniform(0.0, 1, (batch, 1)).astype(np.float64)
    t_np = np.sort(t_np.flatten()).reshape((batch, 1))
    t_np[:-n_samples] = np.ones((n_samples, 1), dtype=np.float64)
    bt = max(64, batch // 3)
    tT_np = np.ones((bt, 1), dtype=np.float64)
    ts = np.concatenate([t_np, tT_np])

    for t in ts:

        q_hist[k, :] = q

        # Add trajectory point at current q (velocities set below)
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.time_from_start = rospy.Duration.from_sec(t)

        # Torch inputs
        qt = torch.tensor(q[None, :], dtype=torch.float32, device=device, requires_grad=True)
        tt = torch.tensor([[t]], dtype=torch.float32, device=device)  # normalize time to [0,1]
        gt = torch.tensor(goal_pos[None, :], dtype=torch.float32, device=device)

        x = build_input(qt, tt, gt)

        model.apply(enable_dropout) # add dropout to enable stochasticity
        # with torch.no_grad():
        V = model(x)  # (1,)

        # grad_q V
        grad_q = torch.autograd.grad(V.sum(), qt, create_graph=False, retain_graph=False)[0]  # (1,7)
        grad_q_np = grad_q.detach().cpu().numpy().reshape(7)

        if not np.all(np.isfinite(grad_q_np)):
            nan_hits += 1
            if nan_hits > cfg.max_nan_guard:
                raise RuntimeError("DGM rollout: too many non-finite gradients; aborting.")
            # fallback: zero velocity
            u = np.zeros(7, dtype=np.float64)
        else:
            # u* = -0.5 R^{-1} grad
            u = -0.5 * R_inv * grad_q_np

        # clamp velocities
        u = clamp(u, -cfg.vel_limits, cfg.vel_limits)
        # rospy.loginfo("Clamped velocities u = \%s", u)

        pt.velocities = u.tolist()
        traj.joint_trajectory.points.append(pt)

        ee_pos, _ = get_final_joint_state_translation(traj)
        # rospy.loginfo("goal_pos  = %s, ee_pos = %s", ee_pos)
        proximity_ee = euclidean_dist(ee_pos, goal_pos)
        rospy.loginfo("goal_pos  = %s, ee_pos = %s, proximity_ee: %s", goal_pos, ee_pos, proximity_ee)

        # integrate forward (except after last point)
        if k < N - 1:
            q = q + cfg.dt * u
            q = clamp(q, cfg.joint_min, cfg.joint_max)

        k += 1

    rospy.loginfo("traj last point = %s", traj.joint_trajectory.points[-1])

    return traj, q_hist