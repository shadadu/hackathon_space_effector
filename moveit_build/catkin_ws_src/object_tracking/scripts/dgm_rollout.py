#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np
import torch

import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# from .dgm_model import build_input, DGMValueNet

# from object_tracking.scripts.dgm_model import load_model, DGMValueNet, build_input
from object_tracking.dgm_model import DGMValueNet


from dataclasses import dataclass


@dataclass
class RolloutConfig:
    T: float = 2.0
    dt: float = 0.02
    vel_limits: np.ndarray | None = None
    joint_min: np.ndarray | None = None
    joint_max: np.ndarray | None = None
    R_diag: np.ndarray | None = None
    max_nan_guard: int = 5


def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def rollout_dgm_joint_policy(
        model: DGMValueNet,
        q0: np.ndarray,
        goal_pos: np.ndarray,
        active_joints: List[str],
        cfg: RolloutConfig,
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

    for k in range(N):
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

        x = DGMValueNet.build_input(qt, tt, gt)
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
        pt.velocities = u.tolist()
        traj.joint_trajectory.points.append(pt)

        # integrate forward (except after last point)
        if k < N - 1:
            q = q + cfg.dt * u
            q = clamp(q, cfg.joint_min, cfg.joint_max)

    return traj, q_hist
