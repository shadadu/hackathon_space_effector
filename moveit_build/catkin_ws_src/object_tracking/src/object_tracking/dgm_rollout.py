import numpy as np
import torch
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from .dgm_model import build_input


class DGMRollout:
    class RolloutConfig:
        def __init__(self):
            super().__init__()

        T: float = 2.0
        dt: float = 0.02
        vel_limits: np.ndarray = None  # (7,)
        joint_min: np.ndarray = None  # (7,)
        joint_max: np.ndarray = None  # (7,)
        R_diag: np.ndarray = None  # (7,)
        max_nan_guard: int = 5

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


# @dataclass
# class RolloutConfig():
#     def __init__(self):
#         super().__init__()
#
#     T: float = 2.0
#     dt: float = 0.02
#     vel_limits: np.ndarray = None  # (7,)
#     joint_min: np.ndarray = None  # (7,)
#     joint_max: np.ndarray = None  # (7,)
#     R_diag: np.ndarray = None  # (7,)
#     max_nan_guard: int = 5

#
# def clamp(x, lo, hi):
#     return np.minimum(np.maximum(x, lo), hi)
#
#
# def rollout_value_policy(model, q0, goal_pos, active_joints,
#                          T=2.0, dt=0.02, R_diag=None,
#                          vel_limits=None, joint_min=None, joint_max=None,
#                          device="cpu"):
#     """
#     u* = -0.5 R^{-1} grad_q V, qdot=u
#     """
#     if R_diag is None:
#         R_diag = np.array([0.15] * 7, dtype=np.float64)
#     if vel_limits is None:
#         vel_limits = np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0], dtype=np.float64)
#
#     R_inv = 1.0 / np.maximum(R_diag, 1e-9)
#
#     N = int(round(T / dt)) + 1
#     q = np.array(q0, dtype=np.float64).copy()
#
#     traj = RobotTrajectory()
#     traj.joint_trajectory.joint_names = list(active_joints)
#
#     for k in range(N):
#         t = k * dt
#         # Torch (requires_grad for q)
#         qt = torch.tensor(q[None, :], dtype=torch.float32, device=device, requires_grad=True)
#         tt = torch.tensor([[t / T]], dtype=torch.float32, device=device, requires_grad=True)
#         gt = torch.tensor(goal_pos[None, :], dtype=torch.float32, device=device)
#
#         V = model(build_input(qt, tt, gt))  # (1,)
#         grad_q = torch.autograd.grad(V.sum(), qt, create_graph=False)[0].detach().cpu().numpy().reshape(7)
#
#         if not np.all(np.isfinite(grad_q)):
#             u = np.zeros(7, dtype=np.float64)
#         else:
#             u = -0.5 * R_inv * grad_q
#
#         u = clamp(u, -vel_limits, vel_limits)
#
#         pt = JointTrajectoryPoint()
#         pt.positions = q.tolist()
#         pt.velocities = u.tolist()
#         pt.time_from_start = rospy.Duration.from_sec(t)
#         traj.joint_trajectory.points.append(pt)
#
#         if k < N - 1:
#             q = q + dt * u
#             if joint_min is not None and joint_max is not None:
#                 q = clamp(q, joint_min, joint_max)
#
#     return traj
