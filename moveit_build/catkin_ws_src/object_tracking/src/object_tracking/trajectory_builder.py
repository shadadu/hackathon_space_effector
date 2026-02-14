"""
trajectory_builder.py

Utilities to build moveit_msgs/RobotTrajectory / trajectory_msgs/JointTrajectory
from arrays of joint positions and timestamps.

This is the main output format both OMPL and DGM paths must produce.
"""

from typing import List, Sequence, Optional

def joint_interpolation(q0: Sequence[float], q1: Sequence[float], n: int, duration: float):
    """
    Placeholder trajectory generator. Replace with DGM rollout later.
    Returns list of (positions, time_from_start_sec).
    """
    if n < 2:
        n = 2
    pts = []
    for i in range(n):
        a = i / (n - 1)
        q = [(1 - a) * q0j + a * q1j for q0j, q1j in zip(q0, q1)]
        pts.append((q, a * duration))
    return pts


def build_robot_trajectory(joint_names: List[str],
                          positions_over_time: List[Sequence[float]],
                          times: List[float]):
    """
    Create and return moveit_msgs/RobotTrajectory from joint positions + times.
    """
    raise NotImplementedError("Implement using trajectory_msgs/JointTrajectoryPoint.")
