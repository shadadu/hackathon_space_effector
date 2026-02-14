"""
moveit_clients.py

Thin wrappers around MoveIt services so the DGM planner service node
doesn't directly own ROS plumbing logic.

Typical usage:
- compute_ik(pose_stamped, group_name, ee_link, seed_state)
- get_planning_scene()  (optional)
"""

from typing import Optional, Dict, Any, List

def compute_ik(*args, **kwargs):
    """
    Compute IK using moveit_msgs/GetPositionIK.
    Return: (success: bool, solution_state, error_code)
    """
    raise NotImplementedError("Implement IK client wrapper here.")


def get_planning_scene(*args, **kwargs):
    """
    Fetch planning scene snapshot using moveit_msgs/GetPlanningScene.
    Return: moveit_msgs/PlanningScene
    """
    raise NotImplementedError("Implement planning scene fetch here.")


def get_active_joints(*args, **kwargs) -> List[str]:
    """
    Utility to get active joint names for a MoveIt group.
    """
    raise NotImplementedError("Implement via RobotCommander/MoveGroupCommander.")
