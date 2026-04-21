"""
interface.py

Defines a minimal internal interface used by both:
- OMPL baseline caller (benchmark_runner)
- DGM planner service (dgm_planner_node)

We keep ROS message types at the boundary (GetMotionPlan),
but inside the codebase we convert to/from a simple Python dataclass
to keep logic testable and modular.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class PlanRequestLite:
    group_name: str
    ee_link: str
    world_frame: str

    # Start state and goal are kept as opaque objects so
    # this module doesn't import heavy ROS types.
    start_state: Any
    goal_constraints: Any

    allowed_planning_time: float = 2.0
    num_planning_attempts: int = 5
    vel_scale: float = 0.3
    acc_scale: float = 0.3

    # Optional object dynamics for microgravity interception
    object_state: Optional[Any] = None
    execution_latency_s: float = 0.0

    # Optional cost weights for optimal control
    cost_weights: Optional[Dict[str, float]] = None


@dataclass
class PlanResponseLite:
    success: bool
    error_code: int
    planning_time: float
    trajectory: Any  # moveit_msgs/RobotTrajectory

    # Optional diagnostics
    diagnostics: Optional[Dict[str, Any]] = None
