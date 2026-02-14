"""
rollout.py

Rollout a control policy (from DGM) to generate a trajectory.

This is the critical bridge:
- Inputs: start state, goal, constraints, object prediction
- Outputs: time-indexed joint targets (positions, optional velocities)

Later, your dgm_planner_node will call into here.
"""

from typing import Any, Dict, List, Sequence, Tuple

def rollout_policy(*args, **kwargs):
    """
    Return (positions_over_time, times, diagnostics).
    """
    raise NotImplementedError
