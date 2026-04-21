"""
costs.py

Cost terms for DGM / HJB formulation.

These terms should be usable both:
- during training (loss composition)
- during rollout evaluation (trajectory scoring)
"""

from typing import Dict, Any

def terminal_pose_error_cost(*args, **kwargs) -> float:
    raise NotImplementedError

def control_effort_cost(*args, **kwargs) -> float:
    raise NotImplementedError

def clearance_cost(*args, **kwargs) -> float:
    raise NotImplementedError

def time_cost(*args, **kwargs) -> float:
    raise NotImplementedError

def total_cost(components: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in components.items())
