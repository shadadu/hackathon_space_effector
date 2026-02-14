"""
metrics.py

Benchmark metrics for OMPL vs DGM comparison.

Examples:
- planning_time
- trajectory duration
- path length in joint space
- smoothness proxy (sum of squared accelerations)
- terminal pose error (requires FK)
- min clearance (requires scene distance queries)
"""

from typing import Dict, Any

def compute_basic_metrics(*args, **kwargs) -> Dict[str, Any]:
    """
    Return a dict of comparable metrics between planners.
    Keep it lightweight initially; add advanced metrics later.
    """
    return {}
