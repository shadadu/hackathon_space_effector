import numpy as np
from dataclasses import dataclass
@dataclass
class RolloutConfig:
    T: float
    dt: float
    vel_limits: np.ndarray # | None = None
    joint_min: np.ndarray # | None = None
    joint_max: np.ndarray #| None = None
    R_diag: np.ndarray #| None = None
    max_nan_guard: int #= 5