"""Public API for the WAAM validation simulator."""

from ._version import __version__
from .collision.simulator import run_collision_simulation
from .config.loader import load_config
from .pipeline import run_validation
from .progress import ValidationProgress
from .schedule.metrics import compute_schedule_metrics
from .shape.deposition import build_deposited_layers
from .shape.metrics import compute_shape_metrics
from .shape.target import slice_target_layers
from .trajectory.interpolation import interpolate_robot_state
from .trajectory.loader import load_trajectory_csv
from .trajectory.reach import compute_reach_metrics
from .trajectory.sampling import iter_simulation_samples
from .trajectory.validator import validate_trajectory_set

__all__ = [
    "build_deposited_layers",
    "compute_shape_metrics",
    "compute_schedule_metrics",
    "compute_reach_metrics",
    "interpolate_robot_state",
    "iter_simulation_samples",
    "load_config",
    "load_trajectory_csv",
    "run_validation",
    "run_collision_simulation",
    "slice_target_layers",
    "validate_trajectory_set",
    "ValidationProgress",
    "__version__",
]
