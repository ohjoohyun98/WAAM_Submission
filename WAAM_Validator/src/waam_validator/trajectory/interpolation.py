"""Piecewise-linear trajectory interpolation."""

from __future__ import annotations

import numpy as np

from ..constants import MODE_W
from ..models import RobotTrajectory, TrajectorySet


def interpolate_robot_state(
    trajectory: RobotTrajectory,
    time_s: float,
) -> tuple[np.ndarray, int]:
    """Return interpolated float32 XYZ and the left-row interval mode."""
    if time_s < 0:
        raise ValueError("time_s must be non-negative")
    if time_s >= trajectory.time_s[-1]:
        return trajectory.xyz_mm[-1].copy(), int(MODE_W)
    index = int(np.searchsorted(trajectory.time_s, time_s, side="right") - 1)
    index = max(index, 0)
    start = trajectory.time_s[index]
    end = trajectory.time_s[index + 1]
    alpha = (time_s - start) / (end - start)
    xyz = trajectory.xyz_mm[index].astype(np.float64) + alpha * (
        trajectory.xyz_mm[index + 1].astype(np.float64)
        - trajectory.xyz_mm[index].astype(np.float64)
    )
    return xyz.astype(np.float32), int(trajectory.mode[index])


def interpolate_all_states(
    trajectories: TrajectorySet,
    time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.empty((3, 3), dtype=np.float32)
    modes = np.empty(3, dtype=np.uint8)
    for index, trajectory in enumerate(trajectories.robots):
        position, mode = interpolate_robot_state(trajectory, time_s)
        xyz[index] = position
        modes[index] = mode
    return xyz, modes
