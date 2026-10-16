"""Adaptive global timeline generation."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from ..config.models import Config
from ..models import SimulationSample, TrajectorySet
from .interpolation import interpolate_all_states


def iter_simulation_samples(
    trajectories: TrajectorySet,
    config: Config,
) -> Iterator[SimulationSample]:
    """Yield adaptive samples without retaining the global timeline."""
    breakpoints = np.unique(np.concatenate([item.time_s for item in trajectories.robots]))
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        left_f = float(left)
        right_f = float(right)
        xyz_left, _ = interpolate_all_states(trajectories, left_f)
        xyz_right, _ = interpolate_all_states(trajectories, right_f)
        duration = right_f - left_f
        subdivisions_time = math.ceil(duration / config.simulation.max_time_step_s)
        displacements = np.linalg.norm(
            xyz_right.astype(np.float64) - xyz_left.astype(np.float64), axis=1
        )
        subdivisions_position = int(
            np.max(np.ceil(displacements / config.simulation.max_tcp_step_mm))
        )
        subdivisions = max(1, subdivisions_time, subdivisions_position)
        for offset in range(subdivisions):
            sample_time = left_f + duration * offset / subdivisions
            xyz, modes = interpolate_all_states(trajectories, sample_time)
            yield SimulationSample(sample_time, xyz, modes)
    final_time = float(breakpoints[-1])
    xyz, modes = interpolate_all_states(trajectories, final_time)
    yield SimulationSample(final_time, xyz, modes)
