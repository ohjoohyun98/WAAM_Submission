"""Exact schedule metrics from original trajectory intervals."""

from __future__ import annotations

import math

import numpy as np

from ..constants import MODE_D, MODE_T, MODE_W
from ..models import RobotMetrics, ScheduleMetrics, TrajectorySet


def compute_schedule_metrics(trajectories: TrajectorySet) -> ScheduleMetrics:
    """Compute exact metrics from original timestamps, never sampled times."""
    robot_metrics: list[RobotMetrics] = []
    for trajectory in trajectories.robots:
        durations = np.diff(trajectory.time_s)
        distances = np.linalg.norm(np.diff(trajectory.xyz_mm.astype(np.float64), axis=0), axis=1)
        interval_modes = trajectory.mode[:-1]
        completion = float(trajectory.time_s[-1])

        deposition_time = float(durations[interval_modes == MODE_D].sum())
        travel_time = float(durations[interval_modes == MODE_T].sum())
        wait_time = float(durations[interval_modes == MODE_W].sum())
        if not math.isclose(
            deposition_time + travel_time + wait_time,
            completion,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ArithmeticError("Mode durations do not sum to completion time")
        deposition_length = float(distances[interval_modes == MODE_D].sum())
        travel_length = float(distances[interval_modes == MODE_T].sum())
        robot_metrics.append(
            RobotMetrics(
                robot_id=trajectory.robot_id,
                completion_s=completion,
                deposition_time_s=deposition_time,
                travel_time_s=travel_time,
                wait_time_s=wait_time,
                deposition_length_mm=deposition_length,
                travel_length_mm=travel_length,
                mean_deposition_speed_mm_s=(
                    deposition_length / deposition_time if deposition_time > 0 else None
                ),
                mean_travel_speed_mm_s=(travel_length / travel_time if travel_time > 0 else None),
            )
        )
    completions = [item.completion_s for item in robot_metrics]
    makespan = max(completions)
    imbalance = max(completions) - min(completions)
    return ScheduleMetrics(
        makespan_s=makespan,
        robots=robot_metrics,
        workload_imbalance_s=imbalance,
        normalized_imbalance=imbalance / makespan,
    )
