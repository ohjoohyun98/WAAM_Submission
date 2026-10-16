"""Robot TCP reach metrics for the World XY reach proxy."""

from __future__ import annotations

import numpy as np

from ..config.models import Config
from ..models import ReachMetrics, RobotReachMetrics, TrajectorySet


def compute_reach_metrics(trajectories: TrajectorySet, config: Config) -> ReachMetrics:
    """Measure original TCP points against each robot's configured XY reach circle."""
    robots: list[RobotReachMetrics] = []
    epsilon = config.collision.geometry_epsilon_mm
    for trajectory in trajectories.robots:
        robot = config.robot(trajectory.robot_id)
        base_xy = np.asarray(robot.base_xyz_mm[:2], dtype=np.float64)
        tcp_xy = trajectory.xyz_mm[:, :2].astype(np.float64)
        distances = np.linalg.norm(tcp_xy - base_xy[None, :], axis=1)
        violating = distances > robot.xy_reach_radius_mm + epsilon
        indices = np.flatnonzero(violating)
        maximum = float(distances.max())
        robots.append(
            RobotReachMetrics(
                robot_id=robot.id,
                xy_reach_radius_mm=robot.xy_reach_radius_mm,
                maximum_xy_distance_mm=maximum,
                minimum_xy_margin_mm=robot.xy_reach_radius_mm - maximum,
                xy_utilization_ratio=maximum / robot.xy_reach_radius_mm,
                xy_violation_point_count=int(len(indices)),
                first_xy_violation_s=(
                    float(trajectory.time_s[int(indices[0])]) if len(indices) else None
                ),
                last_xy_violation_s=(
                    float(trajectory.time_s[int(indices[-1])]) if len(indices) else None
                ),
            )
        )
    return ReachMetrics(robots=robots)
