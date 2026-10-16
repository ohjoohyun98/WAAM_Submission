"""Streaming 2D capsule and TCP-radius collision simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.models import Config
from ..constants import ROBOT_PAIRS
from ..models import (
    ArmEnvelopeResult,
    CollisionEvent,
    CollisionSimulationResult,
    TcpRadiusResult,
    TrajectorySet,
)
from ..progress import StageProgressCallback
from ..trajectory.sampling import iter_simulation_samples
from .geometry2d import check_arm_envelope_xy_batch


@dataclass(slots=True)
class _ActiveEvent:
    collision_type: str
    robot_a: int
    robot_b: int
    start_s: float
    last_true_s: float
    gap_start_s: float | None = None
    minimum_distance_mm: float = float("inf")
    required_distance_mm: float = 0.0
    minimum_safety_margin_mm: float = float("inf")
    minimum_capsule_surface_clearance_mm: float | None = None
    minimum_distance_time_s: float = 0.0
    closest_a_x_mm: float = 0.0
    closest_a_y_mm: float = 0.0
    closest_b_x_mm: float = 0.0
    closest_b_y_mm: float = 0.0


class CollisionEventAccumulator:
    """Merge sampled collision states into deterministic interval events."""

    def __init__(self, merge_gap_s: float) -> None:
        self.merge_gap_s = merge_gap_s
        self._active: dict[tuple[str, int, int], _ActiveEvent] = {}
        self._finished: list[CollisionEvent] = []

    def update(
        self,
        time_s: float,
        pair: tuple[int, int],
        arm_result: ArmEnvelopeResult,
        tcp_result: TcpRadiusResult,
        tcp_positions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Scalar convenience API used by small callers and unit tests."""
        arm_a = np.array([[arm_result.closest_a_x_mm, arm_result.closest_a_y_mm]])
        arm_b = np.array([[arm_result.closest_b_x_mm, arm_result.closest_b_y_mm]])
        self.update_batch(
            np.array([time_s], dtype=np.float64),
            pair,
            np.array([arm_result.collision]),
            np.array([arm_result.centerline_distance_mm]),
            arm_result.required_distance_mm,
            np.array([arm_result.safety_margin_mm]),
            np.array([arm_result.capsule_surface_clearance_mm]),
            arm_a,
            arm_b,
            np.array([tcp_result.collision]),
            np.array([tcp_result.distance_mm]),
            tcp_result.required_distance_mm,
            np.array([tcp_result.distance_mm - tcp_result.required_distance_mm]),
            np.asarray([tcp_positions[0]], dtype=np.float64),
            np.asarray([tcp_positions[1]], dtype=np.float64),
        )

    def update_batch(
        self,
        time_s: np.ndarray,
        pair: tuple[int, int],
        arm_collision: np.ndarray,
        arm_distance_mm: np.ndarray,
        required_arm_distance_mm: float,
        arm_safety_margin_mm: np.ndarray,
        arm_surface_clearance_mm: np.ndarray,
        arm_closest_a_xy: np.ndarray,
        arm_closest_b_xy: np.ndarray,
        tcp_collision: np.ndarray,
        tcp_distance_mm: np.ndarray,
        required_tcp_distance_mm: float,
        tcp_safety_margin_mm: np.ndarray,
        tcp_a_xy: np.ndarray,
        tcp_b_xy: np.ndarray,
    ) -> None:
        """Consume collision arrays while visiting state runs, not every sample."""
        self._update_batch_one(
            time_s,
            pair,
            "ARM_ENVELOPE",
            arm_collision,
            arm_distance_mm,
            required_arm_distance_mm,
            arm_safety_margin_mm,
            arm_closest_a_xy,
            arm_closest_b_xy,
            arm_surface_clearance_mm,
        )
        self._update_batch_one(
            time_s,
            pair,
            "TCP_RADIUS",
            tcp_collision,
            tcp_distance_mm,
            required_tcp_distance_mm,
            tcp_safety_margin_mm,
            tcp_a_xy,
            tcp_b_xy,
            None,
        )

    def _update_batch_one(
        self,
        time_s: np.ndarray,
        pair: tuple[int, int],
        collision_type: str,
        collided: np.ndarray,
        distance_mm: np.ndarray,
        required_distance_mm: float,
        safety_margin_mm: np.ndarray,
        closest_a_xy: np.ndarray,
        closest_b_xy: np.ndarray,
        surface_clearance_mm: np.ndarray | None,
    ) -> None:
        if len(time_s) == 0:
            return
        key = (collision_type, pair[0], pair[1])
        changes = np.flatnonzero(collided[1:] != collided[:-1]) + 1
        starts = np.concatenate((np.array([0]), changes))
        stops = np.concatenate((changes, np.array([len(collided)])))
        for start_raw, stop_raw in zip(starts, stops, strict=True):
            start = int(start_raw)
            stop = int(stop_raw)
            active = self._active.get(key)
            if not bool(collided[start]):
                if active is not None and active.gap_start_s is None:
                    active.gap_start_s = float(time_s[start])
                continue

            run_start_s = float(time_s[start])
            if active is not None and active.gap_start_s is not None:
                if run_start_s - active.gap_start_s > self.merge_gap_s:
                    self._finish(key)
                    active = None
                else:
                    active.gap_start_s = None
            if active is None:
                active = _ActiveEvent(collision_type, pair[0], pair[1], run_start_s, run_start_s)
                self._active[key] = active
            active.last_true_s = float(time_s[stop - 1])
            active.gap_start_s = None

            relative_minimum = int(np.argmin(safety_margin_mm[start:stop]))
            minimum_index = start + relative_minimum
            minimum_margin = float(safety_margin_mm[minimum_index])
            if minimum_margin < active.minimum_safety_margin_mm:
                active.minimum_safety_margin_mm = minimum_margin
                active.minimum_distance_mm = float(distance_mm[minimum_index])
                active.required_distance_mm = required_distance_mm
                active.minimum_distance_time_s = float(time_s[minimum_index])
                active.closest_a_x_mm = float(closest_a_xy[minimum_index, 0])
                active.closest_a_y_mm = float(closest_a_xy[minimum_index, 1])
                active.closest_b_x_mm = float(closest_b_xy[minimum_index, 0])
                active.closest_b_y_mm = float(closest_b_xy[minimum_index, 1])
                if surface_clearance_mm is not None:
                    active.minimum_capsule_surface_clearance_mm = float(
                        surface_clearance_mm[minimum_index]
                    )

    def _finish(self, key: tuple[str, int, int], final_time: float | None = None) -> None:
        active = self._active.pop(key)
        end_s = active.last_true_s if final_time is None else final_time
        self._finished.append(
            CollisionEvent(
                event_id=0,
                collision_type=active.collision_type,
                robot_a=active.robot_a,
                robot_b=active.robot_b,
                start_s=active.start_s,
                end_s=end_s,
                duration_s=end_s - active.start_s,
                minimum_distance_mm=active.minimum_distance_mm,
                required_distance_mm=active.required_distance_mm,
                minimum_safety_margin_mm=active.minimum_safety_margin_mm,
                minimum_capsule_surface_clearance_mm=(active.minimum_capsule_surface_clearance_mm),
                minimum_distance_time_s=active.minimum_distance_time_s,
                closest_a_x_mm=active.closest_a_x_mm,
                closest_a_y_mm=active.closest_a_y_mm,
                closest_b_x_mm=active.closest_b_x_mm,
                closest_b_y_mm=active.closest_b_y_mm,
            )
        )

    def finalize(self, makespan_s: float) -> list[CollisionEvent]:
        for key, active in list(self._active.items()):
            final_time = makespan_s if active.last_true_s == makespan_s else None
            self._finish(key, final_time)
        self._finished.sort(
            key=lambda item: (item.start_s, item.collision_type, item.robot_a, item.robot_b)
        )
        for event_id, event in enumerate(self._finished, start=1):
            event.event_id = event_id
        return self._finished


def run_collision_analysis(
    trajectories: TrajectorySet,
    config: Config,
    *,
    progress_callback: StageProgressCallback | None = None,
) -> CollisionSimulationResult:
    """Run sampled 2D collision checks while always tracking both minima."""
    bases: dict[int, np.ndarray] = {
        robot.id: np.asarray(robot.base_xyz_mm[:2], dtype=np.float64) for robot in config.robots
    }
    tcp_radii: dict[int, float] = {robot.id: robot.tcp_radius_mm for robot in config.robots}
    arm_radii: dict[int, float] = {
        robot.id: robot.arm_envelope_radius_mm for robot in config.robots
    }
    accumulator = CollisionEventAccumulator(config.simulation.event_merge_gap_s)

    minimum_arm_margin = float("inf")
    minimum_arm_distance = float("inf")
    minimum_arm_required = arm_radii[1] + arm_radii[2] + config.collision.arm_clearance_mm
    minimum_arm_surface = float("inf")
    minimum_arm_pair = ROBOT_PAIRS[0]
    minimum_arm_time = 0.0
    minimum_arm_closest_a = (0.0, 0.0)
    minimum_arm_closest_b = (0.0, 0.0)
    minimum_arm_tcp_positions = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    minimum_tcp_distance = float("inf")
    minimum_tcp_pair = ROBOT_PAIRS[0]
    minimum_tcp_required = tcp_radii[1] + tcp_radii[2]
    minimum_tcp_time = 0.0

    sample_count = 0
    makespan = max(float(item.time_s[-1]) for item in trajectories.robots)
    batch_size = config.simulation.batch_size
    time_buffer = np.empty(batch_size, dtype=np.float64)
    xy_buffer = np.empty((batch_size, 3, 2), dtype=np.float64)
    buffered = 0

    def process_batch(count: int) -> None:
        nonlocal minimum_arm_closest_a, minimum_arm_closest_b
        nonlocal minimum_arm_distance, minimum_arm_margin, minimum_arm_pair
        nonlocal minimum_arm_required, minimum_arm_surface, minimum_arm_tcp_positions
        nonlocal minimum_arm_time, minimum_tcp_distance, minimum_tcp_pair
        nonlocal minimum_tcp_required, minimum_tcp_time
        times = time_buffer[:count]
        positions = xy_buffer[:count]
        for robot_a, robot_b in ROBOT_PAIRS:
            tcp_a = positions[:, robot_a - 1]
            tcp_b = positions[:, robot_b - 1]
            tcp_distances = np.hypot(tcp_a[:, 0] - tcp_b[:, 0], tcp_a[:, 1] - tcp_b[:, 1])
            required_tcp = tcp_radii[robot_a] + tcp_radii[robot_b]
            tcp_safety_margin = tcp_distances - required_tcp
            tcp_minimum_index = int(np.argmin(tcp_distances))
            local_tcp_minimum = float(tcp_distances[tcp_minimum_index])
            if local_tcp_minimum < minimum_tcp_distance:
                minimum_tcp_distance = local_tcp_minimum
                minimum_tcp_pair = (robot_a, robot_b)
                minimum_tcp_required = required_tcp
                minimum_tcp_time = float(times[tcp_minimum_index])

            arm = check_arm_envelope_xy_batch(
                bases[robot_a],
                tcp_a,
                arm_radii[robot_a],
                bases[robot_b],
                tcp_b,
                arm_radii[robot_b],
                config.collision.arm_clearance_mm,
                config.collision.geometry_epsilon_mm,
                config.collision.touching_is_collision,
            )
            arm_minimum_index = int(np.argmin(arm.safety_margin_mm))
            local_arm_margin = float(arm.safety_margin_mm[arm_minimum_index])
            if local_arm_margin < minimum_arm_margin:
                minimum_arm_margin = local_arm_margin
                minimum_arm_distance = float(arm.centerline_distance_mm[arm_minimum_index])
                minimum_arm_required = arm.required_distance_mm
                minimum_arm_surface = float(arm.capsule_surface_clearance_mm[arm_minimum_index])
                minimum_arm_pair = (robot_a, robot_b)
                minimum_arm_time = float(times[arm_minimum_index])
                point_a = arm.closest_a_xy_mm[arm_minimum_index]
                point_b = arm.closest_b_xy_mm[arm_minimum_index]
                minimum_arm_closest_a = (
                    float(point_a[0]),
                    float(point_a[1]),
                )
                minimum_arm_closest_b = (
                    float(point_b[0]),
                    float(point_b[1]),
                )
                worst_positions = positions[arm_minimum_index]
                minimum_arm_tcp_positions = (
                    (float(worst_positions[0, 0]), float(worst_positions[0, 1])),
                    (float(worst_positions[1, 0]), float(worst_positions[1, 1])),
                    (float(worst_positions[2, 0]), float(worst_positions[2, 1])),
                )

            arm_collision = (
                arm.collision
                if config.collision.check_arm_envelope
                else np.zeros(count, dtype=np.bool_)
            )
            if config.collision.check_tcp_radius:
                if config.collision.touching_is_collision:
                    tcp_collision = tcp_distances <= required_tcp
                else:
                    tcp_collision = tcp_distances < required_tcp
            else:
                tcp_collision = np.zeros(count, dtype=np.bool_)
            accumulator.update_batch(
                times,
                (robot_a, robot_b),
                arm_collision,
                arm.centerline_distance_mm,
                arm.required_distance_mm,
                arm.safety_margin_mm,
                arm.capsule_surface_clearance_mm,
                arm.closest_a_xy_mm,
                arm.closest_b_xy_mm,
                tcp_collision,
                tcp_distances,
                required_tcp,
                tcp_safety_margin,
                tcp_a,
                tcp_b,
            )

    last_reported = -1.0
    for sample in iter_simulation_samples(trajectories, config):
        time_buffer[buffered] = sample.time_s
        xy_buffer[buffered] = sample.xyz_by_robot[:, :2]
        buffered += 1
        sample_count += 1
        if buffered == batch_size:
            process_batch(buffered)
            fraction = min(1.0, float(time_buffer[buffered - 1]) / makespan) if makespan else 1.0
            if progress_callback is not None and fraction - last_reported >= 0.001:
                progress_callback(fraction, float(time_buffer[buffered - 1]), makespan)
                last_reported = fraction
            buffered = 0
    if buffered:
        process_batch(buffered)
    if progress_callback is not None and last_reported < 1.0:
        progress_callback(1.0, makespan, makespan)
    return CollisionSimulationResult(
        events=accumulator.finalize(makespan),
        minimum_arm_safety_margin_mm=minimum_arm_margin,
        arm_centerline_distance_at_worst_mm=minimum_arm_distance,
        arm_required_distance_at_worst_mm=minimum_arm_required,
        arm_capsule_surface_clearance_at_worst_mm=minimum_arm_surface,
        minimum_arm_pair=minimum_arm_pair,
        minimum_arm_time_s=minimum_arm_time,
        minimum_arm_closest_a_xy=minimum_arm_closest_a,
        minimum_arm_closest_b_xy=minimum_arm_closest_b,
        minimum_arm_tcp_positions_xy=minimum_arm_tcp_positions,
        minimum_tcp_distance_mm=minimum_tcp_distance,
        minimum_tcp_pair=minimum_tcp_pair,
        minimum_tcp_required_distance_mm=minimum_tcp_required,
        minimum_tcp_time_s=minimum_tcp_time,
        sample_count=sample_count,
    )


def run_collision_simulation(
    trajectories: TrajectorySet,
    config: Config,
    *,
    progress_callback: StageProgressCallback | None = None,
) -> list[CollisionEvent]:
    """Run configured sampled collision checks and return merged events."""
    return run_collision_analysis(trajectories, config, progress_callback=progress_callback).events
