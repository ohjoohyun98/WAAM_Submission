"""Shared compact runtime and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry

from ._version import __version__
from .errors import ValidationIssue
from .provenance import RESULT_SCHEMA_VERSION, InputSignature


@dataclass(slots=True)
class RobotTrajectory:
    robot_id: int
    time_s: npt.NDArray[np.float64]
    xyz_mm: npt.NDArray[np.float32]
    mode: npt.NDArray[np.uint8]


@dataclass(slots=True)
class TrajectorySet:
    robots: tuple[RobotTrajectory, RobotTrajectory, RobotTrajectory]
    row_count: int

    def by_id(self, robot_id: int) -> RobotTrajectory:
        return self.robots[robot_id - 1]


@dataclass(slots=True)
class SimulationSample:
    time_s: float
    xyz_by_robot: npt.NDArray[np.float32]
    mode_by_robot: npt.NDArray[np.uint8]


@dataclass(slots=True, frozen=True)
class TcpRadiusResult:
    collision: bool
    distance_mm: float
    required_distance_mm: float


@dataclass(slots=True, frozen=True)
class ArmEnvelopeResult:
    collision: bool
    centerline_distance_mm: float
    required_distance_mm: float
    safety_margin_mm: float
    capsule_surface_clearance_mm: float
    closest_a_x_mm: float
    closest_a_y_mm: float
    closest_b_x_mm: float
    closest_b_y_mm: float


@dataclass(slots=True)
class CollisionEvent:
    event_id: int
    collision_type: str
    robot_a: int
    robot_b: int
    start_s: float
    end_s: float
    duration_s: float
    minimum_distance_mm: float
    required_distance_mm: float
    minimum_safety_margin_mm: float
    minimum_capsule_surface_clearance_mm: float | None
    minimum_distance_time_s: float
    closest_a_x_mm: float
    closest_a_y_mm: float
    closest_b_x_mm: float
    closest_b_y_mm: float


@dataclass(slots=True)
class CollisionSimulationResult:
    events: list[CollisionEvent]
    minimum_arm_safety_margin_mm: float
    arm_centerline_distance_at_worst_mm: float
    arm_required_distance_at_worst_mm: float
    arm_capsule_surface_clearance_at_worst_mm: float
    minimum_arm_pair: tuple[int, int]
    minimum_arm_time_s: float
    minimum_arm_closest_a_xy: tuple[float, float]
    minimum_arm_closest_b_xy: tuple[float, float]
    minimum_arm_tcp_positions_xy: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    minimum_tcp_distance_mm: float
    minimum_tcp_pair: tuple[int, int]
    minimum_tcp_required_distance_mm: float
    minimum_tcp_time_s: float
    sample_count: int

    @property
    def arm_envelope_event_count(self) -> int:
        return sum(event.collision_type == "ARM_ENVELOPE" for event in self.events)

    @property
    def tcp_radius_event_count(self) -> int:
        return sum(event.collision_type == "TCP_RADIUS" for event in self.events)


@dataclass(slots=True)
class RobotMetrics:
    robot_id: int
    completion_s: float
    deposition_time_s: float
    travel_time_s: float
    wait_time_s: float
    deposition_length_mm: float
    travel_length_mm: float
    mean_deposition_speed_mm_s: float | None
    mean_travel_speed_mm_s: float | None


@dataclass(slots=True)
class ScheduleMetrics:
    makespan_s: float
    robots: list[RobotMetrics]
    workload_imbalance_s: float
    normalized_imbalance: float


@dataclass(slots=True)
class RobotReachMetrics:
    robot_id: int
    xy_reach_radius_mm: float
    maximum_xy_distance_mm: float
    minimum_xy_margin_mm: float
    xy_utilization_ratio: float
    xy_violation_point_count: int
    first_xy_violation_s: float | None
    last_xy_violation_s: float | None

    @property
    def passed(self) -> bool:
        return self.xy_violation_point_count == 0


@dataclass(slots=True)
class ReachMetrics:
    robots: list[RobotReachMetrics]

    @property
    def passed(self) -> bool:
        return all(robot.passed for robot in self.robots)


@dataclass(slots=True)
class LayerGeometry:
    layer_index: int
    z_bottom_mm: float
    z_top_mm: float
    z_slice_mm: float
    deposited_polygon: BaseGeometry
    target_polygon: BaseGeometry


@dataclass(slots=True)
class LayerMetrics:
    layer_index: int
    z_bottom_mm: float
    z_top_mm: float
    z_slice_mm: float
    target_area_mm2: float
    deposited_area_mm2: float
    intersection_area_mm2: float
    underfill_area_mm2: float
    overfill_area_mm2: float
    coverage: float
    underfill_ratio: float
    overfill_ratio: float | None
    iou: float
    passed: bool


@dataclass(slots=True)
class ShapeMetrics:
    target_volume_mm3: float
    deposited_volume_mm3: float
    intersection_volume_mm3: float
    underfill_volume_mm3: float
    overfill_volume_mm3: float
    coverage: float
    underfill_ratio: float
    overfill_ratio: float
    iou: float
    failed_layer_count: int
    evaluated_layer_count: int
    failed_layer_ratio: float
    target_mesh_volume_mm3: float
    target_volume_discrepancy_ratio: float
    passed: bool


@dataclass(slots=True)
class ValidationResult:
    status: str
    input_dir: Path
    output_dir: Path
    trajectory_rows: int
    target_watertight: bool
    input_signature: InputSignature
    schedule: ScheduleMetrics
    reach: ReachMetrics
    collision: CollisionSimulationResult
    shape: ShapeMetrics
    layer_metrics: list[LayerMetrics]
    warnings: list[ValidationIssue] = field(default_factory=list)
    violations: list[ValidationIssue] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    checks_enabled: dict[str, bool] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def collision_free(self) -> bool:
        return not self.collision.events

    def summary_dict(self) -> dict[str, Any]:
        completions = {str(item.robot_id): item.completion_s for item in self.schedule.robots}
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "validator_version": __version__,
            "status": self.status,
            "input": {
                "directory": str(self.input_dir),
                "trajectory_rows": self.trajectory_rows,
                "target_watertight": self.target_watertight,
            },
            "schedule": {
                "makespan_s": self.schedule.makespan_s,
                "robot_completion_s": completions,
                "workload_imbalance_s": self.schedule.workload_imbalance_s,
                "normalized_imbalance": self.schedule.normalized_imbalance,
            },
            "reach": {
                "distance_basis": "XY",
                "passed": self.reach.passed,
                "robots": [
                    {
                        "robot_id": item.robot_id,
                        "passed": item.passed,
                        "xy_reach_radius_mm": item.xy_reach_radius_mm,
                        "maximum_xy_distance_mm": item.maximum_xy_distance_mm,
                        "minimum_xy_margin_mm": item.minimum_xy_margin_mm,
                        "xy_utilization_ratio": item.xy_utilization_ratio,
                        "xy_violation_point_count": item.xy_violation_point_count,
                        "first_xy_violation_s": item.first_xy_violation_s,
                        "last_xy_violation_s": item.last_xy_violation_s,
                    }
                    for item in self.reach.robots
                ],
            },
            "collision": {
                "passed": self.collision_free,
                "collision_event_count": len(self.collision.events),
                "arm_envelope": {
                    "enabled": self.checks_enabled.get("arm_envelope", False),
                    "passed": self.collision.arm_envelope_event_count == 0,
                    "event_count": self.collision.arm_envelope_event_count,
                    "minimum_safety_margin_mm": (self.collision.minimum_arm_safety_margin_mm),
                    "centerline_distance_at_worst_mm": (
                        self.collision.arm_centerline_distance_at_worst_mm
                    ),
                    "required_distance_at_worst_mm": (
                        self.collision.arm_required_distance_at_worst_mm
                    ),
                    "capsule_surface_clearance_at_worst_mm": (
                        self.collision.arm_capsule_surface_clearance_at_worst_mm
                    ),
                    "pair": list(self.collision.minimum_arm_pair),
                    "time_s": self.collision.minimum_arm_time_s,
                    "closest_points_xy_mm": [
                        list(self.collision.minimum_arm_closest_a_xy),
                        list(self.collision.minimum_arm_closest_b_xy),
                    ],
                    "tcp_positions_xy_mm": [
                        list(point) for point in self.collision.minimum_arm_tcp_positions_xy
                    ],
                },
                "tcp_radius": {
                    "enabled": self.checks_enabled.get("tcp_radius", False),
                    "passed": self.collision.tcp_radius_event_count == 0,
                    "event_count": self.collision.tcp_radius_event_count,
                    "minimum_distance_mm": self.collision.minimum_tcp_distance_mm,
                    "required_distance_at_minimum_mm": (
                        self.collision.minimum_tcp_required_distance_mm
                    ),
                    "pair": list(self.collision.minimum_tcp_pair),
                    "time_s": self.collision.minimum_tcp_time_s,
                },
            },
            "shape": {
                "passed": self.shape.passed,
                "target_volume_mm3": self.shape.target_volume_mm3,
                "deposited_volume_mm3": self.shape.deposited_volume_mm3,
                "intersection_volume_mm3": self.shape.intersection_volume_mm3,
                "underfill_volume_mm3": self.shape.underfill_volume_mm3,
                "overfill_volume_mm3": self.shape.overfill_volume_mm3,
                "coverage": self.shape.coverage,
                "underfill_ratio": self.shape.underfill_ratio,
                "overfill_ratio": self.shape.overfill_ratio,
                "iou": self.shape.iou,
                "failed_layer_count": self.shape.failed_layer_count,
                "evaluated_layer_count": self.shape.evaluated_layer_count,
                "failed_layer_ratio": self.shape.failed_layer_ratio,
                "target_mesh_volume_mm3": self.shape.target_mesh_volume_mm3,
                "target_volume_discrepancy_ratio": (self.shape.target_volume_discrepancy_ratio),
            },
            "failure_reasons": self.failure_reasons,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "robot_id": issue.robot_id,
                    "start_s": issue.start_s,
                    "end_s": issue.end_s,
                }
                for issue in [*self.violations, *self.warnings]
            ],
            "output_directory": str(self.output_dir),
        }
