"""Finite-segment 2D Capsule and TCP-radius collision primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..models import ArmEnvelopeResult, TcpRadiusResult


@dataclass(slots=True, frozen=True)
class ArmEnvelopeBatchResult:
    collision: npt.NDArray[np.bool_]
    centerline_distance_mm: npt.NDArray[np.float64]
    required_distance_mm: float
    safety_margin_mm: npt.NDArray[np.float64]
    capsule_surface_clearance_mm: npt.NDArray[np.float64]
    closest_a_xy_mm: npt.NDArray[np.float64]
    closest_b_xy_mm: npt.NDArray[np.float64]


def _point2(value: np.ndarray) -> tuple[float, float]:
    point = np.asarray(value, dtype=np.float64)
    return float(point[0]), float(point[1])


def _cross2d(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    return result


def _project_points_to_segments(
    points: npt.NDArray[np.float64],
    starts: npt.NDArray[np.float64],
    ends: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    direction = ends - starts
    denominator = np.einsum("ij,ij->i", direction, direction)
    numerator = np.einsum("ij,ij->i", points - starts, direction)
    parameter = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    result: npt.NDArray[np.float64] = starts + parameter[:, None] * direction
    return result


def check_arm_envelope_xy_batch(
    base_a_xy: np.ndarray,
    tcp_a_xy: np.ndarray,
    radius_a_mm: float,
    base_b_xy: np.ndarray,
    tcp_b_xy: np.ndarray,
    radius_b_mm: float,
    clearance_mm: float,
    epsilon_mm: float,
    touching_is_collision: bool,
) -> ArmEnvelopeBatchResult:
    """Measure and test two Base-to-TCP XY Capsules for every sample.

    For separated segments, the four endpoint-to-segment candidates are
    sufficient and give a stable tie order: A-base, A-TCP, B-base, B-TCP.
    Proper intersections use the analytical intersection point. Collinear
    overlaps retain the first zero-distance candidate.
    """
    base_a = np.asarray(base_a_xy, dtype=np.float64).reshape(2)
    base_b = np.asarray(base_b_xy, dtype=np.float64).reshape(2)
    tcp_a = np.asarray(tcp_a_xy, dtype=np.float64).reshape((-1, 2))
    tcp_b = np.asarray(tcp_b_xy, dtype=np.float64).reshape((-1, 2))
    if len(tcp_a) != len(tcp_b):
        raise ValueError("tcp_a_xy and tcp_b_xy must contain the same number of rows")
    count = len(tcp_a)
    required = float(radius_a_mm + radius_b_mm + clearance_mm)
    if count == 0:
        empty = np.empty(0, dtype=np.float64)
        empty_points = np.empty((0, 2), dtype=np.float64)
        return ArmEnvelopeBatchResult(
            np.empty(0, dtype=np.bool_),
            empty,
            required,
            empty.copy(),
            empty.copy(),
            empty_points,
            empty_points.copy(),
        )

    repeated_a = np.broadcast_to(base_a, (count, 2))
    repeated_b = np.broadcast_to(base_b, (count, 2))
    projection_a_base_on_b = _project_points_to_segments(repeated_a, repeated_b, tcp_b)
    projection_a_tcp_on_b = _project_points_to_segments(tcp_a, repeated_b, tcp_b)
    projection_b_base_on_a = _project_points_to_segments(repeated_b, repeated_a, tcp_a)
    projection_b_tcp_on_a = _project_points_to_segments(tcp_b, repeated_a, tcp_a)

    candidates_a = np.stack(
        (repeated_a, tcp_a, projection_b_base_on_a, projection_b_tcp_on_a), axis=1
    )
    candidates_b = np.stack(
        (projection_a_base_on_b, projection_a_tcp_on_b, repeated_b, tcp_b), axis=1
    )
    candidate_distance_sq = np.sum((candidates_a - candidates_b) ** 2, axis=2)
    selected = np.argmin(candidate_distance_sq, axis=1)
    rows = np.arange(count)
    closest_a = candidates_a[rows, selected].copy()
    closest_b = candidates_b[rows, selected].copy()
    distance = np.sqrt(candidate_distance_sq[rows, selected])

    direction_a = tcp_a - repeated_a
    direction_b = tcp_b - repeated_b
    offset = repeated_b - repeated_a
    denominator = _cross2d(direction_a, direction_b)
    numerical_tolerance = (
        np.finfo(np.float64).eps
        * np.maximum(np.linalg.norm(direction_a, axis=1) * np.linalg.norm(direction_b, axis=1), 1.0)
        * 16.0
    )
    non_parallel = np.abs(denominator) > numerical_tolerance
    parameter_a = np.divide(
        _cross2d(offset, direction_b),
        denominator,
        out=np.zeros(count, dtype=np.float64),
        where=non_parallel,
    )
    parameter_b = np.divide(
        _cross2d(offset, direction_a),
        denominator,
        out=np.zeros(count, dtype=np.float64),
        where=non_parallel,
    )
    intersects = (
        non_parallel
        & (parameter_a >= 0.0)
        & (parameter_a <= 1.0)
        & (parameter_b >= 0.0)
        & (parameter_b <= 1.0)
    )
    if np.any(intersects):
        point = repeated_a[intersects] + parameter_a[intersects, None] * direction_a[intersects]
        closest_a[intersects] = point
        closest_b[intersects] = point
        distance[intersects] = 0.0

    surface_clearance = distance - float(radius_a_mm + radius_b_mm)
    safety_margin = distance - required
    collision = (
        safety_margin <= epsilon_mm if touching_is_collision else safety_margin < -epsilon_mm
    )
    return ArmEnvelopeBatchResult(
        collision=collision,
        centerline_distance_mm=distance,
        required_distance_mm=required,
        safety_margin_mm=safety_margin,
        capsule_surface_clearance_mm=surface_clearance,
        closest_a_xy_mm=closest_a,
        closest_b_xy_mm=closest_b,
    )


def check_arm_envelope_xy(
    base_a_xy: np.ndarray,
    tcp_a_xy: np.ndarray,
    radius_a_mm: float,
    base_b_xy: np.ndarray,
    tcp_b_xy: np.ndarray,
    radius_b_mm: float,
    clearance_mm: float,
    epsilon_mm: float,
    touching_is_collision: bool,
) -> ArmEnvelopeResult:
    """Scalar form of :func:`check_arm_envelope_xy_batch`."""
    batch = check_arm_envelope_xy_batch(
        base_a_xy,
        np.asarray(tcp_a_xy, dtype=np.float64).reshape((1, 2)),
        radius_a_mm,
        base_b_xy,
        np.asarray(tcp_b_xy, dtype=np.float64).reshape((1, 2)),
        radius_b_mm,
        clearance_mm,
        epsilon_mm,
        touching_is_collision,
    )
    return ArmEnvelopeResult(
        collision=bool(batch.collision[0]),
        centerline_distance_mm=float(batch.centerline_distance_mm[0]),
        required_distance_mm=batch.required_distance_mm,
        safety_margin_mm=float(batch.safety_margin_mm[0]),
        capsule_surface_clearance_mm=float(batch.capsule_surface_clearance_mm[0]),
        closest_a_x_mm=float(batch.closest_a_xy_mm[0, 0]),
        closest_a_y_mm=float(batch.closest_a_xy_mm[0, 1]),
        closest_b_x_mm=float(batch.closest_b_xy_mm[0, 0]),
        closest_b_y_mm=float(batch.closest_b_xy_mm[0, 1]),
    )


def check_tcp_radius_xy(
    tcp_a_xy: np.ndarray,
    radius_a_mm: float,
    tcp_b_xy: np.ndarray,
    radius_b_mm: float,
    touching_is_collision: bool,
) -> TcpRadiusResult:
    """Check whether two XY TCP clearance circles overlap or touch."""
    tcp_a = _point2(tcp_a_xy)
    tcp_b = _point2(tcp_b_xy)
    distance = math.hypot(tcp_a[0] - tcp_b[0], tcp_a[1] - tcp_b[1])
    required = radius_a_mm + radius_b_mm
    collision = distance <= required if touching_is_collision else distance < required
    return TcpRadiusResult(collision, distance, required)
