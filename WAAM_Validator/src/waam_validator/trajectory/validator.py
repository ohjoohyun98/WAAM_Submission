"""Semantic validation for trajectory intervals."""

from __future__ import annotations

import numpy as np

from ..config.models import Config
from ..constants import EXTREME_COORDINATE_WARNING_MM, MODE_D, MODE_T, MODE_W
from ..errors import InputValidationError, ValidationMessages
from ..models import TrajectorySet
from ..shape.layer_index import determine_layer_index
from .reach import compute_reach_metrics


def validate_trajectory_set(
    trajectories: TrajectorySet,
    config: Config,
) -> ValidationMessages:
    """Validate W movement, D layers, and configured reference speeds."""
    messages = ValidationMessages()
    geometry_epsilon = config.collision.geometry_epsilon_mm
    process = config.process
    validation = config.validation

    reach = compute_reach_metrics(trajectories, config)
    for item in reach.robots:
        if not item.passed:
            messages.violation(
                "ROBOT_XY_REACH_VIOLATION",
                f"Robot {item.robot_id} maximum TCP XY distance "
                f"{item.maximum_xy_distance_mm:.6g} mm exceeds configured XY Reach "
                f"radius {item.xy_reach_radius_mm:.6g} mm at "
                f"{item.xy_violation_point_count} point(s).",
                robot_id=item.robot_id,
                start_s=item.first_xy_violation_s,
                end_s=item.last_xy_violation_s,
            )

    for trajectory in trajectories.robots:
        xyz_mm = trajectory.xyz_mm.astype(np.float64)
        deltas = np.diff(xyz_mm, axis=0)
        durations = np.diff(trajectory.time_s)
        lengths = np.linalg.norm(deltas, axis=1)
        xy_lengths = np.linalg.norm(deltas[:, :2], axis=1)
        interval_modes = trajectory.mode[:-1]
        is_deposition = interval_modes == MODE_D
        is_travel = interval_modes == MODE_T
        is_wait = interval_modes == MODE_W
        workspace_center = np.asarray(config.workspace.center_xy_mm, dtype=np.float64)
        workspace_offsets = xyz_mm[:, :2] - workspace_center
        workspace_radii = np.hypot(workspace_offsets[:, 0], workspace_offsets[:, 1])

        if np.max(np.abs(xyz_mm)) > EXTREME_COORDINATE_WARNING_MM:
            messages.warning(
                "EXTREME_COORDINATE_WARNING",
                f"Robot {trajectory.robot_id} contains coordinates above "
                f"{EXTREME_COORDINATE_WARNING_MM:g} mm.",
                robot_id=trajectory.robot_id,
            )

        representative_z = (xyz_mm[:-1, 2] + xyz_mm[1:, 2]) / 2.0
        if process.tcp_z_reference == "top":
            raw_layer = (representative_z - process.build_plane_z_mm) / process.layer_height_mm
            rounded_layer = np.rint(raw_layer)
            layer_below_zero = rounded_layer < 1.0
            expected_z = process.build_plane_z_mm + rounded_layer * process.layer_height_mm
        else:
            raw_layer = (
                representative_z - process.build_plane_z_mm - process.layer_height_mm / 2.0
            ) / process.layer_height_mm
            rounded_layer = np.rint(raw_layer)
            layer_below_zero = rounded_layer < 0.0
            expected_z = process.build_plane_z_mm + (rounded_layer + 0.5) * process.layer_height_mm

        deposited_outer_radius = (
            np.maximum(workspace_radii[:-1], workspace_radii[1:]) + process.bead_width_mm / 2.0
        )
        outside_workspace = is_deposition & (
            deposited_outer_radius > config.workspace.radius_mm + geometry_epsilon
        )
        zero_length = is_deposition & (xy_lengths <= geometry_epsilon)
        vertical_move = is_deposition & (np.abs(deltas[:, 2]) > validation.layer_z_tolerance_mm)
        below_build_plane = is_deposition & (
            representative_z < process.build_plane_z_mm - validation.layer_z_tolerance_mm
        )
        layer_mismatch = is_deposition & (
            layer_below_zero
            | (np.abs(representative_z - expected_z) > validation.layer_z_tolerance_mm)
        )
        fatal_indices = np.flatnonzero(
            outside_workspace | zero_length | vertical_move | below_build_plane | layer_mismatch
        )
        if len(fatal_indices):
            index = int(fatal_indices[0])
            start_s = float(trajectory.time_s[index])
            if outside_workspace[index]:
                raise InputValidationError(
                    "DEPOSITION_OUTSIDE_WORKSPACE",
                    f"Robot {trajectory.robot_id} D interval at {start_s:.6g} s "
                    f"extends to radius {deposited_outer_radius[index]:.6g} mm outside "
                    f"the {config.workspace.radius_mm:.6g} mm workspace.",
                )
            if zero_length[index]:
                raise InputValidationError(
                    "DEPOSITION_ZERO_LENGTH",
                    f"Robot {trajectory.robot_id} has a zero-length D interval at {start_s:.6g} s.",
                )
            if vertical_move[index]:
                raise InputValidationError(
                    "DEPOSITION_VERTICAL_MOVE",
                    f"Robot {trajectory.robot_id} D interval at {start_s:.6g} s "
                    "moves vertically beyond layer tolerance.",
                )
            if below_build_plane[index]:
                raise InputValidationError(
                    "DEPOSITION_BELOW_BUILD_PLANE",
                    f"Robot {trajectory.robot_id} deposits below the build plane.",
                )
            try:
                determine_layer_index(float(representative_z[index]), config)
            except ValueError as exc:
                raise InputValidationError("DEPOSITION_LAYER_MISMATCH", str(exc)) from exc
            raise AssertionError("Fatal deposition mask did not map to an input error")

        speeds = lengths / durations
        wait_position_changed = is_wait & (lengths > validation.wait_position_tolerance_mm)
        deposition_speed_changed = is_deposition & (
            np.abs(speeds - process.deposition_speed_mm_s) / process.deposition_speed_mm_s
            > validation.speed_relative_tolerance
        )
        stationary_travel = is_travel & (lengths <= geometry_epsilon)
        maximum_travel_speed = process.travel_speed_mm_s * (
            1.0 + validation.speed_relative_tolerance
        )
        travel_speed_exceeded = is_travel & (speeds > maximum_travel_speed)
        issue_indices = np.flatnonzero(
            wait_position_changed
            | deposition_speed_changed
            | stationary_travel
            | travel_speed_exceeded
        )
        for index_raw in issue_indices:
            index = int(index_raw)
            start_s = float(trajectory.time_s[index])
            context = {
                "robot_id": trajectory.robot_id,
                "start_s": start_s,
                "end_s": float(trajectory.time_s[index + 1]),
            }
            if wait_position_changed[index]:
                messages.violation(
                    "WAIT_POSITION_CHANGED",
                    f"Robot {trajectory.robot_id} moved {lengths[index]:.6g} mm during W.",
                    **context,
                )
            elif deposition_speed_changed[index]:
                _speed_issue(
                    messages,
                    validation.fail_on_speed_violation,
                    "DEPOSITION_SPEED_VIOLATION",
                    f"Robot {trajectory.robot_id} deposition speed {speeds[index]:.6g} mm/s "
                    f"differs from reference {process.deposition_speed_mm_s:.6g} mm/s.",
                    context,
                )
            else:
                if stationary_travel[index]:
                    messages.warning(
                        "STATIONARY_TRAVEL_WARNING",
                        f"Robot {trajectory.robot_id} has stationary T interval at "
                        f"{start_s:.6g} s.",
                        **context,
                    )
                if travel_speed_exceeded[index]:
                    _speed_issue(
                        messages,
                        validation.fail_on_speed_violation,
                        "TRAVEL_SPEED_VIOLATION",
                        f"Robot {trajectory.robot_id} travel speed "
                        f"{speeds[index]:.6g} mm/s exceeds tolerated maximum "
                        f"{maximum_travel_speed:.6g} mm/s.",
                        context,
                    )
    return messages


def _speed_issue(
    messages: ValidationMessages,
    fail: bool,
    code: str,
    message: str,
    context: dict[str, int | float],
) -> None:
    if fail:
        messages.violation(code, message, **context)
    else:
        messages.warning(code, message, **context)
