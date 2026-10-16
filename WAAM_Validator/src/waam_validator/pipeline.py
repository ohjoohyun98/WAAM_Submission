"""End-to-end WAAM validation orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from ._version import __version__
from .collision.simulator import run_collision_analysis
from .config.loader import load_config
from .config.models import Config
from .errors import (
    ComputationError,
    InputValidationError,
    ValidationMessages,
    WaamValidatorError,
)
from .models import CollisionSimulationResult, ReachMetrics, ShapeMetrics, ValidationResult
from .progress import ProgressCallback, ValidationProgress
from .provenance import input_signature
from .reporting.writers import (
    close_file_logging,
    configure_file_logging,
    prepare_output_directory,
    write_error_json,
    write_result_files,
)
from .schedule.metrics import compute_schedule_metrics
from .shape.deposition import build_deposited_layers
from .shape.metrics import compute_shape_metrics
from .shape.target import (
    determine_evaluation_layers,
    load_target_mesh,
    slice_target_layers,
    validate_coordinate_consistency,
)
from .trajectory.loader import load_trajectory_csv
from .trajectory.reach import compute_reach_metrics
from .trajectory.validator import validate_trajectory_set

LOGGER = logging.getLogger("waam_validator")

_PROGRESS_RANGES = {
    "loading_inputs": (0.00, 0.05),
    "collision": (0.05, 0.45),
    "deposition": (0.45, 0.60),
    "target_slicing": (0.60, 0.85),
    "shape_metrics": (0.85, 0.95),
    "results": (0.95, 1.00),
    "completed": (1.00, 1.00),
}


def _report_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    stage_progress: float = 0.0,
    completed_units: float | None = None,
    total_units: float | None = None,
    unit: str | None = None,
) -> None:
    if callback is not None:
        start, end = _PROGRESS_RANGES[stage]
        bounded = min(1.0, max(0.0, stage_progress))
        callback(
            ValidationProgress(
                stage=stage,
                message=message,
                stage_progress=bounded,
                overall_progress=start + (end - start) * bounded,
                completed_units=completed_units,
                total_units=total_units,
                unit=unit,
            )
        )


def _guard_progress_callback(callback: ProgressCallback | None) -> ProgressCallback | None:
    """Disable a broken observer after one warning without changing validation."""
    if callback is None:
        return None
    failed = False

    def guarded(event: ValidationProgress) -> None:
        nonlocal failed
        if failed:
            return
        try:
            callback(event)
        except Exception:
            failed = True
            LOGGER.warning(
                "Progress callback failed; remaining progress events are disabled",
                exc_info=True,
            )

    return guarded


def _resolve_input(input_dir: Path) -> tuple[Path, Path, Path, Path]:
    resolved = input_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise InputValidationError("MISSING_CONFIG", f"Input directory does not exist: {resolved}")
    config_path = resolved / "config.yaml"
    trajectory_path = resolved / "trajectory.csv"
    target_path = resolved / "target.stl"
    if not config_path.is_file():
        raise InputValidationError("MISSING_CONFIG", "config.yaml is required.")
    if not trajectory_path.is_file():
        raise InputValidationError("MISSING_TRAJECTORY", "trajectory.csv is required.")
    if not target_path.is_file():
        raise InputValidationError("MISSING_TARGET", "target.stl is required.")
    return resolved, config_path, trajectory_path, target_path


def _build_failure_reasons(
    shape: ShapeMetrics,
    reach: ReachMetrics,
    messages: ValidationMessages,
    collision: CollisionSimulationResult,
    config: Config,
) -> list[str]:
    reasons: list[str] = []
    speed_codes = {"DEPOSITION_SPEED_VIOLATION", "TRAVEL_SPEED_VIOLATION"}
    for issue in messages.violations:
        if issue.code not in speed_codes | {"ROBOT_XY_REACH_VIOLATION"}:
            reasons.append(f"Input/process validation failed: {issue.code} - {issue.message}")
    for item in reach.robots:
        if not item.passed:
            reasons.append(
                f"ROBOT_XY_REACH: R{item.robot_id} maximum TCP XY distance "
                f"{item.maximum_xy_distance_mm:.3f} mm exceeds configured "
                f"XY Reach radius {item.xy_reach_radius_mm:.3f} mm."
            )
    if collision.arm_envelope_event_count:
        reasons.append(
            f"ARM_ENVELOPE: {collision.arm_envelope_event_count} event(s) detected; "
            f"worst safety margin {collision.minimum_arm_safety_margin_mm:.3f} mm."
        )
    if collision.tcp_radius_event_count:
        reasons.append(f"TCP_RADIUS: {collision.tcp_radius_event_count} event(s) detected.")
    thresholds = config.shape_validation
    if shape.coverage < thresholds.minimum_overall_coverage:
        reasons.append(
            f"Overall coverage {shape.coverage:.2%} is below minimum "
            f"{thresholds.minimum_overall_coverage:.2%}."
        )
    if shape.overfill_ratio > thresholds.maximum_overall_overfill_ratio:
        reasons.append(
            f"Overall overfill {shape.overfill_ratio:.2%} exceeds maximum "
            f"{thresholds.maximum_overall_overfill_ratio:.2%}."
        )
    if shape.iou < thresholds.minimum_overall_iou:
        reasons.append(
            f"Overall IoU {shape.iou:.2%} is below minimum {thresholds.minimum_overall_iou:.2%}."
        )
    if shape.failed_layer_ratio > thresholds.maximum_failed_layer_ratio:
        reasons.append(
            f"Failed-layer ratio {shape.failed_layer_ratio:.2%} exceeds maximum "
            f"{thresholds.maximum_failed_layer_ratio:.2%}."
        )
    speed_violations = sum(issue.code in speed_codes for issue in messages.violations)
    if speed_violations:
        reasons.append(f"Speed validation failed: {speed_violations} violating interval(s).")
    return list(dict.fromkeys(reasons))


def run_validation(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ValidationResult:
    """Execute validation and write the fixed core result bundle."""
    resolved_input = input_dir.expanduser().resolve()
    if not resolved_input.is_dir():
        raise InputValidationError(
            "MISSING_CONFIG", f"Input directory does not exist: {resolved_input}"
        )
    run_output = prepare_output_directory(resolved_input, output_dir)
    progress_callback = _guard_progress_callback(progress_callback)
    try:
        log_handler = configure_file_logging(run_output)
    except WaamValidatorError as exc:
        exc.output_dir = run_output
        write_error_json(run_output, exc, resolved_input)
        raise
    try:
        LOGGER.info("WAAM Validator version %s", __version__)
        _report_progress(progress_callback, "loading_inputs", "입력 파일을 읽고 있습니다.")
        resolved_input, config_path, trajectory_path, target_path = _resolve_input(resolved_input)
        starting_signature = input_signature(resolved_input)
        LOGGER.info("Loading configuration")
        config = load_config(config_path)
        LOGGER.info("Loading trajectory")
        trajectories = load_trajectory_csv(trajectory_path, config)
        messages = validate_trajectory_set(trajectories, config)
        LOGGER.info("Loaded %d trajectory rows", trajectories.row_count)

        target_mesh = load_target_mesh(target_path, config, messages)
        validate_coordinate_consistency(trajectories, target_mesh, config)
        LOGGER.info(
            "Loaded target: %d faces, watertight=%s",
            len(target_mesh.faces),
            target_mesh.is_watertight,
        )
        _report_progress(
            progress_callback,
            "loading_inputs",
            "입력 파일과 좌표계를 확인했습니다.",
            1.0,
        )

        schedule = compute_schedule_metrics(trajectories)
        reach = compute_reach_metrics(trajectories, config)
        LOGGER.info(
            "Schedule: makespan=%.3f s, workload_imbalance=%.3f s (%.3f%%)",
            schedule.makespan_s,
            schedule.workload_imbalance_s,
            schedule.normalized_imbalance * 100.0,
        )
        for schedule_item, reach_item in zip(schedule.robots, reach.robots, strict=True):
            LOGGER.info(
                "R%d metrics: D=%.3f s/%.3f mm/%.6f mm/s, "
                "T=%.3f s/%.3f mm/%.6f mm/s, W=%.3f s, "
                "xy_reach=%.3f/%.3f mm (%.3f%%), violations=%d",
                schedule_item.robot_id,
                schedule_item.deposition_time_s,
                schedule_item.deposition_length_mm,
                schedule_item.mean_deposition_speed_mm_s or 0.0,
                schedule_item.travel_time_s,
                schedule_item.travel_length_mm,
                schedule_item.mean_travel_speed_mm_s or 0.0,
                schedule_item.wait_time_s,
                reach_item.maximum_xy_distance_mm,
                reach_item.xy_reach_radius_mm,
                reach_item.xy_utilization_ratio * 100.0,
                reach_item.xy_violation_point_count,
            )
        _report_progress(progress_callback, "collision", "충돌 시뮬레이션을 실행 중입니다.")
        collision = run_collision_analysis(
            trajectories,
            config,
            progress_callback=lambda fraction, completed, total: _report_progress(
                progress_callback,
                "collision",
                "충돌 시뮬레이션을 실행 중입니다.",
                fraction,
                completed,
                total,
                "simulation_s",
            ),
        )
        LOGGER.info(
            "Collision scan: %d samples, %d events",
            collision.sample_count,
            len(collision.events),
        )
        LOGGER.info(
            "Arm Envelope worst case: margin=%.6f mm, centerline=%.6f mm, "
            "required=%.6f mm, pair=R%d-R%d, time=%.6f s",
            collision.minimum_arm_safety_margin_mm,
            collision.arm_centerline_distance_at_worst_mm,
            collision.arm_required_distance_at_worst_mm,
            collision.minimum_arm_pair[0],
            collision.minimum_arm_pair[1],
            collision.minimum_arm_time_s,
        )

        if not config.collision.check_arm_envelope:
            messages.warning(
                "ARM_ENVELOPE_CHECK_DISABLED",
                "2D Base-to-TCP Capsule collision check is disabled; minimum safety "
                "margin is still reported as a metric.",
            )
        if not config.collision.check_tcp_radius:
            messages.warning("TCP_RADIUS_CHECK_DISABLED", "TCP radius check is disabled.")

        _report_progress(progress_callback, "deposition", "적층 Layer 형상을 생성 중입니다.")
        deposited_layers = build_deposited_layers(
            trajectories,
            config,
            progress_callback=lambda fraction, completed, total: _report_progress(
                progress_callback,
                "deposition",
                "적층 Layer 형상을 생성 중입니다.",
                fraction,
                completed,
                total,
                "intervals",
            ),
        )
        LOGGER.info("Built deposited geometry for %d layers", len(deposited_layers))
        layer_indices = determine_evaluation_layers(target_mesh, deposited_layers, config)
        if layer_indices:
            LOGGER.info(
                "Evaluating %d layers (indices %d to %d)",
                len(layer_indices),
                layer_indices[0],
                layer_indices[-1],
            )
        else:
            LOGGER.info("No layers selected for shape evaluation")
        _report_progress(progress_callback, "target_slicing", "Target STL을 slicing 중입니다.")
        target_layers = slice_target_layers(
            target_mesh,
            layer_indices,
            config,
            messages,
            progress_callback=lambda fraction, completed, total: _report_progress(
                progress_callback,
                "target_slicing",
                "Target STL을 slicing 중입니다.",
                fraction,
                completed,
                total,
                "layers",
            ),
        )
        _report_progress(progress_callback, "shape_metrics", "Layer 형상 지표를 계산 중입니다.")
        shape, layer_metrics = compute_shape_metrics(
            deposited_layers,
            target_layers,
            config,
            target_mesh_volume_mm3=float(abs(target_mesh.volume)),
            progress_callback=lambda fraction, completed, total: _report_progress(
                progress_callback,
                "shape_metrics",
                "Layer 형상 지표를 계산 중입니다.",
                fraction,
                completed,
                total,
                "layers",
            ),
        )
        LOGGER.info(
            "Shape metrics: coverage=%.6f, underfill=%.6f, overfill=%.6f, "
            "IoU=%.6f, failed_layers=%d/%d, layer_volume=%.3f mm3, "
            "mesh_volume=%.3f mm3, discrepancy=%.6f",
            shape.coverage,
            shape.underfill_ratio,
            shape.overfill_ratio,
            shape.iou,
            shape.failed_layer_count,
            shape.evaluated_layer_count,
            shape.target_volume_mm3,
            shape.target_mesh_volume_mm3,
            shape.target_volume_discrepancy_ratio,
        )
        if (
            shape.target_volume_discrepancy_ratio
            > config.validation.target_volume_discrepancy_warning_ratio
        ):
            messages.warning(
                "TARGET_VOLUME_DISCREPANCY_WARNING",
                f"Target mesh and layer-integrated volumes differ by "
                f"{shape.target_volume_discrepancy_ratio:.2%}.",
            )

        failure_reasons = _build_failure_reasons(
            shape,
            reach,
            messages,
            collision,
            config,
        )
        status = "PASS" if not failure_reasons else "FAIL"
        LOGGER.info(
            "Validation findings: %d failure reasons, %d violations, %d warnings",
            len(failure_reasons),
            len(messages.violations),
            len(messages.warnings),
        )
        for reason in failure_reasons:
            LOGGER.info("Failure reason: %s", reason)
        if input_signature(resolved_input) != starting_signature:
            raise InputValidationError(
                "INPUT_CHANGED_DURING_VALIDATION",
                "Validation 실행 중 입력 파일이 변경되었습니다. 변경이 끝난 뒤 다시 실행하세요.",
            )
        result = ValidationResult(
            status=status,
            input_dir=resolved_input,
            output_dir=run_output,
            trajectory_rows=trajectories.row_count,
            target_watertight=bool(target_mesh.is_watertight),
            input_signature=starting_signature,
            schedule=schedule,
            reach=reach,
            collision=collision,
            shape=shape,
            layer_metrics=layer_metrics,
            warnings=list(dict.fromkeys(messages.warnings)),
            violations=list(dict.fromkeys(messages.violations)),
            failure_reasons=failure_reasons,
            checks_enabled={
                "arm_envelope": config.collision.check_arm_envelope,
                "tcp_radius": config.collision.check_tcp_radius,
            },
            config_snapshot=config.model_dump(mode="json"),
        )

        _report_progress(progress_callback, "results", "결과 파일을 기록 중입니다.")
        write_result_files(result)
        LOGGER.info("Validation completed with status %s", status)
        _report_progress(
            progress_callback,
            "completed",
            f"Validation이 {status}로 완료됐습니다.",
            1.0,
        )
        return result
    except WaamValidatorError as exc:
        exc.output_dir = run_output
        LOGGER.error("%s - %s", exc.code, exc.message)
        write_error_json(run_output, exc, resolved_input)
        raise
    except Exception as exc:
        LOGGER.exception("Unexpected calculation error")
        wrapped = ComputationError("INTERNAL_CALCULATION_ERROR", str(exc), output_dir=run_output)
        write_error_json(run_output, wrapped, resolved_input)
        raise wrapped from exc
    finally:
        close_file_logging(log_handler)


def check_input(input_dir: Path) -> tuple[int, bool, int]:
    """Validate required inputs without collision, shape slicing, or output creation."""
    _, config_path, trajectory_path, target_path = _resolve_input(input_dir)
    config = load_config(config_path)
    trajectories = load_trajectory_csv(trajectory_path, config)
    # Semantic violations returned as messages (XY reach, wait movement, and
    # configured speed failures) are valid, runnable inputs whose full result is
    # FAIL. Fatal interval/layer problems are raised directly by the validator.
    validate_trajectory_set(trajectories, config)
    mesh = load_target_mesh(target_path, config)
    validate_coordinate_consistency(trajectories, mesh, config)
    return trajectories.row_count, bool(mesh.is_watertight), len(mesh.faces)
