"""Stable JSON/CSV/Markdown and console output."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._version import __version__
from ..constants import LIMITATIONS_TEXT
from ..errors import OutputWriteError, WaamValidatorError
from ..models import ValidationResult
from ..provenance import RESULT_SCHEMA_VERSION, write_validation_input_manifest

CORE_RESULT_FILES = (
    "summary.json",
    "validation_report.md",
    "robot_metrics.csv",
    "collision_events.csv",
    "layer_metrics.csv",
    "run.log",
    "validation_inputs.json",
)
_RUN_HANDLER_NAME = "waam-validator-run-file"


@dataclass(slots=True)
class RunLogContext:
    """Logger state that must be restored after a library API call."""

    handler: logging.Handler
    previous_level: int
    previous_propagate: bool


def prepare_output_directory(input_dir: Path, output_dir: Path | None) -> Path:
    """Create a non-destructive run output directory."""
    try:
        if output_dir is not None:
            resolved = output_dir.expanduser().resolve()
            if resolved.exists() and any(resolved.iterdir()):
                raise OutputWriteError(
                    "OUTPUT_WRITE_FAILED",
                    f"Explicit output directory is not empty: {resolved}",
                )
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        root = input_dir / "output"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        candidate = root / stamp
        suffix = 1
        while candidate.exists():
            candidate = root / f"{stamp}_{suffix:03d}"
            suffix += 1
        candidate.mkdir()
        return candidate.resolve()
    except OutputWriteError:
        raise
    except OSError as exc:
        raise OutputWriteError("OUTPUT_WRITE_FAILED", str(exc)) from exc


def configure_file_logging(output_dir: Path) -> RunLogContext:
    """Configure the package logger to write a run-local file."""
    logger = logging.getLogger("waam_validator")
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        if handler.get_name() == _RUN_HANDLER_NAME:
            handler.close()
            logger.removeHandler(handler)
    try:
        handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    except OSError as exc:
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        raise OutputWriteError("OUTPUT_WRITE_FAILED", str(exc)) from exc
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    handler.set_name(_RUN_HANDLER_NAME)
    logger.addHandler(handler)
    logger.propagate = False
    return RunLogContext(handler, previous_level, previous_propagate)


def close_file_logging(context: RunLogContext) -> None:
    """Detach and close one run-local handler without touching consumer handlers."""
    logger = logging.getLogger("waam_validator")
    logger.removeHandler(context.handler)
    context.handler.flush()
    context.handler.close()
    logger.setLevel(context.previous_level)
    logger.propagate = context.previous_propagate


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise OutputWriteError("OUTPUT_WRITE_FAILED", str(exc)) from exc


def _optional(value: float | None) -> str | float:
    return "" if value is None else value


def write_result_files(
    result: ValidationResult,
) -> None:
    """Write a complete result bundle and publish ``summary.json`` last."""
    output = result.output_dir
    bundle = Path(tempfile.mkdtemp(prefix=".result-bundle-", dir=output))
    try:
        write_validation_input_manifest(
            result.input_dir,
            bundle,
            result.input_signature,
            result.config_snapshot,
        )
        (bundle / "summary.json").write_text(
            json.dumps(
                result.summary_dict(), ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        _write_csv(
            bundle / "collision_events.csv",
            [
                "event_id",
                "type",
                "robot_a",
                "robot_b",
                "start_s",
                "end_s",
                "duration_s",
                "minimum_distance_mm",
                "required_distance_mm",
                "minimum_safety_margin_mm",
                "minimum_capsule_surface_clearance_mm",
                "minimum_distance_time_s",
                "closest_a_x_mm",
                "closest_a_y_mm",
                "closest_b_x_mm",
                "closest_b_y_mm",
            ],
            [
                {
                    "event_id": event.event_id,
                    "type": event.collision_type,
                    "robot_a": event.robot_a,
                    "robot_b": event.robot_b,
                    "start_s": event.start_s,
                    "end_s": event.end_s,
                    "duration_s": event.duration_s,
                    "minimum_distance_mm": event.minimum_distance_mm,
                    "required_distance_mm": event.required_distance_mm,
                    "minimum_safety_margin_mm": event.minimum_safety_margin_mm,
                    "minimum_capsule_surface_clearance_mm": _optional(
                        event.minimum_capsule_surface_clearance_mm
                    ),
                    "minimum_distance_time_s": event.minimum_distance_time_s,
                    "closest_a_x_mm": event.closest_a_x_mm,
                    "closest_a_y_mm": event.closest_a_y_mm,
                    "closest_b_x_mm": event.closest_b_x_mm,
                    "closest_b_y_mm": event.closest_b_y_mm,
                }
                for event in result.collision.events
            ],
        )
        _write_csv(
            bundle / "robot_metrics.csv",
            [
                "robot_id",
                "completion_s",
                "deposition_time_s",
                "travel_time_s",
                "wait_time_s",
                "inactive_after_completion_s",
                "deposition_length_mm",
                "travel_length_mm",
                "mean_deposition_speed_mm_s",
                "mean_travel_speed_mm_s",
                "xy_reach_radius_mm",
                "maximum_xy_distance_mm",
                "minimum_xy_margin_mm",
                "xy_utilization_ratio",
                "xy_violation_point_count",
                "first_xy_violation_s",
                "last_xy_violation_s",
            ],
            [
                {
                    "robot_id": item.robot_id,
                    "completion_s": item.completion_s,
                    "deposition_time_s": item.deposition_time_s,
                    "travel_time_s": item.travel_time_s,
                    "wait_time_s": item.wait_time_s,
                    "inactive_after_completion_s": max(
                        0.0, result.schedule.makespan_s - item.completion_s
                    ),
                    "deposition_length_mm": item.deposition_length_mm,
                    "travel_length_mm": item.travel_length_mm,
                    "mean_deposition_speed_mm_s": _optional(item.mean_deposition_speed_mm_s),
                    "mean_travel_speed_mm_s": _optional(item.mean_travel_speed_mm_s),
                    "xy_reach_radius_mm": reach.xy_reach_radius_mm,
                    "maximum_xy_distance_mm": reach.maximum_xy_distance_mm,
                    "minimum_xy_margin_mm": reach.minimum_xy_margin_mm,
                    "xy_utilization_ratio": reach.xy_utilization_ratio,
                    "xy_violation_point_count": reach.xy_violation_point_count,
                    "first_xy_violation_s": _optional(reach.first_xy_violation_s),
                    "last_xy_violation_s": _optional(reach.last_xy_violation_s),
                }
                for item, reach in zip(result.schedule.robots, result.reach.robots, strict=True)
            ],
        )
        _write_csv(
            bundle / "layer_metrics.csv",
            [
                "layer_index",
                "z_bottom_mm",
                "z_top_mm",
                "z_slice_mm",
                "target_area_mm2",
                "deposited_area_mm2",
                "intersection_area_mm2",
                "underfill_area_mm2",
                "overfill_area_mm2",
                "coverage",
                "underfill_ratio",
                "overfill_ratio",
                "iou",
                "passed",
            ],
            [
                {
                    "layer_index": item.layer_index,
                    "z_bottom_mm": item.z_bottom_mm,
                    "z_top_mm": item.z_top_mm,
                    "z_slice_mm": item.z_slice_mm,
                    "target_area_mm2": item.target_area_mm2,
                    "deposited_area_mm2": item.deposited_area_mm2,
                    "intersection_area_mm2": item.intersection_area_mm2,
                    "underfill_area_mm2": item.underfill_area_mm2,
                    "overfill_area_mm2": item.overfill_area_mm2,
                    "coverage": item.coverage,
                    "underfill_ratio": item.underfill_ratio,
                    "overfill_ratio": _optional(item.overfill_ratio),
                    "iou": item.iou,
                    "passed": str(item.passed).lower(),
                }
                for item in result.layer_metrics
            ],
        )
        (bundle / "validation_report.md").write_text(
            _render_markdown(result), encoding="utf-8"
        )
        # A summary is the completed-run marker. Publishing it last prevents a
        # partially written bundle from being mistaken for a valid PASS/FAIL.
        for filename in (
            "validation_inputs.json",
            "collision_events.csv",
            "robot_metrics.csv",
            "layer_metrics.csv",
            "validation_report.md",
            "summary.json",
        ):
            os.replace(bundle / filename, output / filename)
    except OutputWriteError:
        raise
    except OSError as exc:
        raise OutputWriteError("OUTPUT_WRITE_FAILED", str(exc)) from exc
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def _render_markdown(result: ValidationResult) -> str:
    robot_lines = "\n".join(
        f"- R{item.robot_id}: completion {item.completion_s:.2f} s; "
        f"Deposition {item.deposition_time_s:.2f} s; "
        f"Travel {item.travel_time_s:.2f} s; Wait {item.wait_time_s:.2f} s"
        for item in result.schedule.robots
    )
    reach_lines = "\n".join(
        f"- R{item.robot_id}: max XY distance {item.maximum_xy_distance_mm:.2f} / "
        f"{item.xy_reach_radius_mm:.2f} mm; XY margin {item.minimum_xy_margin_mm:.2f} mm; "
        f"violating points {item.xy_violation_point_count}"
        for item in result.reach.robots
    )
    failures = (
        "\n".join(f"{index}. {reason}" for index, reason in enumerate(result.failure_reasons, 1))
        or "None"
    )
    warnings = "\n".join(f"- {item.display()}" for item in result.warnings) or "None"
    violations = "\n".join(f"- {item.display()}" for item in result.violations) or "None"
    outputs = "\n".join(f"- `{name}`" for name in CORE_RESULT_FILES)
    return f"""# WAAM Validator Report

## Overall Result

**{result.status}**

- WAAM Validator version: {__version__}

## Input Summary

- Directory: `{result.input_dir}`
- Trajectory rows: {result.trajectory_rows}
- Target watertight: {result.target_watertight}

## Schedule

- Makespan: {result.schedule.makespan_s:.2f} s
{robot_lines}

## Robot XY Reach

- Passed: {result.reach.passed}
{reach_lines}

## Collision

- Arm Envelope events: {result.collision.arm_envelope_event_count}
- Minimum Arm Envelope safety margin: {result.collision.minimum_arm_safety_margin_mm:.2f} mm
- Arm centerline distance at worst case:
  {result.collision.arm_centerline_distance_at_worst_mm:.2f} mm
- Arm required centerline distance at worst case:
  {result.collision.arm_required_distance_at_worst_mm:.2f} mm
- TCP_RADIUS events: {result.collision.tcp_radius_event_count}
- Minimum TCP distance: {result.collision.minimum_tcp_distance_mm:.2f} mm

## Shape

- Target layer-integrated volume: {result.shape.target_volume_mm3:.3f} mm³
- Deposited volume: {result.shape.deposited_volume_mm3:.3f} mm³
- Intersection volume: {result.shape.intersection_volume_mm3:.3f} mm³
- Underfill volume: {result.shape.underfill_volume_mm3:.3f} mm³
- Overfill volume: {result.shape.overfill_volume_mm3:.3f} mm³
- Coverage: {result.shape.coverage:.2%}
- Underfill: {result.shape.underfill_ratio:.2%}
- Overfill: {result.shape.overfill_ratio:.2%}
- IoU: {result.shape.iou:.2%}
- Failed layers: {result.shape.failed_layer_count} / {result.shape.evaluated_layer_count}

## Failure Reasons

{failures}

## Warnings

{warnings}

## Validation Violations

These findings contribute to a normal `FAIL`; they do not mean that the pipeline ended with
the fatal status `ERROR`.

{violations}

## Output Files

{outputs}

## Interpretation Limitations

{LIMITATIONS_TEXT}
"""


def write_error_json(output_dir: Path, error: WaamValidatorError, input_dir: Path) -> None:
    """Best-effort minimal fatal error artifact."""
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "validator_version": __version__,
        "status": "ERROR",
        "code": error.code,
        "message": error.message,
        "input_directory": str(input_dir),
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "error.json"
        temporary = output_dir / ".error.json.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        # ERROR is the only terminal marker after a fatal failure. If an older
        # summary was created before the exception, keep it from misleading
        # non-UI consumers as well.
        (output_dir / "summary.json").unlink(missing_ok=True)
    except OSError:
        return


def render_console_summary(result: ValidationResult) -> str:
    """Render the fixed human-oriented console result block."""
    lines = [
        "=" * 60,
        "WAAM PATH VALIDATION RESULT",
        "=" * 60,
        f"Overall Result : {result.status}",
        "",
        "[Schedule]",
        f"Makespan : {result.schedule.makespan_s:.2f} s",
    ]
    for item in result.schedule.robots:
        lines.append(
            f"R{item.robot_id} : completion={item.completion_s:.2f} s | "
            f"D={item.deposition_time_s:.2f} s | T={item.travel_time_s:.2f} s | "
            f"W={item.wait_time_s:.2f} s"
        )
    lines.extend(["", "[Robot XY Reach]"])
    for reach_item in result.reach.robots:
        lines.append(
            f"R{reach_item.robot_id} : max_xy={reach_item.maximum_xy_distance_mm:.2f} mm | "
            f"limit={reach_item.xy_reach_radius_mm:.2f} mm | "
            f"xy_margin={reach_item.minimum_xy_margin_mm:.2f} mm | "
            f"violations={reach_item.xy_violation_point_count}"
        )
    arm_pair = result.collision.minimum_arm_pair
    tcp_pair = result.collision.minimum_tcp_pair
    lines.extend(
        [
            "",
            "[Collision]",
            f"ARM_ENVELOPE events: {result.collision.arm_envelope_event_count}",
            f"Minimum Arm safety margin: "
            f"{result.collision.minimum_arm_safety_margin_mm:.2f} mm "
            f"(R{arm_pair[0]}-R{arm_pair[1]}; centerline "
            f"{result.collision.arm_centerline_distance_at_worst_mm:.2f} mm; required >= "
            f"{result.collision.arm_required_distance_at_worst_mm:.2f} mm)",
            f"TCP_RADIUS events: {result.collision.tcp_radius_event_count}",
            f"Total events     : {len(result.collision.events)}",
            f"Minimum TCP distance : {result.collision.minimum_tcp_distance_mm:.2f} mm "
            f"(R{tcp_pair[0]}-R{tcp_pair[1]}; required >= "
            f"{result.collision.minimum_tcp_required_distance_mm:.2f} mm)",
            f"Collision-free       : {'YES' if result.collision_free else 'NO'}",
            "",
            "[Shape]",
            f"Target volume    : {result.shape.target_volume_mm3 / 1_000_000:.3f} L",
            f"Deposited volume : {result.shape.deposited_volume_mm3 / 1_000_000:.3f} L",
            f"Underfill volume : {result.shape.underfill_volume_mm3 / 1_000_000:.3f} L",
            f"Overfill volume  : {result.shape.overfill_volume_mm3 / 1_000_000:.3f} L",
            f"Coverage         : {_percent(result.shape.coverage)}",
            f"Underfill        : {_percent(result.shape.underfill_ratio)}",
            f"Overfill         : {_percent(result.shape.overfill_ratio)}",
            f"IoU              : {_percent(result.shape.iou)}",
            f"Failed layers    : {result.shape.failed_layer_count} / "
            f"{result.shape.evaluated_layer_count} "
            f"({_percent(result.shape.failed_layer_ratio)})",
            f"Shape-valid      : {'YES' if result.shape.passed else 'NO'}",
            "",
            "[Failure Reasons]",
        ]
    )
    if result.failure_reasons:
        lines.extend(f"{index}. {reason}" for index, reason in enumerate(result.failure_reasons, 1))
    else:
        lines.append("None")
    lines.extend(["", "[Warnings]"])
    if result.warnings:
        lines.append(
            f"{len(result.warnings)} warning(s). See summary.json and validation_report.md."
        )
    else:
        lines.append("None")
    lines.extend(["", f"Results saved to: {result.output_dir}", "=" * 60])
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f} %"


def render_error_block(error: WaamValidatorError, input_dir: Path) -> str:
    """Render a short fatal error block for stderr."""
    return "\n".join(
        [
            "=" * 60,
            "WAAM VALIDATION ERROR",
            "=" * 60,
            f"Code    : {error.code}",
            f"Message : {error.message}",
            f"Input   : {input_dir}",
            "=" * 60,
        ]
    )
