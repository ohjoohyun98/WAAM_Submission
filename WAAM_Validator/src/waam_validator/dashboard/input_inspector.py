"""One-shot input inspection for a single UI validation job."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..config.loader import load_config
from ..config.models import Config
from ..constants import MODE_D, MODE_T, MODE_W
from ..errors import ValidationIssue, ValidationMessages, WaamValidatorError
from ..models import TrajectorySet
from ..provenance import file_fingerprint
from ..schedule.metrics import compute_schedule_metrics
from ..shape.target import load_target_mesh, validate_coordinate_consistency
from ..trajectory.loader import load_trajectory_csv
from ..trajectory.reach import compute_reach_metrics
from ..trajectory.validator import validate_trajectory_set
from .data import REQUIRED_INPUT_FILES, DashboardDataError

JsonDict = dict[str, Any]
InspectionStatus = Literal["READY", "WARNING", "EXPECTED_FAIL", "BLOCKED"]


@dataclass(slots=True, frozen=True)
class DashboardInputPaths:
    """Resolved, allowlisted paths for one UI job."""

    job_dir: Path
    config: Path
    trajectory: Path
    target: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "job_dir": str(self.job_dir),
            "config.yaml": str(self.config),
            "trajectory.csv": str(self.trajectory),
            "target.stl": str(self.target),
        }


@dataclass(slots=True, frozen=True)
class InputInspection:
    """JSON-safe result of a complete input preflight."""

    status: InspectionStatus
    can_run: bool
    checked_at: str
    paths: JsonDict
    signature: JsonDict
    files: JsonDict
    config: JsonDict
    trajectory: JsonDict
    target: JsonDict
    warnings: list[JsonDict]
    expected_failures: list[JsonDict]
    blocking_errors: list[JsonDict]

    def to_dict(self) -> JsonDict:
        return asdict(self)


def _clean_path(value: str | Path) -> Path:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        raise DashboardDataError("경로를 입력하세요.")
    return Path(text).expanduser().resolve()


def resolve_input_directory(job_dir: str | Path) -> DashboardInputPaths:
    """Resolve the three fixed inputs from one unrestricted canonical job directory."""
    resolved_job = _clean_path(job_dir)
    if not resolved_job.is_dir():
        raise DashboardDataError(f"입력 폴더가 존재하지 않습니다: {resolved_job}")
    resolved_files: list[Path] = []
    for filename in REQUIRED_INPUT_FILES:
        candidate = resolved_job / filename
        if not candidate.is_file():
            raise DashboardDataError(f"입력 파일이 존재하지 않습니다: {candidate}")
        resolved = candidate.resolve()
        if resolved.parent != resolved_job or resolved.name != filename:
            raise DashboardDataError(f"{filename}은 입력 폴더 밖을 참조할 수 없습니다.")
        resolved_files.append(resolved)
    return DashboardInputPaths(
        job_dir=resolved_job,
        config=resolved_files[0],
        trajectory=resolved_files[1],
        target=resolved_files[2],
    )


def input_signature(paths: DashboardInputPaths) -> JsonDict:
    """Return a content-backed signature used to invalidate a stale preflight."""
    result: JsonDict = {}
    for filename, path in (
        ("config.yaml", paths.config),
        ("trajectory.csv", paths.trajectory),
        ("target.stl", paths.target),
    ):
        fingerprint = file_fingerprint(path)
        # Nanosecond timestamps exceed JavaScript's safe integer range. This
        # signature crosses a browser Store, so serialize that one field only.
        fingerprint["mtime_ns"] = str(fingerprint["mtime_ns"])
        result[filename] = {
            "path": str(path),
            **fingerprint,
        }
    return result


def inspection_is_current(inspection: JsonDict, paths: DashboardInputPaths) -> bool:
    """Return whether stored inspection paths and file identities still match."""
    try:
        return bool(inspection.get("can_run")) and inspection.get("signature") == input_signature(
            paths
        )
    except OSError:
        return False


def _file_details(path: Path) -> JsonDict:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "status": "OK",
    }


def _issue_dict(issue: ValidationIssue) -> JsonDict:
    return asdict(issue)


def _config_details(config: Config) -> JsonDict:
    return config.model_dump(mode="json")


def _trajectory_details(trajectories: TrajectorySet, config: Config) -> JsonDict:
    schedule = compute_schedule_metrics(trajectories)
    reach_by_robot = {
        item.robot_id: item for item in compute_reach_metrics(trajectories, config).robots
    }
    robot_details: list[JsonDict] = []
    for trajectory, metrics in zip(trajectories.robots, schedule.robots, strict=True):
        interval_modes = trajectory.mode[:-1]
        counts = {
            "D": int(np.count_nonzero(interval_modes == MODE_D)),
            "T": int(np.count_nonzero(interval_modes == MODE_T)),
            "W": int(np.count_nonzero(interval_modes == MODE_W)),
        }
        minimum = trajectory.xyz_mm.min(axis=0).astype(np.float64)
        maximum = trajectory.xyz_mm.max(axis=0).astype(np.float64)
        robot_details.append(
            {
                "robot_id": trajectory.robot_id,
                "row_count": len(trajectory.time_s),
                "start_s": float(trajectory.time_s[0]),
                "end_s": float(trajectory.time_s[-1]),
                "xyz_min_mm": minimum.tolist(),
                "xyz_max_mm": maximum.tolist(),
                "mode_interval_counts": counts,
                "deposition_time_s": metrics.deposition_time_s,
                "travel_time_s": metrics.travel_time_s,
                "wait_time_s": metrics.wait_time_s,
                "deposition_length_mm": metrics.deposition_length_mm,
                "travel_length_mm": metrics.travel_length_mm,
                "mean_deposition_speed_mm_s": metrics.mean_deposition_speed_mm_s,
                "mean_travel_speed_mm_s": metrics.mean_travel_speed_mm_s,
                "xy_reach_radius_mm": (
                    reach_by_robot[trajectory.robot_id].xy_reach_radius_mm
                ),
                "maximum_xy_distance_mm": (
                    reach_by_robot[trajectory.robot_id].maximum_xy_distance_mm
                ),
                "minimum_xy_margin_mm": (
                    reach_by_robot[trajectory.robot_id].minimum_xy_margin_mm
                ),
                "xy_utilization_ratio": (
                    reach_by_robot[trajectory.robot_id].xy_utilization_ratio
                ),
                "xy_violation_point_count": (
                    reach_by_robot[trajectory.robot_id].xy_violation_point_count
                ),
            }
        )
    return {
        "row_count": trajectories.row_count,
        "makespan_s": schedule.makespan_s,
        "workload_imbalance_s": schedule.workload_imbalance_s,
        "robots": robot_details,
    }


def _target_details(mesh: Any) -> JsonDict:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    body_count_raw = mesh.metadata.get("waam_body_count", 1)
    body_count = int(body_count_raw) if isinstance(body_count_raw, int | np.integer) else 1
    volume = float(abs(mesh.volume))
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "body_count": body_count,
        "bounds_min_mm": bounds[0].tolist(),
        "bounds_max_mm": bounds[1].tolist(),
        "dimensions_mm": extents.tolist(),
        "volume_mm3": volume if math.isfinite(volume) else None,
        "watertight": bool(mesh.is_watertight),
    }


def inspect_dashboard_input_bundle(
    paths: DashboardInputPaths,
) -> tuple[InputInspection, Config | None, TrajectorySet | None, Any | None]:
    """Read inputs once and retain parsed objects for a bounded UI preview."""
    starting_signature = input_signature(paths)
    blocking: list[JsonDict] = []
    warnings: list[JsonDict] = []
    expected_failures: list[JsonDict] = []
    files = {
        filename: _file_details(path)
        for filename, path in (
            ("config.yaml", paths.config),
            ("trajectory.csv", paths.trajectory),
            ("target.stl", paths.target),
        )
    }
    config_details: JsonDict = {}
    trajectory_details: JsonDict = {}
    target_details: JsonDict = {}
    config: Config | None = None
    trajectories: TrajectorySet | None = None
    mesh: Any | None = None
    target_messages = ValidationMessages()

    try:
        config = load_config(paths.config)
        config_details = _config_details(config)
    except WaamValidatorError as exc:
        blocking.append({"code": exc.code, "message": exc.message, "source": "config.yaml"})

    if config is not None:
        try:
            trajectories = load_trajectory_csv(paths.trajectory, config)
            messages = validate_trajectory_set(trajectories, config)
            trajectory_details = _trajectory_details(trajectories, config)
            warnings.extend(_issue_dict(issue) for issue in messages.warnings)
            expected_failures.extend(_issue_dict(issue) for issue in messages.violations)
        except WaamValidatorError as exc:
            blocking.append({"code": exc.code, "message": exc.message, "source": "trajectory.csv"})

        try:
            mesh = load_target_mesh(paths.target, config, target_messages)
            target_details = _target_details(mesh)
            warnings.extend(_issue_dict(issue) for issue in target_messages.warnings)
        except WaamValidatorError as exc:
            blocking.append({"code": exc.code, "message": exc.message, "source": "target.stl"})

    if config is not None and trajectories is not None and mesh is not None:
        try:
            validate_coordinate_consistency(trajectories, mesh, config)
        except WaamValidatorError as exc:
            blocking.append({"code": exc.code, "message": exc.message, "source": "combined"})

    ending_signature = input_signature(paths)
    if ending_signature != starting_signature:
        blocking.append(
            {
                "code": "INPUT_CHANGED_DURING_INSPECTION",
                "message": "입력 확인 중 파일이 변경되었습니다. 저장이 끝난 뒤 다시 확인하세요.",
                "source": "combined",
            }
        )

    status: InspectionStatus
    if blocking:
        status = "BLOCKED"
    elif expected_failures:
        status = "EXPECTED_FAIL"
    elif warnings:
        status = "WARNING"
    else:
        status = "READY"
    inspection = InputInspection(
        status=status,
        can_run=not blocking,
        checked_at=datetime.now().isoformat(timespec="seconds"),
        paths=paths.as_dict(),
        signature=ending_signature,
        files=files,
        config=config_details,
        trajectory=trajectory_details,
        target=target_details,
        warnings=warnings,
        expected_failures=expected_failures,
        blocking_errors=blocking,
    )
    return inspection, config, trajectories, mesh
