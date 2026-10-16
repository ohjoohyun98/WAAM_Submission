"""Read-only result and artifact loading for the single-job local UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from ..config.models import Config
from ..provenance import RESULT_SCHEMA_VERSION, verify_validation_inputs

JsonDict = dict[str, Any]

REQUIRED_INPUT_FILES: Final = ("config.yaml", "trajectory.csv", "target.stl")
ALLOWED_ARTIFACTS: Final = frozenset(
    {
        "summary.json",
        "error.json",
        "robot_metrics.csv",
        "collision_events.csv",
        "layer_metrics.csv",
        "validation_report.md",
        "deposited.stl",
        "run.log",
        "replay.html",
        "validation_inputs.json",
    }
)
_STAMP_PATTERN: Final = re.compile(r"^(?P<stamp>\d{4}-\d{2}-\d{2}_\d{6})(?:_(?P<suffix>\d{3}))?$")
CSV_COLUMNS: Final = {
    "robot_metrics.csv": (
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
    ),
    "collision_events.csv": (
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
    ),
    "layer_metrics.csv": (
        "layer_index",
        "z_slice_mm",
        "coverage",
        "underfill_ratio",
        "overfill_ratio",
        "iou",
        "passed",
    ),
}


class DashboardDataError(ValueError):
    """Raised when a UI path or result artifact is unsafe or invalid."""


@dataclass(slots=True, frozen=True)
class JobRecord:
    """One validation-ready job directory."""

    name: str
    path: Path


@dataclass(slots=True, frozen=True)
class RunRecord:
    """The newest completed result for a job."""

    directory: Path
    status: str
    payload: JsonDict
    completed_label: str
    load_error: str | None = None

    @property
    def run_name(self) -> str:
        return self.directory.name


def _run_sort_key(path: Path) -> tuple[int, float, int, str]:
    match = _STAMP_PATTERN.fullmatch(path.name)
    if match is not None:
        try:
            parsed = datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H%M%S")
        except ValueError:
            pass
        else:
            suffix = int(match.group("suffix") or 0)
            return (1, parsed.timestamp(), suffix, path.name)
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return (0, modified, 0, path.name)


def completed_run_directories(job_dir: Path) -> list[Path]:
    """Return terminal output directories, excluding partial runs."""
    output_root = job_dir / "output"
    if not output_root.is_dir():
        return []
    completed = [
        child
        for child in output_root.iterdir()
        if child.is_dir()
        and ((child / "summary.json").is_file() or (child / "error.json").is_file())
    ]
    return sorted(completed, key=_run_sort_key, reverse=True)


def latest_completed_run(job_dir: Path) -> Path | None:
    """Return the newest terminal run for a job."""
    runs = completed_run_directories(job_dir)
    return runs[0] if runs else None


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"{path.name}을 읽을 수 없습니다: {exc}") from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"{path.name}의 최상위 값은 JSON object여야 합니다.")
    return value


def _completed_label(run_dir: Path) -> str:
    match = _STAMP_PATTERN.fullmatch(run_dir.name)
    if match is not None:
        try:
            parsed = datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H%M%S")
        except ValueError:
            pass
        else:
            suffix = match.group("suffix")
            base = parsed.strftime("%Y-%m-%d %H:%M:%S")
            return f"{base} · {suffix}" if suffix else base
    try:
        return datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return run_dir.name


def load_latest_run(job_dir: Path) -> RunRecord | None:
    """Load the latest summary or fatal error without changing its verdict."""
    run_dir = latest_completed_run(job_dir)
    if run_dir is None:
        return None
    source = run_dir / ("error.json" if (run_dir / "error.json").is_file() else "summary.json")
    try:
        payload = _read_json(source)
        status = str(payload.get("status", "ERROR")).upper()
        if status not in {"PASS", "FAIL", "ERROR"}:
            raise DashboardDataError(f"알 수 없는 status 값입니다: {status}")
        return RunRecord(run_dir, status, payload, _completed_label(run_dir))
    except DashboardDataError as exc:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "ERROR",
            "code": "UI_DATA_ERROR",
            "message": str(exc),
        }
        return RunRecord(run_dir, "ERROR", payload, _completed_label(run_dir), str(exc))


def load_run_directory(run_dir: Path) -> RunRecord:
    """Load one exact completed output directory without latest-run rediscovery."""
    resolved = run_dir.expanduser().resolve()
    source = resolved / (
        "error.json" if (resolved / "error.json").is_file() else "summary.json"
    )
    if not source.is_file():
        raise DashboardDataError(f"완료된 결과가 아닙니다: {resolved}")
    payload = _read_json(source)
    status = str(payload.get("status", "ERROR")).upper()
    if status not in {"PASS", "FAIL", "ERROR"}:
        raise DashboardDataError(f"알 수 없는 status 값입니다: {status}")
    return RunRecord(resolved, status, payload, _completed_label(resolved))


def load_latest_matching_run(job_dir: Path) -> RunRecord | None:
    """Return the newest schema-compatible result for the current input signature."""
    for run_dir in completed_run_directories(job_dir):
        if not (run_dir / "summary.json").is_file():
            continue
        try:
            run = load_run_directory(run_dir)
        except DashboardDataError:
            continue
        if run.payload.get("schema_version") != RESULT_SCHEMA_VERSION:
            continue
        matches, _ = verify_validation_inputs(job_dir, run_dir)
        if matches:
            return run
    return None


def load_result_config(run: RunRecord) -> Config | None:
    """Load the immutable Config snapshot embedded in a completed result."""
    try:
        manifest = _read_json(run.directory / "validation_inputs.json")
    except DashboardDataError:
        return None
    raw = manifest.get("config")
    if not isinstance(raw, dict):
        return None
    try:
        return Config.model_validate(raw)
    except ValueError:
        return None


def thresholds_from_config(config: Config | None) -> JsonDict:
    """Return shape thresholds from a result-local Config snapshot."""
    if config is None:
        return {"error": "Validation 당시 Config snapshot이 없습니다. 다시 실행하세요."}
    shape = config.shape_validation
    return {
        "minimum_overall_coverage": shape.minimum_overall_coverage,
        "maximum_overall_overfill_ratio": shape.maximum_overall_overfill_ratio,
        "minimum_overall_iou": shape.minimum_overall_iou,
        "minimum_layer_iou": shape.minimum_layer_iou,
        "maximum_failed_layer_ratio": shape.maximum_failed_layer_ratio,
    }


def read_csv_records(run_dir: Path, filename: str) -> list[JsonDict]:
    """Read a small result CSV through Polars; missing files are valid empty states."""
    if filename not in ALLOWED_ARTIFACTS or not filename.endswith(".csv"):
        raise DashboardDataError(f"허용되지 않은 CSV입니다: {filename}")
    path = run_dir / filename
    if not path.is_file():
        return []
    try:
        return pl.read_csv(
            path,
            columns=list(CSV_COLUMNS.get(filename, ())),
            infer_schema_length=1000,
        ).to_dicts()
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise DashboardDataError(f"{filename}을 읽을 수 없습니다: {exc}") from exc


def read_csv_window(
    run_dir: Path,
    filename: str,
    *,
    start: int,
    end: int,
    sort_by: list[dict[str, str]] | None = None,
) -> tuple[list[JsonDict], int]:
    """Read one AG Grid row window and return the exact filtered row count."""
    if filename not in ALLOWED_ARTIFACTS or not filename.endswith(".csv"):
        raise DashboardDataError(f"허용되지 않은 CSV입니다: {filename}")
    path = run_dir / filename
    if not path.is_file():
        return [], 0
    try:
        frame = pl.read_csv(
            path,
            columns=list(CSV_COLUMNS.get(filename, ())),
            infer_schema_length=1000,
        )
        sort_columns = [
            item.get("column_id", "")
            for item in (sort_by or [])
            if item.get("column_id", "") in frame.columns
        ]
        if sort_columns:
            sort_directions = [
                item.get("direction") == "desc"
                for item in (sort_by or [])
                if item.get("column_id", "") in frame.columns
            ]
            frame = frame.sort(sort_columns, descending=sort_directions)
        safe_start = max(0, start)
        size = max(1, end - safe_start)
        return frame.slice(safe_start, size).to_dicts(), frame.height
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise DashboardDataError(f"{filename}을 읽을 수 없습니다: {exc}") from exc


def resolve_single_job_artifact(job_dir: Path, run_name: str, filename: str) -> Path:
    """Resolve one allowlisted artifact below a canonical single-job output root."""
    if filename not in ALLOWED_ARTIFACTS:
        raise DashboardDataError(f"허용되지 않은 산출물입니다: {filename}")
    output_root = (job_dir.expanduser().resolve() / "output").resolve()
    candidate = (output_root / run_name / filename).resolve()
    if not candidate.is_relative_to(output_root) or candidate.parent.parent != output_root:
        raise DashboardDataError("허용된 결과 폴더 밖의 경로입니다.")
    if not candidate.is_file():
        raise DashboardDataError(f"산출물이 존재하지 않습니다: {filename}")
    return candidate
