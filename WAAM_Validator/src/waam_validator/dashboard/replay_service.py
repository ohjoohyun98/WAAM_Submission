"""Replay presets, conservative estimates, and input provenance helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .data import JobRecord, RunRecord

REPLAY_PRESET_FRAMES: Final = {"fast": 300, "standard": 600, "detail": 1_200}
MAX_REPLAY_FRAMES: Final = 2_000
STATE_DIRECTORY: Final = ".waam_state"
VALIDATION_STATUS: Final = Path(STATE_DIRECTORY) / "validation-status.json"
REPLAY_MANIFEST: Final = Path(STATE_DIRECTORY) / "replay-manifest.json"
REPLAY_STATUS: Final = Path(STATE_DIRECTORY) / "replay-status.json"
DEPOSITED_STL_MANIFEST: Final = Path(STATE_DIRECTORY) / "deposited-stl-manifest.json"
DEPOSITED_STL_STATUS: Final = Path(STATE_DIRECTORY) / "deposited-stl-status.json"


@dataclass(slots=True, frozen=True)
class ReplayEstimate:
    """A deliberately ranged estimate suitable for display before generation."""

    interval_s: float
    frame_count: int
    time_low_s: float
    time_high_s: float
    size_low_bytes: int
    size_high_bytes: int
    allowed: bool
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DepositedStlEstimate:
    """Conservative estimate for rebuilding the nominal deposited STL."""

    time_low_s: float
    time_high_s: float
    size_low_bytes: int
    size_high_bytes: int


def estimate_deposited_stl(run: RunRecord) -> DepositedStlEstimate:
    input_section = run.payload.get("input")
    rows_raw = input_section.get("trajectory_rows") if isinstance(input_section, dict) else 0
    rows = max(1, int(rows_raw)) if isinstance(rows_raw, int | float) else 1
    central_size = max(100_000, rows * 165)
    central_time = max(1.0, 1.0 + rows / 45_000.0)
    prior = _read_json_dict(run.directory / DEPOSITED_STL_MANIFEST)
    if prior is not None:
        prior_seconds = _positive_number(prior.get("duration_s"))
        prior_bytes = _positive_number(prior.get("size_bytes"))
        if prior_seconds and prior_bytes:
            central_time = prior_seconds
            central_size = int(prior_bytes)
    return DepositedStlEstimate(
        time_low_s=max(1.0, central_time * 0.6),
        time_high_s=max(2.0, central_time * 2.0),
        size_low_bytes=max(1, int(central_size * 0.55)),
        size_high_bytes=max(1, int(central_size * 1.8)),
    )


def recommended_interval_s(makespan_s: float, target_frames: int = 600) -> float:
    """Round upward to a human-friendly interval for the requested frame budget."""
    if not math.isfinite(makespan_s) or makespan_s <= 0:
        return 1.0
    raw = makespan_s / max(1, target_frames)
    if raw >= 60.0:
        return float(max(60, math.ceil(raw / 60.0) * 60))
    if raw >= 10.0:
        return float(math.ceil(raw / 10.0) * 10)
    return float(max(1, math.ceil(raw)))


def preset_interval_s(makespan_s: float, preset: str) -> float:
    """Return an interval for fast, standard, or detail replay quality."""
    return recommended_interval_s(makespan_s, REPLAY_PRESET_FRAMES.get(preset, 600))


def _collision_keyframe_candidate_count(run_dir: Path, makespan_s: float) -> int:
    """Count unique event minima and boundaries used by the Replay keyframe planner."""
    path = run_dir / "collision_events.csv"
    if not path.is_file():
        return 0
    values: set[float] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for field in ("start_s", "end_s", "minimum_distance_time_s"):
                    raw = row.get(field)
                    if raw in (None, ""):
                        continue
                    value = float(raw)
                    if math.isfinite(value) and 0.0 <= value <= makespan_s:
                        values.add(value)
    except (OSError, UnicodeError, ValueError, csv.Error):
        return 0
    return len(values)


def estimate_replay(job: JobRecord, run: RunRecord, interval_s: float) -> ReplayEstimate:
    """Estimate output cost from frame count and source file sizes."""
    schedule = run.payload.get("schedule")
    makespan_raw = schedule.get("makespan_s") if isinstance(schedule, dict) else None
    makespan = float(makespan_raw) if isinstance(makespan_raw, int | float) else 0.0
    collision = run.payload.get("collision")
    event_raw = collision.get("collision_event_count") if isinstance(collision, dict) else 0
    event_count = int(event_raw) if isinstance(event_raw, int | float) else 0
    if not math.isfinite(interval_s) or interval_s <= 0 or makespan <= 0:
        return ReplayEstimate(interval_s, 0, 0.0, 0.0, 0, 0, False, "간격을 확인하세요.")

    regular_frames = math.ceil(makespan / interval_s) + 1
    allowed = regular_frames <= MAX_REPLAY_FRAMES
    mode_keyframe_budget = min(
        max(0, MAX_REPLAY_FRAMES - regular_frames),
        math.ceil(regular_frames * 0.1),
    )
    collision_keyframe_budget = max(
        0,
        MAX_REPLAY_FRAMES - regular_frames - mode_keyframe_budget,
    )
    collision_candidates = _collision_keyframe_candidate_count(run.directory, makespan)
    if collision_candidates == 0 and event_count > 0:
        # Results without a readable event CSV cannot be exact, but retaining a
        # conservative estimate keeps the UI useful for optional-output runs.
        collision_candidates = event_count * 3
    selected_collision_frames = min(collision_candidates, collision_keyframe_budget)
    frame_count = min(
        MAX_REPLAY_FRAMES,
        regular_frames + mode_keyframe_budget + selected_collision_frames,
    )
    warning = ""
    if not allowed:
        minimum = recommended_interval_s(makespan, MAX_REPLAY_FRAMES)
        warning = (
            f"정규 프레임이 안전 한도 {MAX_REPLAY_FRAMES:,}개를 초과합니다. "
            f"간격을 최소 {minimum:g}초로 늘리세요."
        )
    elif collision_candidates > collision_keyframe_budget:
        warning = (
            f"충돌 이벤트가 많아 {collision_candidates:,}개의 후보 시점 중 "
            f"{collision_keyframe_budget:,}개를 선택합니다. 최악 충돌 시점과 "
            "시간축 전반의 대표 경계를 우선 보존합니다."
        )
    elif frame_count > 1_200:
        warning = "상세 설정입니다. 생성 시간과 브라우저 메모리 사용량이 증가합니다."

    trajectory_mb = _size_mb(job.path / "trajectory.csv")
    target_mb = _size_mb(job.path / "target.stl")
    central_size_mb = 5.0 + trajectory_mb * 0.55 + target_mb * 1.5 + frame_count * 0.0006
    central_time_s = 1.5 + trajectory_mb * 0.12 + target_mb * 0.1 + frame_count * 0.0007
    prior = read_replay_manifest(run.directory)
    if prior is not None:
        prior_frames = _positive_number(prior.get("frame_count"))
        prior_seconds = _positive_number(prior.get("duration_s"))
        prior_bytes = _positive_number(prior.get("size_bytes"))
        if prior_frames and prior_seconds and prior_bytes:
            ratio = frame_count / prior_frames
            central_time_s = (prior_seconds + 1.5) * (0.7 + 0.3 * ratio)
            central_size_mb = prior_bytes / (1024 * 1024) * (0.8 + 0.2 * ratio)
    return ReplayEstimate(
        interval_s=interval_s,
        frame_count=frame_count,
        time_low_s=max(1.0, central_time_s * 0.65),
        time_high_s=max(2.0, central_time_s * 1.7),
        size_low_bytes=max(1, int(central_size_mb * 0.75 * 1024 * 1024)),
        size_high_bytes=max(1, int(central_size_mb * 1.35 * 1024 * 1024)),
        allowed=allowed,
        warning=warning,
    )


def read_replay_manifest(run_dir: Path) -> dict[str, Any] | None:
    return _read_json_dict(run_dir / REPLAY_MANIFEST)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_replay_status(run_dir: Path) -> dict[str, Any] | None:
    """Read the latest persistent Replay worker status, if present."""
    return _read_json_dict(run_dir / REPLAY_STATUS)


def read_deposited_stl_status(run_dir: Path) -> dict[str, Any] | None:
    return _read_json_dict(run_dir / DEPOSITED_STL_STATUS)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def _size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def _positive_number(value: object) -> float | None:
    if isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0:
        return float(value)
    return None
