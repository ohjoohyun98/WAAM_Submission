"""Subprocess entry point for on-demand UI Replay generation."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config.loader import load_config
from ..errors import ValidationMessages, WaamValidatorError
from ..models import CollisionEvent
from ..provenance import RESULT_SCHEMA_VERSION, input_signature, verify_validation_inputs
from ..shape.target import load_target_mesh
from ..trajectory.loader import load_trajectory_csv
from ..trajectory.validator import validate_trajectory_set
from ..visualization.replay import generate_replay_html
from .replay_service import (
    REPLAY_MANIFEST,
    REPLAY_STATUS,
    write_json_atomic,
)


def _status(run_dir: Path, **payload: Any) -> None:
    write_json_atomic(
        run_dir / REPLAY_STATUS,
        {**payload, "updated_at": datetime.now().isoformat(timespec="seconds")},
    )


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _load_events(path: Path) -> list[CollisionEvent]:
    if not path.is_file():
        return []
    events: list[CollisionEvent] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            events.append(
                CollisionEvent(
                    event_id=int(row["event_id"]),
                    collision_type=row["type"],
                    robot_a=int(row["robot_a"]),
                    robot_b=int(row["robot_b"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    duration_s=float(row["duration_s"]),
                    minimum_distance_mm=float(row["minimum_distance_mm"]),
                    required_distance_mm=float(row["required_distance_mm"]),
                    minimum_safety_margin_mm=float(row["minimum_safety_margin_mm"]),
                    minimum_capsule_surface_clearance_mm=_optional_float(
                        row.get("minimum_capsule_surface_clearance_mm")
                    ),
                    minimum_distance_time_s=float(row["minimum_distance_time_s"]),
                    closest_a_x_mm=float(row["closest_a_x_mm"]),
                    closest_a_y_mm=float(row["closest_a_y_mm"]),
                    closest_b_x_mm=float(row["closest_b_x_mm"]),
                    closest_b_y_mm=float(row["closest_b_y_mm"]),
                )
            )
    return events


def _expected_rows(run_dir: Path) -> int | None:
    path = run_dir / "summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        input_section = payload.get("input") if isinstance(payload, dict) else None
        value = input_section.get("trajectory_rows") if isinstance(input_section, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return int(value) if isinstance(value, int | float) else None


def run_replay_worker(job_dir: Path, run_dir: Path, interval_s: float) -> int:
    """Generate one replay atomically from an already completed validation run."""
    started = time.monotonic()
    temporary = run_dir / ".replay.html.tmp"
    try:
        _status(
            run_dir,
            state="RUNNING",
            stage="loading",
            message="Replay 입력을 불러오는 중입니다.",
            progress=0.0,
        )
        starting_signature = input_signature(job_dir)
        unchanged, message = verify_validation_inputs(
            job_dir,
            run_dir,
            current_signature=starting_signature,
        )
        if not unchanged:
            raise ValueError(message)
        config = load_config(job_dir / "config.yaml")
        trajectories = load_trajectory_csv(job_dir / "trajectory.csv", config)
        validate_trajectory_set(trajectories, config)
        expected_rows = _expected_rows(run_dir)
        if expected_rows is not None and expected_rows != trajectories.row_count:
            raise ValueError(
                "검증 이후 trajectory.csv가 변경되었습니다. Validation을 다시 실행하세요."
            )
        target = load_target_mesh(job_dir / "target.stl", config, ValidationMessages())
        events = _load_events(run_dir / "collision_events.csv")

        def report(completed: int, total: int) -> None:
            _status(
                run_dir,
                state="RUNNING",
                stage="frames",
                message=f"Replay 프레임 계산 중 · {completed:,}/{total:,}",
                progress=completed / max(1, total),
                completed_frames=completed,
                total_frames=total,
            )

        stats = generate_replay_html(
            trajectories,
            target,
            events,
            config,
            temporary,
            interval_s=interval_s,
            progress_callback=report,
        )
        _status(
            run_dir,
            state="RUNNING",
            stage="writing",
            message="Self-contained HTML을 저장하는 중입니다.",
            progress=1.0,
            completed_frames=stats.frame_count,
            total_frames=stats.frame_count,
        )
        if input_signature(job_dir) != starting_signature:
            raise ValueError(
                "Replay 생성 중 입력 파일이 변경되었습니다. Validation을 다시 실행하세요."
            )
        os.replace(temporary, run_dir / "replay.html")
        duration_s = time.monotonic() - started
        size_bytes = (run_dir / "replay.html").stat().st_size
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "READY",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "interval_s": interval_s,
            "frame_count": stats.frame_count,
            "deposition_point_count": stats.deposition_point_count,
            "duration_s": duration_s,
            "size_bytes": size_bytes,
            "inputs": starting_signature,
        }
        write_json_atomic(run_dir / REPLAY_MANIFEST, manifest)
        _status(
            run_dir,
            state="FINISHED",
            stage="completed",
            message="Replay 생성이 완료되었습니다.",
            verdict="READY",
            progress=1.0,
            **manifest,
        )
        return 0
    except (WaamValidatorError, OSError, UnicodeError, ValueError, csv.Error) as exc:
        temporary.unlink(missing_ok=True)
        _status(
            run_dir,
            state="FINISHED",
            stage="failed",
            message=str(exc),
            verdict="ERROR",
            progress=0.0,
        )
        return 4
    except Exception as exc:  # Defensive subprocess boundary.
        temporary.unlink(missing_ok=True)
        _status(
            run_dir,
            state="FINISHED",
            stage="failed",
            message=f"Replay 생성 중 예기치 않은 오류가 발생했습니다: {exc}",
            verdict="ERROR",
            progress=0.0,
        )
        return 5


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        return 2
    try:
        interval_s = float(arguments[2])
    except ValueError:
        return 2
    return run_replay_worker(Path(arguments[0]), Path(arguments[1]), interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
