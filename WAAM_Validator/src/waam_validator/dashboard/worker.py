"""Subprocess entry point used by the local UI run manager."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import ComputationError, WaamValidatorError
from ..pipeline import run_validation
from ..progress import ValidationProgress
from .replay_service import VALIDATION_STATUS, write_json_atomic


def _write_status(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / VALIDATION_STATUS
    enriched = {**payload, "updated_at": datetime.now().isoformat(timespec="seconds")}
    write_json_atomic(target, enriched)


def run_worker(job_dir: Path, output_dir: Path) -> int:
    """Run validation and persist a small phase-oriented UI state."""
    last_write_s = 0.0
    last_progress = -1.0
    last_stage = ""
    current_progress = 0.0

    def report(event: ValidationProgress) -> None:
        nonlocal last_write_s, last_progress, last_stage, current_progress
        current_progress = max(current_progress, event.overall_progress)
        now = time.monotonic()
        should_write = (
            event.stage != last_stage
            or event.overall_progress >= 1.0
            or event.overall_progress - last_progress >= 0.01
            or now - last_write_s >= 0.5
        )
        if not should_write:
            return
        _write_status(
            output_dir,
            {
                "state": "RUNNING",
                "stage": event.stage,
                "message": event.message,
                "stage_progress": event.stage_progress,
                "overall_progress": current_progress,
                "completed_units": event.completed_units,
                "total_units": event.total_units,
                "unit": event.unit,
            },
        )
        last_write_s = now
        last_progress = event.overall_progress
        last_stage = event.stage

    try:
        result = run_validation(
            job_dir,
            output_dir,
            progress_callback=report,
        )
    except WaamValidatorError as exc:
        _write_status(
            output_dir,
            {
                "state": "FINISHED",
                "stage": "failed",
                "message": exc.message,
                "verdict": "ERROR",
                "code": exc.code,
                "exit_code": exc.exit_code,
                "overall_progress": current_progress,
            },
        )
        return exc.exit_code
    except Exception as exc:  # Defensive subprocess boundary.
        wrapped = ComputationError("INTERNAL_CALCULATION_ERROR", str(exc))
        _write_status(
            output_dir,
            {
                "state": "FINISHED",
                "stage": "failed",
                "message": wrapped.message,
                "verdict": "ERROR",
                "code": wrapped.code,
                "exit_code": wrapped.exit_code,
                "overall_progress": current_progress,
            },
        )
        return wrapped.exit_code
    try:
        exit_code = 0 if result.status == "PASS" else 1
        _write_status(
            output_dir,
            {
                "state": "FINISHED",
                "stage": "completed",
                "message": f"Validation이 {result.status}로 완료됐습니다.",
                "verdict": result.status,
                "exit_code": exit_code,
                "stage_progress": 1.0,
                "overall_progress": 1.0,
            },
        )
        return exit_code
    except Exception as exc:  # Keep UI state terminal even if final bookkeeping fails.
        wrapped = ComputationError("UI_FINALIZATION_ERROR", str(exc))
        _write_status(
            output_dir,
            {
                "state": "FINISHED",
                "stage": "failed",
                "message": wrapped.message,
                "verdict": "ERROR",
                "code": wrapped.code,
                "exit_code": wrapped.exit_code,
                "overall_progress": 1.0,
            },
        )
        return wrapped.exit_code


def main(argv: list[str] | None = None) -> int:
    """Parse the two internal positional arguments."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        return 2
    return run_worker(Path(arguments[0]), Path(arguments[1]))


if __name__ == "__main__":
    raise SystemExit(main())
