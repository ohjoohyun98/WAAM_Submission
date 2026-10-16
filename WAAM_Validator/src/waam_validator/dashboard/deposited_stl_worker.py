"""Subprocess entry point for on-demand nominal deposited STL generation."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config.loader import load_config
from ..errors import WaamValidatorError
from ..provenance import RESULT_SCHEMA_VERSION, input_signature, verify_validation_inputs
from ..shape.deposition import build_deposited_layers
from ..shape.mesh_export import export_deposited_stl
from ..trajectory.loader import load_trajectory_csv
from ..trajectory.validator import validate_trajectory_set
from .replay_service import (
    DEPOSITED_STL_MANIFEST,
    DEPOSITED_STL_STATUS,
    write_json_atomic,
)


def _status(run_dir: Path, **payload: Any) -> None:
    write_json_atomic(
        run_dir / DEPOSITED_STL_STATUS,
        {**payload, "updated_at": datetime.now().isoformat(timespec="seconds")},
    )


def run_deposited_stl_worker(job_dir: Path, run_dir: Path) -> int:
    """Rebuild nominal deposited geometry and export it atomically."""
    started = time.monotonic()
    temporary = run_dir / ".deposited.stl.tmp"
    try:
        _status(
            run_dir,
            state="RUNNING",
            stage="loading",
            message="적층 경로를 불러오는 중입니다.",
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

        def report(fraction: float, completed: float, total: float) -> None:
            _status(
                run_dir,
                state="RUNNING",
                stage="geometry",
                message=f"Layer 형상 계산 중 · {int(completed):,}/{int(total):,}",
                progress=min(0.9, max(0.0, fraction) * 0.9),
            )

        layers = build_deposited_layers(
            trajectories,
            config,
            progress_callback=report,
        )
        _status(
            run_dir,
            state="RUNNING",
            stage="writing",
            message="STL mesh를 저장하는 중입니다.",
            progress=0.92,
        )
        export_deposited_stl(layers, config, temporary)
        if input_signature(job_dir) != starting_signature:
            raise ValueError(
                "적층 STL 생성 중 입력 파일이 변경되었습니다. Validation을 다시 실행하세요."
            )
        os.replace(temporary, run_dir / "deposited.stl")
        duration_s = time.monotonic() - started
        size_bytes = (run_dir / "deposited.stl").stat().st_size
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "READY",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": duration_s,
            "size_bytes": size_bytes,
            "layer_count": len(layers),
            "inputs": starting_signature,
        }
        write_json_atomic(run_dir / DEPOSITED_STL_MANIFEST, manifest)
        _status(
            run_dir,
            state="FINISHED",
            stage="completed",
            message="deposited.stl 생성이 완료되었습니다.",
            verdict="READY",
            progress=1.0,
            **manifest,
        )
        return 0
    except (WaamValidatorError, OSError, UnicodeError, ValueError) as exc:
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
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        _status(
            run_dir,
            state="FINISHED",
            stage="failed",
            message=f"적층 STL 생성 중 예기치 않은 오류가 발생했습니다: {exc}",
            verdict="ERROR",
            progress=0.0,
        )
        return 5


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        return 2
    return run_deposited_stl_worker(Path(arguments[0]), Path(arguments[1]))


if __name__ == "__main__":
    raise SystemExit(main())
