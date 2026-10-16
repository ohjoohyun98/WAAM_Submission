"""Isolated validation process management for the local UI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .data import JobRecord, RunRecord
from .replay_service import (
    DEPOSITED_STL_STATUS,
    REPLAY_STATUS,
    VALIDATION_STATUS,
    write_json_atomic,
)


class ValidationAlreadyRunningError(RuntimeError):
    """Raised when a second UI validation is requested."""


@dataclass(slots=True)
class ActiveRun:
    job: JobRecord
    output_dir: Path
    process: subprocess.Popen[bytes]
    started_at: datetime
    started_monotonic: float


@dataclass(slots=True)
class ActiveReplay:
    job: JobRecord
    run: RunRecord
    interval_s: float | None
    process: subprocess.Popen[bytes]
    started_at: datetime
    started_monotonic: float
    artifact_kind: str = "replay"


def allocate_output_path(job_dir: Path, now: datetime | None = None) -> Path:
    """Allocate, but do not create, the next timestamped output directory."""
    output_root = job_dir / "output"
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    candidate = output_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stamp}_{suffix:03d}"
        suffix += 1
    return candidate.resolve()


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def tail_log(output_dir: Path, line_count: int = 14) -> str:
    """Return a bounded tail of the active run log."""
    path = output_dir / "run.log"
    if not path.is_file():
        return "실행 로그를 준비하고 있습니다…"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "실행 로그를 읽을 수 없습니다."
    return "\n".join(lines[-max(1, line_count) :])


def _completed_result_verdict(output_dir: Path) -> str:
    """Infer a verdict from durable result markers, never a bare process code.

    Python also uses exit code 1 for an unhandled interpreter failure.  Treating
    every such exit as a normal validation FAIL would hide a crashed worker.
    """
    if (output_dir / "error.json").is_file():
        return "ERROR"
    summary = _read_status(output_dir / "summary.json")
    verdict = str(summary.get("status", "")).upper()
    return verdict if verdict in {"PASS", "FAIL"} else "ERROR"


class ValidationRunManager:
    """Run at most one validator worker without blocking the Dash server."""

    def __init__(self, python_executable: str | None = None) -> None:
        self._python = python_executable or sys.executable
        self._lock = threading.Lock()
        self._active: ActiveRun | None = None

    def start(self, job: JobRecord) -> Path:
        """Start a full validation worker for one allowlisted job."""
        with self._lock:
            if self._active is not None and self._active.process.poll() is None:
                raise ValidationAlreadyRunningError(
                    f"{self._active.job.name} validation이 이미 실행 중입니다."
                )
            output_dir = allocate_output_path(job.path)
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(
                [
                    self._python,
                    "-m",
                    "waam_validator.dashboard.worker",
                    str(job.path),
                    str(output_dir),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            now = datetime.now()
            self._active = ActiveRun(job, output_dir, process, now, time.monotonic())
            return output_dir

    def snapshot(self) -> dict[str, Any]:
        """Return the current process state as JSON-safe data."""
        with self._lock:
            active = self._active
            if active is None:
                return {
                    "state": "IDLE",
                    "stage": "idle",
                    "message": "실행 대기",
                    "running": False,
                    "elapsed_s": 0.0,
                    "log": "",
                }
            exit_code = active.process.poll()
            status_path = active.output_dir / VALIDATION_STATUS
            status = _read_status(status_path) if status_path.is_file() else {}
            running = exit_code is None
            if not status:
                status = {
                    "state": "RUNNING" if running else "FINISHED",
                    "stage": "starting" if running else "process_exit",
                    "message": (
                        "Validation 프로세스를 시작하고 있습니다…"
                        if running
                        else "Validation 프로세스가 상태 파일 없이 종료되었습니다."
                    ),
                }
            if not running:
                status["exit_code"] = exit_code
                if status.get("state") != "FINISHED":
                    status.update(
                        state="FINISHED",
                        stage="process_exit",
                        message="Validation 프로세스가 완료 상태를 기록하지 못하고 종료되었습니다.",
                    )
                inferred = _completed_result_verdict(active.output_dir)
                if str(status.get("verdict", "")).upper() not in {"PASS", "FAIL", "ERROR"}:
                    status["verdict"] = inferred
            elapsed = max(0.0, time.monotonic() - active.started_monotonic)
            return {
                **status,
                "running": running,
                "job_name": active.job.name,
                "output_directory": str(active.output_dir),
                "started_at": active.started_at.isoformat(timespec="seconds"),
                "elapsed_s": elapsed,
                "log": tail_log(active.output_dir),
            }


class ReplayAlreadyRunningError(RuntimeError):
    """Raised when another replay subprocess already owns the generator slot."""


class ReplayRunManager:
    """Generate at most one derived result artifact in an isolated process."""

    def __init__(self, python_executable: str | None = None) -> None:
        self._python = python_executable or sys.executable
        self._lock = threading.Lock()
        self._active: ActiveReplay | None = None

    def start(self, job: JobRecord, run: RunRecord, interval_s: float) -> Path:
        with self._lock:
            if self._active is not None and self._active.process.poll() is None:
                raise ReplayAlreadyRunningError(
                    f"{self._active.job.name} Replay가 이미 생성 중입니다."
                )
            write_json_atomic(
                run.directory / REPLAY_STATUS,
                {
                    "state": "RUNNING",
                    "stage": "starting",
                    "message": "Replay 생성 프로세스를 시작하고 있습니다.",
                    "progress": 0.0,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    [
                        self._python,
                        "-m",
                        "waam_validator.dashboard.replay_worker",
                        str(job.path),
                        str(run.directory),
                        f"{interval_s:.9g}",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                write_json_atomic(
                    run.directory / REPLAY_STATUS,
                    {
                        "state": "FINISHED",
                        "stage": "failed",
                        "message": f"Replay 생성 프로세스를 시작하지 못했습니다: {exc}",
                        "verdict": "ERROR",
                        "progress": 0.0,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                raise
            now = datetime.now()
            self._active = ActiveReplay(
                job,
                run,
                interval_s,
                process,
                now,
                time.monotonic(),
            )
            return run.directory / "replay.html"

    def start_deposited_stl(self, job: JobRecord, run: RunRecord) -> Path:
        """Start an on-demand nominal deposited STL worker."""
        with self._lock:
            if self._active is not None and self._active.process.poll() is None:
                raise ReplayAlreadyRunningError("다른 추가 산출물이 이미 생성 중입니다.")
            write_json_atomic(
                run.directory / DEPOSITED_STL_STATUS,
                {
                    "state": "RUNNING",
                    "stage": "starting",
                    "message": "적층 STL 생성 프로세스를 시작하고 있습니다.",
                    "progress": 0.0,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    [
                        self._python,
                        "-m",
                        "waam_validator.dashboard.deposited_stl_worker",
                        str(job.path),
                        str(run.directory),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                write_json_atomic(
                    run.directory / DEPOSITED_STL_STATUS,
                    {
                        "state": "FINISHED",
                        "stage": "failed",
                        "message": f"적층 STL 생성 프로세스를 시작하지 못했습니다: {exc}",
                        "verdict": "ERROR",
                        "progress": 0.0,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                raise
            now = datetime.now()
            self._active = ActiveReplay(
                job,
                run,
                None,
                process,
                now,
                time.monotonic(),
                artifact_kind="deposited_stl",
            )
            return run.directory / "deposited.stl"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            if active is None:
                return {
                    "state": "IDLE",
                    "stage": "idle",
                    "message": "Replay 생성 대기",
                    "running": False,
                    "elapsed_s": 0.0,
                    "progress": 0.0,
                }
            exit_code = active.process.poll()
            status_name = (
                REPLAY_STATUS if active.artifact_kind == "replay" else DEPOSITED_STL_STATUS
            )
            status_path = active.run.directory / status_name
            status = _read_status(status_path) if status_path.is_file() else {}
            running = exit_code is None
            if not status:
                status = {
                    "state": "RUNNING" if running else "FINISHED",
                    "stage": "starting" if running else "process_exit",
                    "message": (
                        "추가 산출물 생성 프로세스를 시작하고 있습니다."
                        if running
                        else "산출물 프로세스가 상태 파일 없이 종료되었습니다."
                    ),
                    "progress": 0.0,
                }
            if not running:
                status["exit_code"] = exit_code
                if status.get("state") != "FINISHED":
                    status.update(
                        state="FINISHED",
                        stage="process_exit",
                        message="산출물 프로세스가 완료 상태를 기록하지 못하고 종료되었습니다.",
                        verdict="ERROR",
                    )
                status.setdefault("verdict", "READY" if exit_code == 0 else "ERROR")
            elapsed = max(0.0, time.monotonic() - active.started_monotonic)
            return {
                **status,
                "running": running,
                "job_name": active.job.name,
                "run_name": active.run.run_name,
                "interval_s": active.interval_s,
                "artifact_kind": active.artifact_kind,
                "started_at": active.started_at.isoformat(timespec="seconds"),
                "elapsed_s": elapsed,
            }
