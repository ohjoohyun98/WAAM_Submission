"""Strict six-column trajectory CSV loader."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import polars as pl

from ..config.models import Config
from ..constants import CSV_COLUMNS, MODE_FROM_TEXT, MODE_W, ROBOT_IDS
from ..errors import InputValidationError
from ..models import RobotTrajectory, TrajectorySet

_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")


def _validate_header(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != list(CSV_COLUMNS):
                expected = ",".join(CSV_COLUMNS)
                raise InputValidationError(
                    "INVALID_CSV_HEADER",
                    f"trajectory.csv must contain exactly: {expected}",
                )
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(CSV_COLUMNS):
                    raise InputValidationError(
                        "INVALID_CSV_HEADER",
                        f"CSV row {row_number} must contain exactly six cells.",
                    )
                if "" in row:
                    raise InputValidationError(
                        "NONFINITE_VALUE", f"CSV row {row_number} contains an empty cell."
                    )
                if not _INTEGER_PATTERN.fullmatch(row[0]):
                    raise InputValidationError(
                        "INVALID_ROBOT_ID",
                        f"CSV row {row_number} robot_id must be an integer 1, 2, or 3.",
                    )
                if row[5] not in MODE_FROM_TEXT:
                    raise InputValidationError(
                        "INVALID_CSV_HEADER",
                        f"CSV row {row_number} mode must be exactly T, D, or W.",
                    )
    except InputValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputValidationError("INVALID_CSV_HEADER", str(exc)) from exc


def _read_frame(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(
            path,
            schema={
                "robot_id": pl.Int64,
                "time_s": pl.Float64,
                "x_mm": pl.Float64,
                "y_mm": pl.Float64,
                "z_mm": pl.Float64,
                "mode": pl.String,
            },
            null_values=[],
            ignore_errors=False,
            try_parse_dates=False,
            encoding="utf8",
        )
    except Exception as exc:
        raise InputValidationError("INVALID_CSV_HEADER", f"Unable to parse CSV: {exc}") from exc


def load_trajectory_csv(path: Path, config: Config) -> TrajectorySet:
    """Strictly load the six-column CSV and convert to compact arrays."""
    del config  # Schema loading itself is config-independent.
    if not path.is_file():
        raise InputValidationError("MISSING_TRAJECTORY", "trajectory.csv is required.")
    _validate_header(path)
    frame = _read_frame(path)
    if tuple(frame.columns) != CSV_COLUMNS:
        raise InputValidationError(
            "INVALID_CSV_HEADER", "trajectory.csv header or column order is invalid."
        )
    if frame.height == 0:
        raise InputValidationError("MISSING_ROBOT", "trajectory.csv has no data rows.")

    if any(frame.null_count().row(0)):
        raise InputValidationError("NONFINITE_VALUE", "trajectory.csv contains a null value.")
    try:
        robot_ids = np.asarray(frame["robot_id"], dtype=np.int64)
        time_s = np.asarray(frame["time_s"], dtype=np.float64)
        xyz64 = np.column_stack(
            [np.asarray(frame[name], dtype=np.float64) for name in ("x_mm", "y_mm", "z_mm")]
        )
        modes = np.asarray(
            frame["mode"].replace_strict(MODE_FROM_TEXT, return_dtype=pl.UInt8),
            dtype=np.uint8,
        )
    except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        raise InputValidationError(
            "NONFINITE_VALUE", f"CSV numeric conversion failed: {exc}"
        ) from exc
    if not np.isfinite(time_s).all() or not np.isfinite(xyz64).all():
        raise InputValidationError("NONFINITE_VALUE", "CSV contains NaN or infinite values.")
    if np.any(~np.isin(robot_ids, ROBOT_IDS)):
        raise InputValidationError("INVALID_ROBOT_ID", "robot_id must be one of 1, 2, and 3.")

    sorted_indices = np.lexsort((time_s, robot_ids))
    if not np.array_equal(sorted_indices, np.arange(frame.height)):
        raise InputValidationError(
            "UNSORTED_TRAJECTORY",
            "Rows must be sorted by robot_id and then strictly by time_s.",
        )

    xyz32 = xyz64.astype(np.float32)
    if not np.isfinite(xyz32).all():
        raise InputValidationError(
            "NONFINITE_VALUE", "Coordinate magnitude overflows the required float32 dtype."
        )

    robots: list[RobotTrajectory] = []
    for robot_id in ROBOT_IDS:
        mask = robot_ids == robot_id
        count = int(mask.sum())
        if count == 0:
            raise InputValidationError("MISSING_ROBOT", f"Robot {robot_id} is missing.")
        if count < 2:
            raise InputValidationError(
                "MISSING_ROBOT", f"Robot {robot_id} must contain at least two rows."
            )
        robot_times = np.ascontiguousarray(time_s[mask], dtype=np.float64)
        differences = np.diff(robot_times)
        if np.any(differences == 0):
            raise InputValidationError(
                "DUPLICATE_TIMESTAMP", f"Robot {robot_id} contains a duplicate timestamp."
            )
        if np.any(differences < 0):
            raise InputValidationError(
                "NON_MONOTONIC_TIME", f"Robot {robot_id} timestamps are not increasing."
            )
        if robot_times[0] != 0.0:
            raise InputValidationError(
                "FIRST_TIME_NOT_ZERO", f"Robot {robot_id} first timestamp must be 0.0."
            )
        robot_modes = np.ascontiguousarray(modes[mask], dtype=np.uint8)
        if robot_modes[-1] != MODE_W:
            raise InputValidationError(
                "FINAL_MODE_NOT_W", f"Robot {robot_id} final mode must be W."
            )
        robots.append(
            RobotTrajectory(
                robot_id=robot_id,
                time_s=robot_times,
                xyz_mm=np.ascontiguousarray(xyz32[mask], dtype=np.float32),
                mode=robot_modes,
            )
        )
    return TrajectorySet(robots=(robots[0], robots[1], robots[2]), row_count=frame.height)
