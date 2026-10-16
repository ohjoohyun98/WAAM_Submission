"""Typed progress events emitted by the validation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ValidationProgress:
    stage: str
    message: str
    stage_progress: float
    overall_progress: float
    completed_units: float | None = None
    total_units: float | None = None
    unit: str | None = None


ProgressCallback = Callable[[ValidationProgress], None]
StageProgressCallback = Callable[[float, float, float], None]
