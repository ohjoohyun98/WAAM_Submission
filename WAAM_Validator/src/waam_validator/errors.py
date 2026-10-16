"""Stable error types and validation messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    """One deterministic warning or non-fatal validation violation."""

    code: str
    message: str
    severity: Literal["warning", "violation"]
    robot_id: int | None = None
    start_s: float | None = None
    end_s: float | None = None

    def display(self) -> str:
        """Return the stable human-readable form used in reports."""
        return f"{self.code} - {self.message}"


@dataclass(slots=True)
class ValidationMessages:
    """Collected non-fatal messages produced by validators."""

    warnings: list[ValidationIssue] = field(default_factory=list)
    violations: list[ValidationIssue] = field(default_factory=list)

    def warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(
            ValidationIssue(code=code, message=message, severity="warning", **context)
        )

    def violation(self, code: str, message: str, **context: Any) -> None:
        self.violations.append(
            ValidationIssue(code=code, message=message, severity="violation", **context)
        )

    def extend(self, other: ValidationMessages) -> None:
        self.warnings.extend(other.warnings)
        self.violations.extend(other.violations)


class WaamValidatorError(Exception):
    """Base exception carrying a stable code and CLI exit status."""

    exit_code = 4

    def __init__(self, code: str, message: str, *, output_dir: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.output_dir = output_dir


class InputValidationError(WaamValidatorError):
    """Required input or schema error."""

    exit_code = 2


class TargetValidationError(WaamValidatorError):
    """Target mesh load, validity, or coordinate error."""

    exit_code = 3


class ComputationError(WaamValidatorError):
    """Unexpected numerical or geometry computation error."""

    exit_code = 4


class OutputWriteError(WaamValidatorError):
    """Output directory or artifact write error."""

    exit_code = 5
