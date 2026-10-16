"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ..errors import InputValidationError
from .models import Config


def load_config(path: Path) -> Config:
    """Load YAML, validate the fixed schema, and return immutable config."""
    if not path.is_file():
        raise InputValidationError("MISSING_CONFIG", "config.yaml is required.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError("INVALID_CONFIG_SCHEMA", str(exc)) from exc
    if not isinstance(data, dict):
        raise InputValidationError(
            "INVALID_CONFIG_SCHEMA", "config.yaml must contain one mapping object."
        )
    if "schema_version" in data:
        raise InputValidationError(
            "INVALID_CONFIG_SCHEMA",
            "schema_version is not a Config field. Remove it; WAAM Validator versions "
            "the application independently from config.yaml.",
        )
    if "output" in data:
        raise InputValidationError(
            "INVALID_CONFIG_SCHEMA",
            "The config.output section was removed. Core result files are now always "
            "written, while deposited.stl and replay.html are generated on demand "
            "from the WAAM Validator result screen. Remove the entire output block.",
        )
    collision = data.get("collision")
    if isinstance(collision, dict):
        if "check_arm_crossing" in collision:
            raise InputValidationError(
                "INVALID_CONFIG_SCHEMA",
                "check_arm_crossing is no longer supported. Use required fields "
                "collision.check_arm_envelope and collision.arm_clearance_mm for "
                "the 2D Capsule collision model.",
            )
        missing_collision = [
            name for name in ("check_arm_envelope", "arm_clearance_mm") if name not in collision
        ]
        if missing_collision:
            raise InputValidationError(
                "INVALID_CONFIG_SCHEMA",
                "2D Capsule collision configuration is incomplete; missing: "
                + ", ".join(f"collision.{name}" for name in missing_collision),
            )
    robots = data.get("robots")
    if isinstance(robots, list):
        legacy_reach_ids = [
            str(robot.get("id", "?"))
            for robot in robots
            if isinstance(robot, dict) and "reach_radius_mm" in robot
        ]
        if legacy_reach_ids:
            raise InputValidationError(
                "INVALID_CONFIG_SCHEMA",
                "reach_radius_mm is no longer supported because Reach is now measured "
                "only in the World XY plane. Rename it to positive "
                "xy_reach_radius_mm on robot ID(s): "
                + ", ".join(legacy_reach_ids)
                + ".",
            )
        missing_robot_ids = [
            str(robot.get("id", "?"))
            for robot in robots
            if isinstance(robot, dict) and "arm_envelope_radius_mm" not in robot
        ]
        if missing_robot_ids:
            raise InputValidationError(
                "INVALID_CONFIG_SCHEMA",
                "Every robot requires positive arm_envelope_radius_mm for the 2D "
                f"Capsule model; missing on robot ID(s): {', '.join(missing_robot_ids)}.",
            )
        missing_xy_reach_ids = [
            str(robot.get("id", "?"))
            for robot in robots
            if isinstance(robot, dict) and "xy_reach_radius_mm" not in robot
        ]
        if missing_xy_reach_ids:
            raise InputValidationError(
                "INVALID_CONFIG_SCHEMA",
                "Every robot requires positive xy_reach_radius_mm for the World XY "
                "Reach model; missing on robot ID(s): "
                + ", ".join(missing_xy_reach_ids)
                + ".",
            )
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise InputValidationError("INVALID_CONFIG_SCHEMA", str(exc)) from exc
