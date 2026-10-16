"""Mapping deposition Z references to integer layer indices."""

from __future__ import annotations

from ..config.models import Config


def determine_layer_index(representative_z_mm: float, config: Config) -> int:
    """Return the nearest valid layer index and validate its expected Z."""
    process = config.process
    if process.tcp_z_reference == "top":
        raw_index = (representative_z_mm - process.build_plane_z_mm) / process.layer_height_mm
        layer_index = round(raw_index) - 1
        expected_z = process.build_plane_z_mm + (layer_index + 1) * process.layer_height_mm
    else:
        raw_index = (
            representative_z_mm - process.build_plane_z_mm - process.layer_height_mm / 2.0
        ) / process.layer_height_mm
        layer_index = round(raw_index)
        expected_z = process.build_plane_z_mm + (layer_index + 0.5) * process.layer_height_mm
    if layer_index < 0:
        raise ValueError(f"Deposition Z {representative_z_mm:.6g} mm maps below layer 0.")
    deviation = abs(representative_z_mm - expected_z)
    if deviation > config.validation.layer_z_tolerance_mm:
        raise ValueError(
            f"Deposition Z {representative_z_mm:.6g} mm differs from expected layer Z "
            f"{expected_z:.6g} mm by {deviation:.6g} mm."
        )
    return layer_index


def layer_bounds(layer_index: int, config: Config) -> tuple[float, float, float]:
    """Return bottom, top, and comparison slice Z for a layer."""
    bottom = config.process.build_plane_z_mm + layer_index * config.process.layer_height_mm
    top = bottom + config.process.layer_height_mm
    return bottom, top, (bottom + top) / 2.0
