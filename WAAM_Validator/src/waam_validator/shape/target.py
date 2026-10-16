"""Target STL loading, validation, and horizontal slicing."""

from __future__ import annotations

import math
from collections.abc import Collection
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
from shapely import LineString, MultiLineString, Polygon, get_parts, line_merge, set_precision
from shapely.geometry.base import BaseGeometry

from ..config.models import Config
from ..constants import EXTREME_COORDINATE_WARNING_MM, MODE_D
from ..errors import TargetValidationError, ValidationMessages
from ..models import TrajectorySet
from ..progress import StageProgressCallback
from .mesh_components import repair_normals_and_count_bodies
from .polygon_utils import empty_polygon, normalize_polygon


def load_target_mesh(
    path: Path,
    config: Config,
    messages: ValidationMessages | None = None,
) -> trimesh.Trimesh:
    """Load and validate the mandatory target STL as one mesh."""
    if not path.is_file():
        raise TargetValidationError("MISSING_TARGET", "target.stl is required.")
    try:
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                geometry
                for geometry in loaded.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError("STL scene contains no triangle mesh")
            mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded.copy()
        else:
            raise ValueError(f"Unsupported mesh object: {type(loaded).__name__}")
    except Exception as exc:
        raise TargetValidationError("TARGET_LOAD_FAILED", str(exc)) from exc
    if mesh.is_empty or len(mesh.faces) == 0:
        raise TargetValidationError("TARGET_EMPTY", "target.stl contains no triangles.")
    if not np.isfinite(mesh.vertices).all():
        raise TargetValidationError("TARGET_NONFINITE", "target.stl contains non-finite vertices.")
    try:
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        body_count = repair_normals_and_count_bodies(mesh)
        if config.validation.attempt_target_repair and not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            mesh.remove_unreferenced_vertices()
            body_count = repair_normals_and_count_bodies(mesh)
            if messages is not None:
                messages.warning(
                    "TARGET_REPAIR_APPLIED", "A limited target mesh repair was attempted."
                )
    except Exception as exc:
        if config.validation.attempt_target_repair:
            raise TargetValidationError("TARGET_REPAIR_FAILED", str(exc)) from exc
        raise TargetValidationError("TARGET_LOAD_FAILED", str(exc)) from exc
    if mesh.is_empty or len(mesh.faces) == 0:
        raise TargetValidationError("TARGET_EMPTY", "No valid target triangles remain.")
    mesh.metadata["waam_body_count"] = body_count
    if config.validation.require_watertight_target and not mesh.is_watertight:
        raise TargetValidationError(
            "TARGET_NOT_WATERTIGHT", "target.stl must be watertight after validation."
        )
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise TargetValidationError("TARGET_NONFINITE", "target.stl bounds are non-finite.")
    if bounds[0, 2] < config.process.build_plane_z_mm - config.collision.geometry_epsilon_mm:
        raise TargetValidationError(
            "TARGET_TRAJECTORY_FRAME_MISMATCH",
            "target.stl extends below the configured build plane.",
        )
    if np.max(np.abs(bounds)) > EXTREME_COORDINATE_WARNING_MM and messages is not None:
        messages.warning("EXTREME_COORDINATE_WARNING", "target.stl has extreme coordinate bounds.")
    return mesh


def validate_coordinate_consistency(
    trajectories: TrajectorySet,
    mesh: trimesh.Trimesh,
    config: Config,
) -> None:
    """Reject a clearly disjoint deposition/target XY frame."""
    deposit_min = np.full(2, np.inf, dtype=np.float64)
    deposit_max = np.full(2, -np.inf, dtype=np.float64)
    found_deposition = False
    for trajectory in trajectories.robots:
        deposition_indices = np.flatnonzero(trajectory.mode[:-1] == MODE_D)
        if deposition_indices.size == 0:
            continue
        found_deposition = True
        xyz = np.asarray(trajectory.xyz_mm, dtype=np.float64)
        starts = xyz[deposition_indices, :2]
        ends = xyz[deposition_indices + 1, :2]
        deposit_min = np.minimum(deposit_min, np.minimum(starts.min(axis=0), ends.min(axis=0)))
        deposit_max = np.maximum(deposit_max, np.maximum(starts.max(axis=0), ends.max(axis=0)))
    if not found_deposition:
        return
    deposit_min -= config.process.bead_width_mm / 2.0
    deposit_max += config.process.bead_width_mm / 2.0
    target_min = np.asarray(mesh.bounds[0, :2], dtype=np.float64)
    target_max = np.asarray(mesh.bounds[1, :2], dtype=np.float64)
    overlaps = np.all(deposit_max >= target_min) and np.all(target_max >= deposit_min)
    if not overlaps:
        raise TargetValidationError(
            "TARGET_TRAJECTORY_FRAME_MISMATCH",
            "Deposition and target XY bounding boxes do not overlap.",
        )


def _slice_once(
    mesh: trimesh.Trimesh,
    z_mm: float,
    config: Config,
) -> BaseGeometry:
    loop_polygons = _horizontal_section_polygons(
        mesh,
        z_mm,
        config.shape_validation.polygon_snap_tolerance_mm,
        max(
            config.shape_validation.polygon_snap_tolerance_mm,
            config.collision.geometry_epsilon_mm,
        ),
    )
    if not loop_polygons:
        return empty_polygon()
    geometry = _even_odd_union(loop_polygons)
    return normalize_polygon(
        geometry,
        config.shape_validation.polygon_snap_tolerance_mm,
        config.shape_validation.area_epsilon_mm2,
    )


def _horizontal_section_polygons(
    mesh: trimesh.Trimesh,
    z_mm: float,
    snap_tolerance_mm: float,
    closure_tolerance_mm: float,
) -> list[BaseGeometry]:
    """Build closed XY loops directly from triangle-plane line segments.

    ``Trimesh.section`` turns segments into paths through optional SciPy graph
    traversal. Horizontal validation only needs closed XY loops, so Shapely can
    merge the raw segments without that heavyweight dependency.
    """
    lines = np.asarray(
        trimesh.intersections.mesh_plane(  # type: ignore[no-untyped-call]
            mesh,
            plane_origin=np.array([0.0, 0.0, z_mm]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        ),
        dtype=np.float64,
    )
    if lines.size == 0:
        return []
    segments = [segment[:, :2] for segment in lines if len(segment) == 2]
    if not segments:
        return []
    linework: BaseGeometry = MultiLineString(segments)
    if snap_tolerance_mm > 0:
        linework = set_precision(linework, snap_tolerance_mm)
    merged = line_merge(linework)
    loop_polygons: list[BaseGeometry] = []
    for part in get_parts(merged):
        if not isinstance(part, LineString):
            continue
        points = np.asarray(part.coords, dtype=np.float64)
        if len(points) < 3:
            continue
        if np.linalg.norm(points[0] - points[-1]) > closure_tolerance_mm:
            continue
        polygon = Polygon(points)
        if not polygon.is_empty:
            loop_polygons.append(polygon)
    return loop_polygons


def slice_target_mesh_at_z(
    mesh: trimesh.Trimesh,
    z_mm: float,
    snap_tolerance_mm: float,
) -> BaseGeometry:
    """Compatibility API for direct target slicing with a supplied snap tolerance."""
    polygons = _horizontal_section_polygons(
        mesh,
        z_mm,
        snap_tolerance_mm,
        max(snap_tolerance_mm, 1e-9),
    )
    if not polygons:
        return empty_polygon()
    return normalize_polygon(_even_odd_union(polygons), snap_tolerance_mm, 1e-12)


def _even_odd_union(polygons: list[BaseGeometry]) -> BaseGeometry:
    """Combine section loops with even-odd parity while preserving holes."""
    ordered = sorted(polygons, key=lambda item: (-item.area, item.bounds))
    geometry: BaseGeometry = empty_polygon()
    for polygon in ordered:
        geometry = geometry.symmetric_difference(polygon)
    return geometry


def slice_target_layers(
    mesh: trimesh.Trimesh,
    layer_indices: Collection[int],
    config: Config,
    messages: ValidationMessages | None = None,
    *,
    progress_callback: StageProgressCallback | None = None,
) -> dict[int, BaseGeometry]:
    """Slice target STL at each requested layer mid-plane with deterministic fallback."""
    result: dict[int, BaseGeometry] = {}
    epsilon_z = min(1e-4, config.process.layer_height_mm * 1e-6)
    bounds = mesh.bounds
    ordered_layers = sorted(layer_indices)
    total_layers = len(ordered_layers)
    for completed, layer_index in enumerate(ordered_layers, start=1):
        z_slice = (
            config.process.build_plane_z_mm + (layer_index + 0.5) * config.process.layer_height_mm
        )
        exact_error: Exception | None = None
        try:
            geometry = _slice_once(mesh, z_slice, config)
        except Exception as exc:
            exact_error = exc
            geometry = empty_polygon()
        used_offset: float | None = None
        if geometry.is_empty and bounds[0, 2] < z_slice < bounds[1, 2]:
            for offset in (epsilon_z, -epsilon_z):
                try:
                    fallback = _slice_once(mesh, z_slice + offset, config)
                except Exception:
                    continue
                if not fallback.is_empty:
                    geometry = fallback
                    used_offset = offset
                    break
        if exact_error is not None and geometry.is_empty:
            raise TargetValidationError("TARGET_SECTION_FAILED", str(exact_error)) from exact_error
        if used_offset is not None and messages is not None:
            messages.warning(
                "TARGET_SECTION_FALLBACK",
                f"Layer {layer_index} used slice offset {used_offset:+.6g} mm.",
            )
        result[layer_index] = geometry
        if progress_callback is not None:
            progress_callback(completed / total_layers, completed, total_layers)
    return result


def determine_evaluation_layers(
    mesh: trimesh.Trimesh,
    deposited_layers: dict[int, BaseGeometry],
    config: Config,
) -> list[int]:
    """Return all target candidate layers plus out-of-range deposited layers."""
    target_height = float(mesh.bounds[1, 2] - config.process.build_plane_z_mm)
    target_count = max(0, math.ceil(target_height / config.process.layer_height_mm))
    return sorted(set(range(target_count)) | set(deposited_layers))
