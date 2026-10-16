"""Deterministic polygon normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable

from shapely import GeometryCollection, MultiPolygon, Polygon, make_valid, set_precision, union_all
from shapely.geometry.base import BaseGeometry


def empty_polygon() -> Polygon:
    return Polygon()


def polygon_components(geometry: BaseGeometry) -> list[Polygon]:
    """Extract polygon components recursively from any Shapely geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        components: list[Polygon] = []
        for child in geometry.geoms:
            components.extend(polygon_components(child))
        return components
    return []


def normalize_polygon(
    geometry: BaseGeometry,
    snap_tolerance_mm: float,
    area_epsilon_mm2: float,
) -> BaseGeometry:
    """Make valid, snap, filter tiny pieces, and return Polygon/MultiPolygon."""
    if geometry.is_empty:
        return empty_polygon()
    valid = make_valid(geometry) if not geometry.is_valid else geometry
    if snap_tolerance_mm > 0:
        valid = set_precision(valid, snap_tolerance_mm, mode="valid_output")
    pieces = [piece for piece in polygon_components(valid) if piece.area > area_epsilon_mm2]
    if not pieces:
        return empty_polygon()
    merged = union_all(pieces)
    if not merged.is_valid:
        merged = make_valid(merged)
    filtered = [piece for piece in polygon_components(merged) if piece.area > area_epsilon_mm2]
    if not filtered:
        return empty_polygon()
    if len(filtered) == 1:
        return filtered[0]
    return MultiPolygon(filtered)


def chunked_union(
    geometries: Iterable[BaseGeometry],
    *,
    chunk_size: int,
    snap_tolerance_mm: float,
    area_epsilon_mm2: float,
) -> BaseGeometry:
    """Union many polygons without retaining an unnecessary giant operation tree."""
    chunks: list[BaseGeometry] = []
    current: list[BaseGeometry] = []
    for geometry in geometries:
        current.append(geometry)
        if len(current) >= chunk_size:
            chunks.append(union_all(current))
            current.clear()
    if current:
        chunks.append(union_all(current))
    if not chunks:
        return empty_polygon()
    return normalize_polygon(union_all(chunks), snap_tolerance_mm, area_epsilon_mm2)
