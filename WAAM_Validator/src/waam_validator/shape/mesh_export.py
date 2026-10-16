"""Export nominal deposited layer polygons as an STL mesh."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import trimesh
from shapely.geometry.base import BaseGeometry

from ..config.models import Config
from ..errors import ComputationError, OutputWriteError
from .layer_index import layer_bounds
from .polygon_utils import polygon_components


def export_deposited_stl(
    deposited_layers: dict[int, BaseGeometry],
    config: Config,
    path: Path,
) -> None:
    """Extrude each polygon component and concatenate without boolean union."""
    meshes: list[trimesh.Trimesh] = []
    try:
        for layer_index, geometry in sorted(deposited_layers.items()):
            bottom, _, _ = layer_bounds(layer_index, config)
            for polygon in polygon_components(geometry):
                mesh = trimesh.creation.extrude_polygon(
                    polygon,
                    config.process.layer_height_mm,
                    engine="earcut",
                )
                mesh.apply_translation((0.0, 0.0, bottom))
                meshes.append(mesh)
        if meshes:
            combined = cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
            combined.merge_vertices()
            combined.export(path, file_type="stl")
        else:
            path.write_text("solid deposited\nendsolid deposited\n", encoding="ascii")
    except OSError as exc:
        raise OutputWriteError("OUTPUT_WRITE_FAILED", str(exc)) from exc
    except Exception as exc:
        raise ComputationError("DEPOSITION_EXTRUSION_FAILED", str(exc)) from exc
