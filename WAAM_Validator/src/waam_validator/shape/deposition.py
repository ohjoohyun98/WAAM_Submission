"""Build nominal layer polygons from D intervals."""

from __future__ import annotations

from collections import defaultdict

from shapely import LineString, union_all
from shapely.geometry.base import BaseGeometry

from ..config.models import Config
from ..constants import MODE_D
from ..errors import ComputationError
from ..models import TrajectorySet
from ..progress import StageProgressCallback
from .layer_index import determine_layer_index
from .polygon_utils import normalize_polygon


def build_deposited_layers(
    trajectories: TrajectorySet,
    config: Config,
    *,
    progress_callback: StageProgressCallback | None = None,
) -> dict[int, BaseGeometry]:
    """Convert D intervals to layer-wise buffered and unioned polygons."""
    pending: dict[int, list[BaseGeometry]] = defaultdict(list)
    partial_unions: dict[int, list[BaseGeometry]] = defaultdict(list)
    radius = config.process.bead_width_mm / 2.0
    chunk_size = max(1000, min(config.simulation.batch_size, 10_000))
    total_intervals = sum(len(item.time_s) - 1 for item in trajectories.robots)
    processed_intervals = 0
    last_reported = -1.0
    try:
        for trajectory in trajectories.robots:
            for index, mode_value in enumerate(trajectory.mode[:-1]):
                processed_intervals += 1
                fraction = processed_intervals / total_intervals
                if progress_callback is not None and (
                    fraction >= 1.0 or fraction - last_reported >= 0.01
                ):
                    progress_callback(fraction, processed_intervals, total_intervals)
                    last_reported = fraction
                if int(mode_value) != int(MODE_D):
                    continue
                start = trajectory.xyz_mm[index].astype(float)
                end = trajectory.xyz_mm[index + 1].astype(float)
                representative_z = float((start[2] + end[2]) / 2.0)
                layer_index = determine_layer_index(representative_z, config)
                line = LineString([(start[0], start[1]), (end[0], end[1])])
                pending[layer_index].append(
                    line.buffer(
                        radius,
                        quad_segs=config.shape_validation.polygon_buffer_resolution,
                        cap_style="round",
                        join_style="round",
                    )
                )
                if len(pending[layer_index]) >= chunk_size:
                    partial_unions[layer_index].append(union_all(pending[layer_index]))
                    pending[layer_index].clear()
        for layer_index, polygons in pending.items():
            if polygons:
                partial_unions[layer_index].append(union_all(polygons))
        return {
            layer_index: normalize_polygon(
                union_all(polygons),
                config.shape_validation.polygon_snap_tolerance_mm,
                config.shape_validation.area_epsilon_mm2,
            )
            for layer_index, polygons in sorted(partial_unions.items())
        }
    except ComputationError:
        raise
    except Exception as exc:
        raise ComputationError("POLYGON_OPERATION_FAILED", str(exc)) from exc
