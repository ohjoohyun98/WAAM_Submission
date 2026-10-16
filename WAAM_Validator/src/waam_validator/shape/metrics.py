"""Layer and volume based nominal geometry metrics."""

from __future__ import annotations

from collections.abc import Mapping

from shapely.geometry.base import BaseGeometry

from ..config.models import Config
from ..errors import ComputationError, TargetValidationError
from ..models import LayerMetrics, ShapeMetrics
from ..progress import StageProgressCallback
from .layer_index import layer_bounds
from .polygon_utils import empty_polygon, normalize_polygon


def _unit_interval(value: float) -> float:
    """Clamp a mathematically unit-bounded ratio against geometry round-off."""
    return min(1.0, max(0.0, value))


def compute_shape_metrics(
    deposited_layers: Mapping[int, BaseGeometry],
    target_layers: Mapping[int, BaseGeometry],
    config: Config,
    *,
    target_mesh_volume_mm3: float | None = None,
    progress_callback: StageProgressCallback | None = None,
) -> tuple[ShapeMetrics, list[LayerMetrics]]:
    """Compute layer-wise and global nominal geometry metrics."""
    epsilon = config.shape_validation.area_epsilon_mm2
    height = config.process.layer_height_mm
    layer_metrics: list[LayerMetrics] = []
    totals = {"target": 0.0, "deposited": 0.0, "intersection": 0.0, "under": 0.0, "over": 0.0}
    failed_layers = 0
    evaluated_layers = 0
    try:
        ordered_layers = sorted(set(deposited_layers) | set(target_layers))
        total_layers = len(ordered_layers)
        for completed, layer_index in enumerate(ordered_layers, start=1):
            target = normalize_polygon(
                target_layers.get(layer_index, empty_polygon()),
                config.shape_validation.polygon_snap_tolerance_mm,
                epsilon,
            )
            deposited = normalize_polygon(
                deposited_layers.get(layer_index, empty_polygon()),
                config.shape_validation.polygon_snap_tolerance_mm,
                epsilon,
            )
            target_area = float(target.area)
            deposited_area = float(deposited.area)
            if target_area <= epsilon and deposited_area <= epsilon:
                if progress_callback is not None:
                    progress_callback(completed / total_layers, completed, total_layers)
                continue
            intersection_area = float(target.intersection(deposited).area)
            underfill_area = float(target.difference(deposited).area)
            overfill_area = float(deposited.difference(target).area)
            union_area = target_area + deposited_area - intersection_area
            if target_area <= epsilon:
                coverage = 0.0
                underfill_ratio = 0.0
                overfill_ratio = None
                iou = 0.0
                passed = False
            elif deposited_area <= epsilon:
                coverage = 0.0
                underfill_ratio = 1.0
                overfill_ratio = 0.0
                iou = 0.0
                passed = False
            else:
                coverage = _unit_interval(intersection_area / target_area)
                underfill_ratio = _unit_interval(underfill_area / target_area)
                overfill_ratio = overfill_area / target_area
                iou = _unit_interval(
                    intersection_area / union_area if union_area > epsilon else 1.0
                )
                passed = iou >= config.shape_validation.minimum_layer_iou
            evaluated_layers += 1
            failed_layers += int(not passed)
            bottom, top, z_slice = layer_bounds(layer_index, config)
            layer_metrics.append(
                LayerMetrics(
                    layer_index=layer_index,
                    z_bottom_mm=bottom,
                    z_top_mm=top,
                    z_slice_mm=z_slice,
                    target_area_mm2=target_area,
                    deposited_area_mm2=deposited_area,
                    intersection_area_mm2=intersection_area,
                    underfill_area_mm2=underfill_area,
                    overfill_area_mm2=overfill_area,
                    coverage=coverage,
                    underfill_ratio=underfill_ratio,
                    overfill_ratio=overfill_ratio,
                    iou=iou,
                    passed=passed,
                )
            )
            totals["target"] += target_area * height
            totals["deposited"] += deposited_area * height
            totals["intersection"] += intersection_area * height
            totals["under"] += underfill_area * height
            totals["over"] += overfill_area * height
            if progress_callback is not None:
                progress_callback(completed / total_layers, completed, total_layers)
    except Exception as exc:
        raise ComputationError("POLYGON_OPERATION_FAILED", str(exc)) from exc

    if totals["target"] <= epsilon * height:
        raise TargetValidationError(
            "TARGET_SECTION_FAILED", "Target slicing produced no non-empty evaluation layer."
        )
    coverage = _unit_interval(totals["intersection"] / totals["target"])
    underfill_ratio = _unit_interval(totals["under"] / totals["target"])
    overfill_ratio = totals["over"] / totals["target"]
    union_volume = totals["target"] + totals["deposited"] - totals["intersection"]
    iou = _unit_interval(totals["intersection"] / union_volume if union_volume > 0 else 1.0)
    failed_ratio = _unit_interval(failed_layers / evaluated_layers if evaluated_layers else 0.0)
    mesh_volume = (
        totals["target"] if target_mesh_volume_mm3 is None else abs(target_mesh_volume_mm3)
    )
    discrepancy = abs(mesh_volume - totals["target"]) / max(mesh_volume, epsilon * height)
    passed = (
        coverage >= config.shape_validation.minimum_overall_coverage
        and overfill_ratio <= config.shape_validation.maximum_overall_overfill_ratio
        and iou >= config.shape_validation.minimum_overall_iou
        and failed_ratio <= config.shape_validation.maximum_failed_layer_ratio
    )
    return (
        ShapeMetrics(
            target_volume_mm3=totals["target"],
            deposited_volume_mm3=totals["deposited"],
            intersection_volume_mm3=totals["intersection"],
            underfill_volume_mm3=totals["under"],
            overfill_volume_mm3=totals["over"],
            coverage=coverage,
            underfill_ratio=underfill_ratio,
            overfill_ratio=overfill_ratio,
            iou=iou,
            failed_layer_count=failed_layers,
            evaluated_layer_count=evaluated_layers,
            failed_layer_ratio=failed_ratio,
            target_mesh_volume_mm3=mesh_volume,
            target_volume_discrepancy_ratio=discrepancy,
            passed=passed,
        ),
        layer_metrics,
    )
