"""Nominal deposition and target shape validation."""

from .deposition import build_deposited_layers
from .metrics import compute_shape_metrics
from .target import load_target_mesh, slice_target_layers

__all__ = [
    "build_deposited_layers",
    "compute_shape_metrics",
    "load_target_mesh",
    "slice_target_layers",
]
