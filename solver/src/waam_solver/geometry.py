"""Independent STL slicing and conservative offset-contour toolpaths."""
import math
import numpy as np
import trimesh
from shapely import (Polygon, MultiLineString, line_merge, get_parts,
                     set_precision, make_valid, union_all, LineString)
from .profiling import timed,stage,count
from .shape_validation import score_paths
from .residual import fill_residual


def polygons(g):
    if isinstance(g, Polygon):
        return [] if g.is_empty else [g]
    return [p for x in getattr(g, "geoms", []) for p in polygons(x)]


@timed('stl_slicing')
def slice_layers(path, config, *, return_mesh=False):
    with stage('stl_loading'):
        mesh = trimesh.load_mesh(path, process=True)
    count('stl_triangle_count',len(mesh.faces))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("Expected a nonempty triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Nonfinite mesh vertices")
    if config['validation']['require_watertight_target'] and not mesh.is_watertight:
        raise ValueError("Target is not watertight; supply a valid input mesh")
    process = config['process']
    h, base = process['layer_height_mm'], process['build_plane_z_mm']
    if mesh.bounds[0, 2] < base - config['collision']['geometry_epsilon_mm']:
        raise ValueError("Target lies below build plane")
    snap = config['shape_validation']['polygon_snap_tolerance_mm']
    layers = {}
    for k in range(max(0, math.ceil((mesh.bounds[1, 2] - base) / h))):
        z = base + (k + .5) * h
        result = Polygon()
        for offset in (0, min(1e-4, h*1e-6), -min(1e-4, h*1e-6)):
            segments = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z+offset])
            if not len(segments):
                continue
            with stage('polygon_preprocessing'):
                lines = line_merge(set_precision(MultiLineString(list(segments[:, :, :2])), snap))
                loops = []
                for line in get_parts(lines):
                    pts = np.asarray(line.coords)
                    if len(pts) >= 4 and np.linalg.norm(pts[0]-pts[-1]) <= snap*2:
                        loops.extend(polygons(make_valid(Polygon(pts))))
                for poly in sorted(loops, key=lambda p: (-p.area, p.bounds)):
                    result = result.symmetric_difference(poly)
            if not result.is_empty:
                break
        with stage('polygon_preprocessing'):
            layers[k] = set_precision(make_valid(result), snap)
    if not any(p.area > 0 for p in layers.values()):
        raise ValueError("No nonempty target sections")
    count('layer_count',len(layers))
    count('polygon_count',sum(len(polygons(g)) for g in layers.values()))
    return (layers,mesh) if return_mesh else layers


def shape_score(target, lines, config):
    r = config['process']['bead_width_mm']/2
    q = config['shape_validation']['polygon_buffer_resolution']
    snap = config['shape_validation']['polygon_snap_tolerance_mm']
    deposited = set_precision(union_all([LineString(x).buffer(r, quad_segs=q)
                                       for x in lines]), snap)
    intersection = deposited.intersection(target).area
    over = deposited.difference(target).area / target.area
    return {'coverage': intersection/target.area, 'overfill': over,
            'iou': intersection/target.union(deposited).area,
            'length_mm': sum(LineString(x).length for x in lines)}


@timed('base_contour_generation')
def toolpaths(target, config):
    """Concentric inset contours; deterministic small inset relaxation.

    The slight radius relaxation retains bead-width thin walls which would
    collapse under an exact half-width erosion. Two nearby passes may remain;
    they are deliberately retained until a future centerline optimizer.
    Every candidate is measured with the actual fixed-width bead model.
    """
    r = config['process']['bead_width_mm']/2
    thresholds = config['shape_validation']
    candidates = []
    for fraction in (0.99, 0.95, 0.85, 0.7, 0.5):
        lines = []
        depth = r*fraction
        while True:
            inset = target.buffer(-depth, quad_segs=thresholds['polygon_buffer_resolution'])
            if inset.is_empty:
                break
            for poly in sorted(polygons(inset), key=lambda p: p.bounds):
                for ring in [poly.exterior, *poly.interiors]:
                    simple = LineString(ring.coords).simplify(r*0.01)
                    pts = np.asarray(simple.coords)
                    if simple.length > 1e-5:
                        lines.append(pts)
            depth += 2*r*0.9
        if not lines:
            continue
        lines,fill_ids=fill_residual(target,lines,config)
        score=score_paths(target,lines,config)
        score.update(length_mm=sum(LineString(x).length for x in lines),path_count=len(lines),
                     residual_fill_used=bool(fill_ids),residual_path_indices=fill_ids)
        candidates.append((score['layer_passed'],score,lines))
        if score['layer_passed']:
            return lines, score
    if not candidates:
        lines,fill_ids=fill_residual(target,[],config)
        score=score_paths(target,lines,config)
        score.update(length_mm=sum(LineString(x).length for x in lines),path_count=len(lines),
                     residual_fill_used=bool(fill_ids),residual_path_indices=fill_ids)
        return lines,score
    best = max(candidates, key=lambda x: x[1]['iou'])
    # A failed individual layer can still satisfy the allowed failed-layer ratio.
    # The complete-volume gate belongs to the pipeline, not this function.
    return best[2],best[1]
