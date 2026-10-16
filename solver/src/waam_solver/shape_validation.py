"""Nominal shape prevalidation with the official layer/overall semantics."""
import numpy as np
from shapely import Polygon, make_valid, set_precision, union_all, linestrings, buffer
from .profiling import timed


def parts(g):
    if isinstance(g,Polygon):
        return [] if g.is_empty else [g]
    return [p for x in getattr(g,'geoms',()) for p in parts(x)]


def normalize(g,config):
    t=config['shape_validation']
    g=set_precision(make_valid(g),t['polygon_snap_tolerance_mm'])
    return union_all([p for p in parts(g) if p.area>t.get('area_epsilon_mm2',1e-6)])


def deposit_polygon(paths,config):
    segments=[]
    for path in paths:
        # Validator consumes float32 TCP points, then buffers each D interval.
        xy=np.asarray(path,dtype=np.float32).astype(float)[:,:2]
        segments.extend(np.stack([xy[:-1],xy[1:]],axis=1))
    chunks=[]
    for i in range(0,len(segments),1000):
        lines=linestrings(np.asarray(segments[i:i+1000]))
        chunks.append(union_all(buffer(lines,config['process']['bead_width_mm']/2,
                                       quad_segs=config['shape_validation']['polygon_buffer_resolution'])))
    return normalize(union_all(chunks),config)


def measure(target,deposited,config):
    target=normalize(target,config); deposited=normalize(deposited,config)
    a,b=target.area,deposited.area
    eps=config['shape_validation'].get('area_epsilon_mm2',1e-6)
    intersection=target.intersection(deposited).area
    residual=target.difference(deposited).area
    over=deposited.difference(target).area
    union=a+b-intersection
    coverage=min(1.,max(0.,intersection/a)) if a>eps else 0.
    iou=min(1.,max(0.,intersection/union)) if a>eps and b>eps and union>eps else 0.
    return {'target_area_mm2':a,'deposited_area_mm2':b,'intersection_area_mm2':intersection,
            'residual_area_mm2':residual,'overfill_area_mm2':over,'coverage':coverage,
            'iou':iou,'overfill':over/a if a>eps else None,
            'evaluated':a>eps or b>eps,
            'layer_passed':a>eps and b>eps and iou>=config['shape_validation']['minimum_layer_iou']}


@timed('internal_validation')
def score_paths(target,paths,config):
    return measure(target,deposit_polygon(paths,config),config)


@timed('internal_validation')
def aggregate(scores,config):
    active=[s for s in scores.values() if s['evaluated']]
    h=config['process']['layer_height_mm']
    a=sum(s['target_area_mm2']*h for s in active)
    b=sum(s['deposited_area_mm2']*h for s in active)
    inter=sum(s['intersection_area_mm2']*h for s in active)
    over=sum(s['overfill_area_mm2']*h for s in active)
    if a<=config['shape_validation'].get('area_epsilon_mm2',1e-6)*h:
        raise ValueError('No nonempty target evaluation volume')
    failed=sum(not s['layer_passed'] for s in active)
    result={'coverage':min(1.,inter/a),'overfill':over/a,'iou':min(1.,inter/(a+b-inter)),
            'failed_layer_count':failed,'evaluated_layer_count':len(active),
            'failed_layer_ratio':failed/len(active),'target_volume_mm3':a}
    t=config['shape_validation']
    result['passed']=(result['coverage']>=t['minimum_overall_coverage'] and
                      result['overfill']<=t['maximum_overall_overfill_ratio'] and
                      result['iou']>=t['minimum_overall_iou'] and
                      result['failed_layer_ratio']<=t.get('maximum_failed_layer_ratio',0.05))
    return result
