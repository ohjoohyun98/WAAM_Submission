"""Geometry-driven longest-axis / narrow raster residual coverage."""
import math
import numpy as np
from shapely import LineString, Point
from .shape_validation import parts, deposit_polygon, measure
from .profiling import timed,count


def lines(g):
    if isinstance(g,LineString):
        return [g] if g.length>1e-6 else []
    return [p for child in getattr(g,'geoms',()) for p in lines(child)]


@timed('residual_fill_generation')
def fill_residual(target,paths,config):
    paths=list(paths)
    deposited=deposit_polygon(paths,config)
    score=measure(target,deposited,config)
    r=config['process']['bead_width_mm']/2
    useful=max(config['shape_validation'].get('area_epsilon_mm2',1e-6),r*r*.25,target.area*.005)
    residual=target.difference(deposited)
    # Small slivers remain diagnostic data; adding a full bead can be worse.
    islands=sorted((p for p in parts(residual) if p.area>=useful),key=lambda p:(-p.area,p.bounds))
    if not islands:
        return paths,[]
    allowed=target.buffer(-r,quad_segs=config['shape_validation']['polygon_buffer_resolution'])
    fill_ids=[]
    for island in islands:
        rect=np.asarray(island.minimum_rotated_rectangle.exterior.coords)[:-1]
        edges=np.roll(rect,-1,axis=0)-rect
        axis=edges[int(np.argmax(np.linalg.norm(edges,axis=1)))]
        axis=axis/np.linalg.norm(axis); normal=np.array([-axis[1],axis[0]])
        center=np.asarray(island.centroid.coords[0])
        along=(rect-center)@axis; across=(rect-center)@normal
        width=float(np.ptp(across)); length=float(np.ptp(along))
        n=max(1,math.ceil(width/(1.8*r)))
        for offset in np.linspace(across.min()+width/(2*n),across.max()-width/(2*n),n):
            base=center+offset*normal
            line=LineString([base-axis*(length+2*r),base+axis*(length+2*r)])
            # Keep the bead footprint in the target where erosion permits it.
            clipped=line.intersection(allowed).intersection(island.buffer(r))
            if clipped.is_empty and width<2*r:
                clipped=line.intersection(island)
            for candidate in lines(clipped):
                if len(fill_ids)>=64:
                    return paths,fill_ids
                trial=np.asarray(candidate.coords)
                addition=deposit_polygon([trial],config)
                updated=deposited.union(addition)
                new=measure(target,updated,config)
                gain=new['intersection_area_mm2']-score['intersection_area_mm2']
                count('residual_candidates')
                if (gain>=useful and new['iou']>score['iou']+1e-9 and
                    new['overfill']<=config['shape_validation']['maximum_overall_overfill_ratio']):
                    fill_ids.append(len(paths)); paths.append(trial)
                    deposited=updated; score=new
                    count('residual_paths_accepted')
    return paths,fill_ids
