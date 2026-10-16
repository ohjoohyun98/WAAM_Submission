"""High-margin only, bounded unique-coverage pruning and contour replacement."""
from time import perf_counter
import itertools,math
import numpy as np
from shapely import LineString,union_all
from ..geometry import polygons
from ..shape_validation import measure,deposit_polygon,aggregate
from ..competition_final.geometry import cost

def spaced_contours(target,config,start_fraction,pitch_fraction):
    bead=config['process']['bead_width_mm'];depth=bead*.5*start_fraction;result=[]
    limit=math.hypot(target.bounds[2]-target.bounds[0],target.bounds[3]-target.bounds[1])+bead
    while depth<limit:
        inset=target.buffer(-depth,quad_segs=config['shape_validation']['polygon_buffer_resolution'])
        if inset.is_empty:break
        for p in polygons(inset):
            for ring in [p.exterior,*p.interiors]:
                line=LineString(ring.coords).simplify(bead*.005)
                if line.length>1e-5:result.append(np.asarray(line.coords))
        depth+=bead*pitch_fraction
    return result

def prune(target,initial,config,budget=2.):
    start=perf_counter();limit=max(16,int(budget*32));paths=list(initial);score=measure(target,deposit_polygon(paths,config),config);old_length=cost(paths);trace=[];tested=0
    t=config['shape_validation'];margin=score['iou']-t['minimum_overall_iou']
    if margin<.03:return paths,score,{'reason':'low_margin_preserved','seconds':perf_counter()-start,'candidate_count':0}
    floor=min(score['iou'],t['minimum_overall_iou']+.015)
    def evaluate(candidate,kind):
        nonlocal paths,score,tested
        if not candidate or cost(candidate)>=cost(paths)-1e-7:return
        s=measure(target,deposit_polygon(candidate,config),config);tested+=1
        if aggregate({0:s},config)['passed'] and s['iou']>=floor and s['layer_passed']:
            trace.append({'kind':kind,'length_saved':cost(paths)-cost(candidate),'iou':s['iou']});paths=candidate;score=s
    # Evaluate bounded replacement families first; no all-path combinatorics.
    for pitch in (.98,1.,1.025,1.05):
        for inset in (.9,1.,1.1):
            if tested>=limit:break
            evaluate(spaced_contours(target,config,inset,pitch),'spacing_replacement')
    footprints=[deposit_polygon([p],config) for p in paths];efficiency=[]
    if len(paths)<=64:
        for i,p in enumerate(paths):
            if tested>=limit:break
            unique=footprints[i].intersection(target).difference(union_all(footprints[:i]+footprints[i+1:])).area
            efficiency.append((unique/max(LineString(p).length,1e-9),i))
        eligible=[i for _,i in sorted(efficiency)[:6]];base=list(paths)
        for count in (1,2,3):
            for indices in itertools.combinations(eligible,count):
                if tested>=limit:break
                evaluate([p for i,p in enumerate(base) if i not in indices],f'remove_{count}')
    return paths,score,{'seconds':perf_counter()-start,'candidate_count':tested,'trace':trace,'unique_coverage_per_mm':efficiency,
        'length_saved':old_length-cost(paths),'floor_iou':floor}
