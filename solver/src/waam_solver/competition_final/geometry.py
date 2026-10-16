"""Bounded bidirectional bead-footprint search, driven only by input geometry."""
from time import perf_counter
import math
import numpy as np
from shapely import LineString,Point,union_all,normalize as canonical
from shapely.affinity import translate
from shapely.ops import substring,linemerge
from ..geometry import polygons,toolpaths
from ..shape_validation import deposit_polygon,measure,aggregate
from ..competition_v4.geometry import contours,sweeps,additions,cost,layer_refine

def apply_move(paths,kind,index=0,candidate=None,amount=0.,other=None):
    """All nine path-set operations; invalid/empty subpaths are discarded."""
    result=list(paths)
    if kind=='ADD':result.append(candidate)
    elif kind=='REMOVE':result.pop(index)
    elif kind=='REPLACE':result[index]=candidate
    elif kind=='SHIFT':result[index]=result[index]+np.asarray(amount)
    elif kind=='OFFSET':
        g=LineString(result[index]).offset_curve(amount)
        from ..residual import lines
        result.pop(index);result.extend(np.asarray(x.coords) for x in lines(g))
    elif kind in ('TRIM_START','TRIM_END'):
        line=LineString(result[index]);g=substring(line,amount if kind=='TRIM_START' else 0.,line.length-amount if kind=='TRIM_END' else line.length)
        result.pop(index)
        if isinstance(g,LineString):result.append(np.asarray(g.coords))
    elif kind=='SPLIT':
        line=LineString(result.pop(index));at=line.length*.5
        result.extend(np.asarray(substring(line,a,b).coords) for a,b in ((0.,at),(at,line.length)))
    elif kind=='MERGE':
        # Never bridge a non-deposited gap implicitly.
        merged=linemerge([LineString(result[index]),LineString(result[other])])
        if not isinstance(merged,LineString):return result
        result=[p for i,p in enumerate(result) if i not in (index,other)]+[np.asarray(merged.coords)]
    else:raise ValueError(kind)
    return [p for p in result if len(p)>1 and LineString(p).length>1e-5]

def state(score):
    under=score['residual_area_mm2'];over=score['overfill_area_mm2'];a=score['target_area_mm2']
    if under<a*1e-4 and over>a*.01:return 'saturated_overfill'
    if over>under*2:return 'overfill_dominant'
    if under>over*2:return 'underfill_dominant'
    return 'mixed'

def micro_candidates(component,config):
    """Exact small candidate enumeration includes no deposition, no area dropping."""
    bead=config['process']['bead_width_mm'];rect=np.asarray(component.minimum_rotated_rectangle.exterior.coords)[:4]
    edges=np.roll(rect,-1,axis=0)-rect;axis=edges[np.argmax(np.linalg.norm(edges,axis=1))];axis/=np.linalg.norm(axis)
    center=np.asarray(component.centroid.coords[0]);normal=np.array([-axis[1],axis[0]])
    yield 'skip',[]
    for fraction in np.linspace(.25,1.15,19):yield 'micro-contour',contours(component,config,float(fraction))
    for direction in (axis,normal,(axis+normal)/2**.5,(axis-normal)/2**.5):
        extent=np.ptp((rect-center)@direction)
        for length in (bead*.002,max(bead*.002,extent-bead),max(bead*.002,extent-bead*.6)):
            for shift in (0.,-.15,.15):
                c=center+normal*bead*shift
                yield 'micro-segment',[np.array([c-direction*length/2,c+direction*length/2])]

def optimize_paths(target,initial,config,budget=3.,minimize=True):
    start=perf_counter();paths=list(initial);initial_length=cost(paths);trace=[];tested=0
    score=measure(target,deposit_polygon(paths,config),config);original=dict(score)
    thresholds=config['shape_validation'];floor=thresholds['minimum_overall_iou']
    # A 0.5 percentage point design margin; never relax configured criteria.
    robust=min(.995,floor+.005)
    def passed(s):return aggregate({0:s},config)['passed']
    def rank(s,p):
        if not minimize:return (int(passed(s)),s['iou'],s['coverage'],-(s['overfill'] or 0.),-cost(p))
        if passed(s) and s['iou']>=robust:return (2,-cost(p),-len(p),s['iou'])
        return (1 if passed(s) else 0,s['iou'],s['coverage'],-(s['overfill'] or 0.))
    def evaluate(candidate,method):
        nonlocal paths,score,tested
        trial=measure(target,deposit_polygon(candidate,config),config);tested+=1
        if rank(trial,candidate)>rank(score,paths):
            keys=['intersection_area_mm2','residual_area_mm2','overfill_area_mm2','iou']
            trace.append({'move':method,'state_before':state(score),'delta':{k:trial[k]-score[k] for k in keys},
                          'delta_union':(trial['deposited_area_mm2']-trial['intersection_area_mm2'])-(score['deposited_area_mm2']-score['intersection_area_mm2']),
                          'delta_length_mm':cost(candidate)-cost(paths),'iou':trial['iou']})
            paths=list(candidate);score=trial;return True
        return False
    components=polygons(target)
    if not passed(score) and len(components)>1:
        combined=[];memo={}
        for p in components:
            if perf_counter()-start>=budget:break
            x,y=p.bounds[:2];local=translate(p,xoff=-x,yoff=-y);key=canonical(local).wkb
            if key not in memo:
                seed=contours(local,config,.99)
                best,s,d=optimize_paths(local,seed,config,max(.1,min(1.,budget/4)),minimize=minimize)
                memo[key]=best
            combined.extend(q+np.array([x,y]) for q in memo[key])
        else:evaluate(combined,'componentwise replacement (including skip candidate)')
    if len(components)==1 and target.area/config['process']['bead_width_mm']**2<9 and not passed(score):
        for method,candidate in micro_candidates(target,config):
            if perf_counter()-start>=budget:break
            evaluate(candidate,method)
    if not passed(score) and state(score)=='underfill_dominant' and perf_counter()-start<budget:
        candidate,s,detail=layer_refine(target,paths,config,max(.1,(budget-(perf_counter()-start))*.65))
        evaluate(candidate,'ADD targeted underfill portfolio')
    # Marginal contribution: long paths first; retain PASS+margin for path minimization.
    for iteration in range(3):
        improved=False
        for index in sorted(range(len(paths)),key=lambda i:-LineString(paths[i]).length):
            if perf_counter()-start>=budget:break
            if evaluate(apply_move(paths,'REMOVE',index),'REMOVE'):improved=True;break
        if not improved:break
    if not passed(score) or not minimize:
        # Generic thin-band replacement: the two inset boundaries can be
        # redundant or overfill the same narrow region. Test one near-medial
        # loop per surviving component, with the full bead simulation as gate.
        width=2*target.area/max(target.length,1e-9)
        if width/config['process']['bead_width_mm']<3.:
            for fraction in (.45,.48,.495):
                if perf_counter()-start>=budget:break
                inset=target.buffer(-width*fraction,quad_segs=thresholds['polygon_buffer_resolution'])
                candidate=[np.asarray(LineString(p.exterior.coords).simplify(config['process']['bead_width_mm']*.005).coords) for p in polygons(inset)]
                evaluate(candidate,'REPLACE near-medial band '+str(fraction))
        # Continuous inset portfolio helps both overlap and coverage, not add-only.
        for fraction in (.9,1.,1.1,1.2,1.35,1.5,1.75,2.,2.5):
            if perf_counter()-start>=budget:break
            evaluate(contours(target,config,fraction),'REPLACE inset '+str(fraction))
        bead=config['process']['bead_width_mm']
        # Limit critical path candidates by their actual overfill contribution.
        over=deposit_polygon(paths,config).difference(target)
        critical=sorted(range(len(paths)),key=lambda i:-LineString(paths[i]).buffer(bead/2).intersection(over).area)[:8]
        for index in critical:
            if index>=len(paths) or perf_counter()-start>=budget:break
            line=LineString(paths[index]);center=np.asarray(line.centroid.coords[0]);toward=np.asarray(target.representative_point().coords[0])-center
            toward=toward/max(np.linalg.norm(toward),1e-9)*bead*.05
            for kind,amount in [('SHIFT',toward),('OFFSET',bead*.05),('OFFSET',-bead*.05),('TRIM_START',min(bead*.2,line.length*.2)),('TRIM_END',min(bead*.2,line.length*.2))]:
                if index>=len(paths) or perf_counter()-start>=budget:break
                evaluate(apply_move(paths,kind,index,amount=amount),kind)
        if perf_counter()-start<budget:
            # v4 portfolio only after deletion/replacement were considered.
            candidate,s,detail=layer_refine(target,paths,config,max(.1,budget-(perf_counter()-start)))
            evaluate(candidate,'ADD/medial/residual v4 portfolio')
    return paths,score,{'seconds':perf_counter()-start,'candidate_count':tested,'trace':trace,'initial_iou':original['iou'],
        'final_iou':score['iou'],'initial_state':state(original),'final_state':state(score),'added_length_mm':cost(paths)-initial_length,
        'reason':'feasible' if passed(score) else 'bounded_portfolio_exhausted'}
