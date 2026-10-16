"""Bounded repair driven by whole-volume I - q(T+O), never per-island PASS."""
from collections import defaultdict
import math
import numpy as np
from shapely import LineString, normalize, set_precision
from shapely.affinity import translate
from ..geometry import polygons
from ..shape_validation import deposit_polygon, measure, aggregate
from ..tasks import generate_tasks
from ..competition_v3.scenario import Scenario
from ..competition_final.geometry import micro_candidates, contours, cost


def volume_state(scores, config):
    h = config['process']['layer_height_mm']; q = config['shape_validation']['minimum_overall_iou']
    result = aggregate(scores,config)
    active=[s for s in scores.values() if s['evaluated']]
    target = sum(s['target_area_mm2'] for s in active)*h
    intersection = sum(s['intersection_area_mm2'] for s in active)*h
    overfill = sum(s['overfill_area_mm2'] for s in active)*h
    result.update(target_volume=target, intersection_volume=intersection, overfill_volume=overfill,
                  iou_slack_volume=intersection-q*(target+overfill))
    return result


def state_key(scores, config):
    s = volume_state(scores,config); t = config['shape_validation']
    violation = (max(0.,t['minimum_overall_coverage']-s['coverage'])+
                 max(0.,s['overfill']-t['maximum_overall_overfill_ratio'])+
                 max(0.,s['failed_layer_ratio']-t['maximum_failed_layer_ratio']))
    return (int(s['passed']),-violation,s['iou_slack_volume'])


def component_proposal(target, config, limit):
    """A local *proposal*, accepted only by recomputing the actual layer union."""
    b = config['process']['bead_width_mm']; q = config['shape_validation']['minimum_overall_iou']
    paths = contours(target,config,.99); score=measure(target,deposit_polygon(paths,config),config)
    count=0; accepted=[]

    def merit(s):
        return s['intersection_area_mm2']-q*s['overfill_area_mm2']

    def consider(candidate,kind):
        nonlocal paths,score,count
        if count>=limit or not candidate:
            return False
        if any(len(p)<2 or not np.isfinite(p).all() or np.any(np.linalg.norm(np.diff(np.asarray(p,dtype=np.float32),axis=0),axis=1)<=1e-6) for p in candidate):
            return False
        count+=1
        s=measure(target,deposit_polygon(candidate,config),config)
        if merit(s)>merit(score)+1e-8:
            accepted.append({'kind':kind,'delta_I_minus_qO':merit(s)-merit(score)})
            paths=[p.copy() for p in candidate];score=s
            return True
        return False

    for name,trial in micro_candidates(target,config):
        consider(trial,name)
    # Vertex short strokes and a small two-segment replacement family.
    for inset in np.linspace(.20,.65,19):
        pieces=polygons(target.buffer(-b*inset))
        centers=[v for p in pieces for v in np.asarray(p.simplify(b*.03).exterior.coords)[:-1]]
        if 1<=len(centers)<=12:
            consider([np.asarray([v-[b*.002,0],v+[b*.002,0]]) for v in centers], 'replace_with_short_strokes')
    rect=np.asarray(target.minimum_rotated_rectangle.exterior.coords)[:4]
    center=np.asarray(target.centroid.coords[0]);edges=np.roll(rect,-1,axis=0)-rect
    axis=edges[np.argmax(np.linalg.norm(edges,axis=1))];axis/=np.linalg.norm(axis)
    normal=np.asarray([-axis[1],axis[0]])
    extent=np.ptp((rect-center)@axis)
    for fraction in (.25,.5,.75):
        for offset in (.1,.25,.4):
            length=max(b*.004,extent-b*fraction)
            trial=[np.asarray([center+sign*offset*b*normal-axis*length/2,
                               center+sign*offset*b*normal+axis*length/2]) for sign in (-1,1)]
            consider(trial,'paired_replace_two_segments')
    # Fixed coarse-to-fine sweeps; improvement never resets an unbounded loop.
    for step in b*np.asarray([.12,.06,.03,.015,.0075,.00375,.001875,.0009375]):
        for _ in range(2):
            if count>=limit:break
            for dx,dy,sx,sy,theta in [(step,0,1,1,0),(-step,0,1,1,0),(0,step,1,1,0),(0,-step,1,1,0),
                    (0,0,1+step/b,1,0),(0,0,1-step/b,1,0),(0,0,1,1+step/b,0),(0,0,1,1-step/b,0),
                    (0,0,1,1,step/b*.25),(0,0,1,1,-step/b*.25)]:
                mat=np.asarray([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
                consider([((p-center)*[sx,sy])@mat.T+center+[dx,dy] for p in paths],'shift_scale_rotate')
            for i,p in enumerate(list(paths)):
                if count>=limit:break
                if len(p)>24:continue
                closed=np.linalg.norm(p[0]-p[-1])<1e-7
                for j in range(len(p)-int(closed)):
                    for coordinate in (0,1):
                        for sign in (-1,1):
                            if count>=limit:break
                            trial=[a.copy() for a in paths];trial[i][j,coordinate]+=sign*step
                            if closed and j==0:trial[i][-1]=trial[i][0]
                            consider(trial,'endpoint_trim_shift')
    return paths,{'evaluations':count,'moves':accepted,'proposal_iou':score['iou'],
                  'local_objective':'I-qO; no local overall-IoU constraint'}


def recover(prepared, config, initial, limit=900):
    layers,mesh=prepared
    paths={k:[p.copy() for p in initial.get(k,contours(g,config,.99))] for k,g in layers.items() if not g.is_empty}
    scores={k:measure(layers[k],deposit_polygon(p,config),config) for k,p in paths.items()}
    before=volume_state(scores,config);details=[];memo={};b=config['process']['bead_width_mm']
    if before['passed']:
        tasks=generate_tasks(layers,paths,config);by=defaultdict(list)
        for task in tasks:by[task.layer].append(task)
        scenario=Scenario(config,float(mesh.bounds[1,2]),tuple(tasks),tuple(sorted(by)),layers,dict(by),scores,0.)
        return scenario,{'before':before,'after':before,'reason':'FEASIBLE_INITIAL_PATHS_NO_REPAIR',
                         'components':[],'unique_proposal_count':0}
    # Repeated exact layers reuse proposals, never fictitious independent areas.
    layer_memo={}
    for k in sorted(paths, key=lambda k:(scores[k]['iou'],k)):
        target=layers[k];key=normalize(target).wkb
        if key in layer_memo:
            proposal=layer_memo[key]
        else:
            proposal=[]
            for part in sorted(polygons(target),key=lambda g:g.bounds):
                if part.area/b**2>=16:
                    proposal.extend(p for p in paths[k] if LineString(p).intersects(part))
                    continue
                x,y=part.bounds[:2];local=translate(part,xoff=-x,yoff=-y)
                # Approximate canonical key only transfers path suggestions.
                local_key=normalize(set_precision(local,b*1e-4)).wkb
                if local_key not in memo:
                    memo[local_key]=component_proposal(local,config,limit)
                ps,detail=memo[local_key]
                proposal.extend(p+np.asarray([x,y]) for p in ps)
                details.append({'layer':k,'component_area_over_bead2':part.area/b**2,**detail})
            layer_memo[key]=proposal
        trial=dict(scores);trial[k]=measure(target,deposit_polygon(proposal,config),config)
        previous=volume_state(scores,config);next_state=volume_state(trial,config)
        if state_key(trial,config)>state_key(scores,config):
            paths[k]=[p.copy() for p in proposal];scores=trial
            details.append({'layer':k,'accepted_full_union':True,
                'delta_intersection_volume':next_state['intersection_volume']-previous['intersection_volume'],
                'delta_overfill_volume':next_state['overfill_volume']-previous['overfill_volume'],
                'delta_global_iou_slack_volume':next_state['iou_slack_volume']-previous['iou_slack_volume']})
        if aggregate(scores,config)['passed']:break
    after=volume_state(scores,config)
    audit={'before':before,'after':after,'components':details,'unique_proposal_count':len(memo),
           'path_length_mm':sum(cost(p) for p in paths.values()),
           'reason':'RECOVERED' if after['passed'] else 'SEARCH_EXHAUSTED',
           'not_an_impossibility_certificate':True}
    if not after['passed']:return None,audit
    tasks=generate_tasks(layers,paths,config);by=defaultdict(list)
    for task in tasks:by[task.layer].append(task)
    scenario=Scenario(config,float(mesh.bounds[1,2]),tuple(tasks),tuple(sorted(by)),layers,dict(by),scores,0.)
    return scenario,audit
