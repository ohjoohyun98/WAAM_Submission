"""Input-local continuous micro recovery and margin-gated replacement portfolio."""
from time import perf_counter
from collections import defaultdict,OrderedDict
import numpy as np
from shapely import normalize,LineString,set_precision
from shapely.affinity import translate
from ..geometry import polygons
from ..shape_validation import measure,deposit_polygon,aggregate
from ..competition_final import scenario as frozen
from ..competition_final.geometry import optimize_paths as old_optimize,contours,cost
from ..competition_v3.scenario import Scenario
from ..tasks import generate_tasks
from ..models import Task
from .micro import refine
from .pruning import prune

AUDIT={}
def merge_contiguous(tasks):
    """Join only exactly touching polylines with a common feasible robot.

    No bridge, D speed change, or lost deposition segment is introduced.
    Reach-circle splits with disjoint feasible sets remain separate.
    """
    result=[];merges=0
    for task in tasks:
        if result and result[-1].layer==task.layer and np.array_equal(result[-1].points[-1],task.points[0]):
            previous=result[-1];robots=tuple(r for r in previous.robots if r in task.robots)
            if robots:
                result[-1]=Task(previous.id,task.layer,np.concatenate([previous.points,task.points[1:]]),robots);merges+=1;continue
        result.append(Task(len(result),task.layer,task.points,task.robots))
    return result,merges
def micro_paths(target,initial,config,budget=3.,minimize=True):
    start=perf_counter();bead=config['process']['bead_width_mm'];parts=polygons(target)
    tiny=all(x.area/bead**2<9 for x in parts)
    initial_score=measure(target,deposit_polygon(initial,config),config) if tiny else None
    limited=tiny and initial_score['iou']<config['shape_validation']['minimum_overall_iou']+.005
    p,s,d=old_optimize(target,initial,config,min(budget,.75) if limited else budget,minimize=minimize)
    if s['iou']>=config['shape_validation']['minimum_overall_iou']+.005 or not all(x.area/bead**2<9 for x in parts):return p,s,d
    combined=[];memo={};details=[]
    for part in parts:
        x,y=part.bounds[:2];local=translate(part,xoff=-x,yoff=-y)
        # Approximate key transfers a path proposal only; full actual layer is
        # remeasured below, including float32 export semantics. Not a PASS cache.
        key=normalize(set_precision(local,bead*1e-4)).wkb
        if key not in memo:
            seed=contours(local,config,.99);seed,_,_=old_optimize(local,seed,config,.5,minimize=False)
            q,t,e=refine(local,seed,config,1.6);memo[key]=(q,t,e)
        q,t,e=memo[key];combined.extend(a+np.array([x,y]) for a in q);details.append(e)
        if perf_counter()-start>max(24.,budget+16.) and len(details)<len(parts):return p,s,{**d,'seconds':perf_counter()-start,'micro_incomplete':True}
    trial=measure(target,deposit_polygon(combined,config),config)
    limits=config['shape_validation']
    if trial['iou']>s['iou']+1e-8 and trial['coverage']>=limits['minimum_overall_coverage'] and trial['overfill']<=limits['maximum_overall_overfill_ratio']:
        p,s=combined,trial;d={**d,'micro_accepted':True,'micro_components':details}
    if limited and not aggregate({0:s},config)['passed']:
        old,old_score,old_detail=old_optimize(target,initial,config,budget,minimize=minimize)
        if old_score['iou']>s['iou']:p,s,d=old,old_score,old_detail
    d['seconds']=perf_counter()-start
    return p,s,d

def cold_scenario(source,memoize=True,micro=True):
    AUDIT.clear();start=perf_counter();original=frozen.optimize_paths
    try:
        # Same frozen path portfolio; add recovery only when its actual
        # component/layer IoU is low. Avoid rerunning the identical failed
        # frozen pipeline before trying a micro candidate.
        if micro:frozen.optimize_paths=micro_paths
        result=frozen.cold_scenario(source,memoize=memoize)
        AUDIT['geometry']=dict(frozen.LAST_AUDIT)
        AUDIT['micro_recovery']=any(d.get('micro_accepted',False) for d in frozen.LAST_AUDIT.get('refinement',{}).values())
        result.geometry_seconds=perf_counter()-start
        return result
    except ValueError as exc:
        AUDIT['failure']=str(exc);AUDIT['geometry']=dict(frozen.LAST_AUDIT);raise
    finally:frozen.optimize_paths=original

def minimized(scenario,budget=20.):
    start=perf_counter();memo=OrderedDict();paths={};scores={};details={};changed=[]
    overall=aggregate(scenario.scores,scenario.config);threshold=scenario.config['shape_validation']['minimum_overall_iou']
    prune_enabled=overall['iou']-threshold>=.03
    for k in scenario.layers:
        target=scenario.layer_geometry[k];original=[t.points[:,:2] for t in scenario.by_layer[k]]
        key=normalize(target.simplify(0.,preserve_topology=True)).wkb
        if not prune_enabled:p,s,d=original,scenario.scores[k],{'reason':'low_overall_margin_geometry_preserved'}
        elif key in memo:
            p,s,d=memo[key]
        elif perf_counter()-start<budget:
            p,s,d=prune(target,original,scenario.config,min(6.,budget-(perf_counter()-start)))
            memo[key]=(p,s,d)
            if len(memo)>16:memo.popitem(last=False)
        else:p,s,d=original,scenario.scores[k],{'reason':'budget'}
        # Exact geometry key is intra-input only. No persisted/model lookup.
        paths[k]=p;scores[k]=s;details[k]=d
        if cost(p)<cost(original)-1e-6:changed.append(k)
    gate=aggregate(scores,scenario.config)
    if not gate['passed'] or gate['iou']<min(overall['iou'],threshold+.005)-1e-8:
        paths={k:[t.points[:,:2] for t in scenario.by_layer[k]] for k in scenario.layers};scores=dict(scenario.scores);changed=[]
    tasks,merges=merge_contiguous(generate_tasks(scenario.layer_geometry,paths,scenario.config));by=defaultdict(list)
    for t in tasks:by[t.layer].append(t)
    changed=sorted(set(changed)|{k for k in scenario.layers if len(by[k])!=len(scenario.by_layer[k])})
    if not changed:return None,{'changed_layers':[],'seconds':perf_counter()-start,'pruning_enabled':prune_enabled}
    result=Scenario(scenario.config,scenario.target_top_mm,tuple(tasks),tuple(sorted(by)),scenario.layer_geometry,dict(by),scores,perf_counter()-start)
    return result,{'changed_layers':changed,'seconds':perf_counter()-start,'details':details,'exact_contiguous_merges':merges,'pruning_enabled':prune_enabled,
        'before_length':sum(t.length for t in scenario.tasks),'after_length':sum(t.length for t in tasks)}
