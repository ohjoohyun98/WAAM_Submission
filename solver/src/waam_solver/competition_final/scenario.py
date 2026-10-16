from collections import defaultdict,OrderedDict
from time import perf_counter
import numpy as np
from shapely import normalize as canonical
from waam_validator.config.loader import load_config
from ..geometry import slice_layers,toolpaths,polygons
from ..shape_validation import measure,deposit_polygon,aggregate
from ..tasks import generate_tasks
from ..competition_v3.scenario import Scenario
from ..competition_v4.scenario import full_shape_metrics
from .geometry import optimize_paths,contours,cost
LAST_AUDIT={}
MINIMIZE=True

def recover_with_v4(source,start,reason):
    from ..competition_v4.scenario import cold_scenario as fallback,LAST_AUDIT as old_audit
    try:reference=fallback(source)
    except ValueError as exc:
        LAST_AUDIT['v4_fallback_failed']=dict(old_audit)
        raise ValueError(f'{reason}; fresh v4 fallback also failed: {exc}') from exc
    LAST_AUDIT['fallback']='fresh v4 deterministic geometry';LAST_AUDIT['fallback_reason']=reason
    LAST_AUDIT['new_candidate_gate']=LAST_AUDIT['full_shape_gate']
    LAST_AUDIT['full_shape_gate']=old_audit['full_shape_gate'];LAST_AUDIT['final_length_mm']=old_audit['final_length_mm']
    LAST_AUDIT['shape_margins']=old_audit['shape_margins'];LAST_AUDIT['fallback_detail']=dict(old_audit)
    LAST_AUDIT['rejected_layer_metrics']=LAST_AUDIT['layer_metrics'];LAST_AUDIT['layer_metrics']=old_audit['layer_metrics']
    reference.geometry_seconds=perf_counter()-start
    return reference

def cold_scenario(source,memoize=True):
    LAST_AUDIT.clear();start=perf_counter();config=load_config(source/'config.yaml').model_dump()
    layers,mesh=slice_layers(source/'target.stl',config,return_mesh=True);sliced=perf_counter()
    paths={};scores={};details={};memo=OrderedDict();hits=0;optimization_s=0.;templates={}
    # First establish full-shape feasibility. A legal 0.85 layer need not
    # independently achieve the 0.90 overall threshold.
    for k,target in layers.items():
        if target.is_empty:continue
        key=canonical(target.simplify(0.,preserve_topology=True)).wkb
        if key in memo:paths[k],scores[k]=memo[key];hits+=1
        else:
            candidate=contours(target,config,.99);s=measure(target,deposit_polygon(candidate,config),config)
            if not s['layer_passed']:
                old,old_score=toolpaths(target,config)
                if old_score['iou']>s['iou']:candidate=old;s=old_score
            paths[k]=candidate;scores[k]=s;memo[key]=(candidate,s)
            if len(memo)>8:memo.popitem(last=False)
    initial=aggregate(scores,config);initial_length=sum(cost(p) for p in paths.values());memo.clear()
    for k,target in layers.items():
        if k not in paths:continue
        candidate=paths[k];s=scores[k];original=list(candidate)
        scalar_pass=aggregate({k:s},config)['passed'];global_pass=aggregate(scores,config)['passed']
        if (not MINIMIZE and global_pass) or (global_pass and not scalar_pass):continue
        key=canonical(target.simplify(0.,preserve_topology=True)).wkb
        if key in memo:
            candidate,s,detail=memo[key];hits+=1;detail={**detail,'memo_reuse':True}
        else:
            template_key=(len(candidate),len(polygons(target)),sum(len(p.interiors) for p in polygons(target)))
            reused=False
            if MINIMIZE and global_pass and template_key in templates:
                trial=[candidate[i] for i in templates[template_key]]
                trial_score=measure(target,deposit_polygon(trial,config),config)
                if aggregate({k:trial_score},config)['passed'] and trial_score['iou']>=min(.995,config['shape_validation']['minimum_overall_iou']+.005):
                    candidate=trial;s=trial_score;reused=True
            budget=3. if optimization_s<20. else 0.
            if not global_pass:budget=8. if optimization_s<100. else .1
            if reused:detail={'seconds':0.,'reason':'exactly_revalidated_path_subset','kept_indices':templates[template_key]}
            elif budget>0:
                candidate,s,detail=optimize_paths(target,candidate,config,budget,minimize=global_pass)
                optimization_s+=detail['seconds']
            else:detail={'seconds':0.,'reason':'minimization_budget_exhausted'}
            indices=[next((i for i,p in enumerate(original) if np.array_equal(p,q)),None) for q in candidate]
            if candidate and len(candidate)<len(original) and None not in indices and len(templates)<16:templates[template_key]=indices
            memo[key]=(candidate,s,detail)
            if len(memo)>8:memo.popitem(last=False)
        # A cost improvement must not turn a global PASS into FAIL.
        trial_scores=dict(scores);trial_scores[k]=s
        global_trial=aggregate(trial_scores,config)
        margin_floor=min(aggregate(scores,config)['iou'],config['shape_validation']['minimum_overall_iou']+.005)
        if not global_pass or (global_trial['passed'] and global_trial['iou']>=margin_floor-1e-8):
            paths[k]=candidate;scores[k]=s
        else:detail={**detail,'global_gate_rejected':True}
        details[k]=detail
    geometry_end=perf_counter();gate=full_shape_metrics(scores,config,float(mesh.volume))
    LAST_AUDIT.update(initial=initial,initial_length_mm=initial_length,final_length_mm=sum(cost(p) for p in paths.values()),
        refinement=details,full_shape_gate=gate,layer_metrics=[dict(s,layer_index=k) for k,s in scores.items()],
        optimization_s=optimization_s,memo_hits=hits,timing={'slicing_s':sliced-start,'paths_and_shape_s':geometry_end-sliced},
        shape_margins={'iou':gate['iou']-config['shape_validation']['minimum_overall_iou'],
        'coverage':gate['coverage']-config['shape_validation']['minimum_overall_coverage'],
        'overfill':config['shape_validation']['maximum_overall_overfill_ratio']-gate['overfill']})
    if not gate['passed']:
        return recover_with_v4(source,start,f'FULL_SHAPE_GATE_FAILED: {gate}')
    try:tasks=generate_tasks(layers,paths,config)
    except ValueError as exc:
        return recover_with_v4(source,start,f'NEW_PATH_REACH_FAILURE: {exc}')
    if not tasks:raise ValueError('NO_DEPOSITION_TASKS')
    by_layer=defaultdict(list)
    for task in tasks:by_layer[task.layer].append(task)
    LAST_AUDIT['timing']['task_generation_s']=perf_counter()-geometry_end
    return Scenario(config,float(mesh.bounds[1,2]),tuple(tasks),tuple(sorted(by_layer)),layers,dict(by_layer),scores,perf_counter()-start,hits)
