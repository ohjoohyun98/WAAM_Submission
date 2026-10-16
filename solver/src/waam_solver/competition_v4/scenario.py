from collections import OrderedDict,defaultdict
from time import perf_counter
import math
from shapely import normalize as canonical
from waam_validator.config.loader import load_config
from ..geometry import slice_layers,toolpaths
from ..shape_validation import aggregate,deposit_polygon
from ..tasks import generate_tasks
from ..competition_v3.scenario import Scenario
from .geometry import layer_refine,cost

LAST_AUDIT={}

def full_shape_metrics(scores,config,mesh_volume=None):
    """Official area/volume reduction, reusing already-buffered layer metrics.

    Buffering and polygon normalization happened in score_paths. Repeating the
    same polygon union here is not a more accurate full-volume evaluation.
    Differential tests compare all fields against compute_shape_metrics.
    """
    result=aggregate(scores,config);h=config['process']['layer_height_mm']
    for field,source in [('deposited_volume_mm3','deposited_area_mm2'),('intersection_volume_mm3','intersection_area_mm2'),
                         ('underfill_volume_mm3','residual_area_mm2'),('overfill_volume_mm3','overfill_area_mm2')]:
        result[field]=sum(s[source]*h for s in scores.values() if s['evaluated'])
    result['overfill_ratio']=result['overfill'];result['underfill_ratio']=result['underfill_volume_mm3']/result['target_volume_mm3']
    result['target_mesh_volume_mm3']=abs(mesh_volume) if mesh_volume is not None else result['target_volume_mm3']
    return result

def cold_scenario(source,memoize=True):
    started=perf_counter(); model=load_config(source/'config.yaml'); config=model.model_dump()
    layers,mesh=slice_layers(source/'target.stl',config,return_mesh=True)
    paths={};scores={};memo=OrderedDict();hits=0
    for k,target in layers.items():
        if target.is_empty: continue
        key=canonical(target.simplify(0.,preserve_topology=True)).wkb
        if key in memo: paths[k],scores[k]=memo[key];hits+=1
        else:
            paths[k],scores[k]=toolpaths(target,config);memo[key]=(paths[k],scores[k])
            if len(memo)>4:memo.popitem(last=False)
    initial=aggregate(scores,config);initial_length=sum(cost(p) for p in paths.values())
    audit={'initial':initial,'refinement':{},'initial_length_mm':initial_length}
    refinement_started=perf_counter(); memo.clear()
    refinement_budget=min(120.,max(12.,3*math.sqrt(len(layers))+sum(len(getattr(g,'geoms',[g])) for g in layers.values())**.5))
    audit['refinement_budget_s']=refinement_budget
    if not initial['passed']:
        for k in sorted(paths):
            # Volume gate first; a layer is refined only when it misses global
            # scalar criteria. The final failed-layer ratio is still global.
            if aggregate({k:scores[k]},config)['passed']:continue
            key=canonical(layers[k].simplify(0.,preserve_topology=True)).wkb
            if key in memo:
                paths[k],scores[k],detail=memo[key];hits+=1
                detail={'memo_reuse':True,'reason':detail['reason'],'added_length_mm':detail['added_length_mm']}
            else:
                if perf_counter()-refinement_started>=refinement_budget:
                    audit['budget_exhausted_before_layer']=k;break
                complexity=len(getattr(layers[k],'geoms',[layers[k]]))
                budget=min(20.,max(3.,2.+complexity*.04))
                paths[k],scores[k],detail=layer_refine(layers[k],paths[k],config,budget)
                memo[key]=(paths[k],scores[k],detail)
                if len(memo)>4:memo.popitem(last=False)
            audit['refinement'][k]=detail
            if aggregate(scores,config)['passed']:break
    audit['refinement_seconds']=perf_counter()-refinement_started
    audit['predicted']=aggregate(scores,config)
    audit['final_length_mm']=sum(cost(p) for p in paths.values())
    audit['added_length_mm']=audit['final_length_mm']-initial_length
    gate=full_shape_metrics(scores,config,float(mesh.volume))
    audit['full_shape_gate']=gate
    audit['layer_metrics']=[dict(s,layer_index=k,passed=s['layer_passed']) for k,s in scores.items()]
    audit['shape_margins']={'iou':gate['iou']-config['shape_validation']['minimum_overall_iou'],
        'coverage':gate['coverage']-config['shape_validation']['minimum_overall_coverage'],
        'overfill':config['shape_validation']['maximum_overall_overfill_ratio']-gate['overfill_ratio']}
    audit['deposition_volume_mm3']=gate['deposited_volume_mm3']
    audit['geometry_redundancy_ratio']=audit['final_length_mm']*config['process']['bead_width_mm']*config['process']['layer_height_mm']/gate['target_volume_mm3']
    LAST_AUDIT.clear();LAST_AUDIT.update(audit)
    if not gate['passed']:
        raise ValueError(f'FULL_SHAPE_GATE_FAILED: IoU={gate["iou"]:.8f}, coverage={gate["coverage"]:.8f}, overfill={gate["overfill_ratio"]:.8f}; no uncertified geometry sent to scheduler')
    tasks=generate_tasks(layers,paths,config,{k:s.get('residual_path_indices',[]) for k,s in scores.items()})
    if not tasks:raise ValueError('NO_DEPOSITION_TASKS')
    by_layer=defaultdict(list)
    for task in tasks:by_layer[task.layer].append(task)
    return Scenario(config,float(mesh.bounds[1,2]),tuple(tasks),tuple(sorted(by_layer)),layers,dict(by_layer),scores,perf_counter()-started,hits)
