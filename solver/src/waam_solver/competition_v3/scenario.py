from dataclasses import dataclass
from collections import defaultdict
from collections import OrderedDict
from time import perf_counter
import numpy as np
from shapely import normalize
from waam_validator.config.loader import load_config
from ..geometry import slice_layers, toolpaths
from ..tasks import generate_tasks

@dataclass
class Scenario:
    config: dict
    target_top_mm: float
    tasks: tuple
    layers: tuple
    layer_geometry: dict
    by_layer: dict
    scores: dict
    geometry_seconds: float
    geometry_memo_hits: int=0

    def tasks_for_layer(self, layer):
        return self.by_layer[layer]

def cold_scenario(source,memoize=False):
    """Only the supplied config/STL are read. No scenario fingerprint/cache lookup."""
    start=perf_counter()
    config=load_config(source/'config.yaml').model_dump()
    layers,mesh=slice_layers(source/'target.stl',config,return_mesh=True)
    paths={}; scores={}; memo=OrderedDict(); hits=0
    for k,target in layers.items():
        if not target.is_empty:
            # Within-this-input memoization only. Exact canonical polygon
            # bytes (zero-tolerance collinear simplification), not an instance
            # fingerprint/feature or a persisted public-shape lookup.
            key=normalize(target.simplify(0.,preserve_topology=True)).wkb if memoize else None
            if memoize and key in memo:
                paths[k],scores[k]=memo[key]; memo.move_to_end(key)
                hits+=1
            else:
                paths[k],scores[k]=toolpaths(target,config)
                if memoize:
                    memo[key]=(paths[k],scores[k])
                    if len(memo)>4: memo.popitem(last=False)
    tasks=generate_tasks(layers,paths,config,{k:s['residual_path_indices'] for k,s in scores.items()})
    if not tasks: raise ValueError('NO_DEPOSITION_TASKS: no realizable toolpaths')
    by_layer=defaultdict(list)
    for task in tasks: by_layer[task.layer].append(task)
    return Scenario(config,float(mesh.bounds[1,2]),tuple(tasks),tuple(sorted(by_layer)),layers,
                    dict(by_layer),scores,perf_counter()-start,hits)

def lower_bound(scenario):
    """Relaxed, task-fixed bound, not a claim of physically attainable optimum.

    Hall-style subset workload bounds include D and mandatory arc; a minimum
    base-to-first-deposition/return travel bound is combined with max(), never
    falsely added to another robot's critical workload. Global-layer bound
    applies ONLY to barrier schedules. Local precedence bound is separate.
    """
    p=scenario.config['process']; robots=scenario.config['robots']
    ids=[r['id'] for r in robots]; arc=(p.get('arc_on_time_s') or 0)+(p.get('arc_off_time_s') or 0)
    costs={t.id:t.length/p['deposition_speed_mm_s']+arc for t in scenario.tasks}
    def subset(tasks):
        return max(sum(costs[t.id] for t in tasks if set(t.robots)<=set(s))/len(s)
                   for mask in range(1,1<<len(ids)) for s in [[r for i,r in enumerate(ids) if mask>>i&1]])
    workload=subset(scenario.tasks)
    travel=min(2*float(np.min(np.linalg.norm(t.points[:,:2]-np.asarray(r['base_xyz_mm'][:2]),axis=1)))
               /p['travel_speed_mm_s'] for t in scenario.tasks for r in robots if r['id'] in t.robots)
    longest=max(costs.values())
    barrier=sum(max(subset(scenario.by_layer[k]),max(costs[t.id] for t in scenario.by_layer[k])) for k in scenario.layers)
    return {'process_workload_s':workload,'minimum_first_return_xy_travel_s':travel,
            'longest_task_s':longest,'relaxed_task_fixed_s':max(workload,longest,travel),
            'global_layer_precedence_s':barrier,'mandatory_D_robot_s':sum(t.length for t in scenario.tasks)/p['deposition_speed_mm_s'],
            'mandatory_arc_robot_s':len(scenario.tasks)*arc,
            'scope':'fixed generated tasks, collision-free relaxation; not a manufacturing certificate'}
