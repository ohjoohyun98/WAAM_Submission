"""Immutable, fingerprinted preprocessing cache shared by RL episodes."""
from dataclasses import dataclass
import hashlib
import json
import pickle
import os
from pathlib import Path
import time
import numpy as np
import yaml
from shapely import get_num_interior_rings

from ..geometry import slice_layers,toolpaths
from ..tasks import generate_tasks
from ..parallel_engine import ParallelExecutionEngine
from ..greedy_parallel_scheduler import GreedyParallelScheduler

CACHE_SCHEMA='waam-scenario-v3'


def canonical_path(value):
    # pathlib.resolve() leaves ``..`` unresolved on the current Windows host
    # when the working path contains non-ASCII characters.  abspath/normpath
    # provides one stable cache identity for relative and absolute spellings.
    return Path(os.path.normpath(os.path.abspath(os.fspath(value))))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ScenarioCache:
    schema_version: str
    source: str
    source_sha256: dict
    config: dict
    target_top_mm: float
    tasks: tuple
    task_static: np.ndarray
    layers: tuple
    layer_task_indices: dict
    layer_geometry: dict
    greedy_makespan_s: float
    greedy_proxy_makespan_s: float
    greedy_decisions: tuple
    greedy_layer_durations_s: dict
    greedy_layer_collision_wait_s: dict
    greedy_rows: dict
    greedy_waits: tuple
    build_seconds: float

    def tasks_for_layer(self,layer):
        return [self.tasks[i] for i in self.layer_task_indices[layer]]


def _geometry_features(target,radius):
    if target.is_empty:
        return (0.,)*5
    polys=[target] if target.geom_type=='Polygon' else list(getattr(target,'geoms',()))
    area=float(target.area); perimeter=float(target.length)
    holes=sum(int(get_num_interior_rings(p)) for p in polys)
    return area/(np.pi*radius*radius),perimeter/(2*np.pi*radius),len(polys)/10.,holes/10.,
    

def _static_features(tasks,layers,config,target_top):
    radius=float(config['workspace']['radius_mm'])
    base=float(config['process']['build_plane_z_mm'])
    height=max(target_top-base,config['process']['layer_height_mm'])
    speed=float(config['process']['deposition_speed_mm_s'])
    max_layer=max((t.layer for t in tasks),default=0)
    by_layer={k:[t for t in tasks if t.layer==k] for k in layers}
    result=[]
    for task in tasks:
        xy=task.points[:,:2]; start,end=xy[0],xy[-1]; center=xy.mean(axis=0)
        lo=xy.min(axis=0); hi=xy.max(axis=0); delta=end-start
        angle=np.arctan2(delta[1],delta[0])
        peers=by_layer[task.layer]
        centers=np.asarray([x.points[:,:2].mean(axis=0) for x in peers])
        local=float(np.sum(np.linalg.norm(centers-center,axis=1)<=max(50.,task.length*.5))-1)
        target=layers[task.layer]
        polys=[target] if target.geom_type=='Polygon' else list(getattr(target,'geoms',()))
        area=float(target.area); perimeter=float(target.length)
        holes=sum(len(p.interiors) for p in polys)
        geom=(area/(np.pi*radius*radius),perimeter/(2*np.pi*radius),len(polys)/10.,holes/10.)
        eligible=[float(r in task.robots) for r in (1,2,3)]
        result.append([
            task.layer/max(1,max_layer), (task.points[0,2]-base)/height,
            *(start/radius),*(end/radius),*(center/radius),*(lo/radius),*(hi/radius),
            task.length/(2*radius),np.sin(angle),np.cos(angle),
            (task.length/speed)/max(1.,2*radius/speed),float(task.residual_fill),
            float(np.linalg.norm(hi-lo))/(2*radius),*eligible,
            local/max(1,len(peers)-1),sum(x.length for x in peers)/(2*radius*max(1,len(peers))),*geom])
    data=np.asarray(result,dtype=np.float32)
    data.setflags(write=False)
    return data


def build_scenario(source):
    started=time.perf_counter(); source=canonical_path(source)
    hashes={n:digest(source/n) for n in ('config.yaml','target.stl')}
    config=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    layers,mesh=slice_layers(source/'target.stl',config,return_mesh=True)
    paths={}; scores={}
    for k,target in layers.items():
        if not target.is_empty:
            paths[k],scores[k]=toolpaths(target,config)
    tasks=generate_tasks(layers,paths,config,{k:s['residual_path_indices'] for k,s in scores.items()})
    top=float(mesh.bounds[1,2])
    engine=ParallelExecutionEngine(config,top)
    policy=GreedyParallelScheduler()
    layer_durations={}; collision={}
    for k in sorted(layers):
        ready=[t for t in tasks if t.layer==k]
        if not ready:
            continue
        release=engine.clock; before=len(engine.waits)
        policy.plan_layer(ready,engine,release)
        layer_durations[k]=engine.clock-release
        collision[k]=sum(w['duration_s'] for w in engine.waits[before:]
                         if w['reason']=='collision_reservation')
    greedy_rows=engine.finish()
    by_id={t.id:t for t in tasks}; decisions=list(engine.decisions); proxy=0.
    parking={r['id']:np.array([*r['base_xyz_mm'][:2],engine.travel_z]) for r in config['robots']}
    for k in sorted(layers):
        layer=[d for d in decisions if d['layer']==k]
        queues={r['id']:[] for r in config['robots']}
        for d in layer:
            queues[d['robot']].append((by_id[d['task']],d['reverse']))
        durations=[]
        for r,actions in queues.items():
            current=parking[r]; duration=0.
            for task,reverse in actions:
                pts=task.points[::-1] if reverse else task.points
                duration+=engine.travel_length(current,pts[0])/config['process']['travel_speed_mm_s']
                duration+=task.length/config['process']['deposition_speed_mm_s']
                duration+=(config['process'].get('arc_on_time_s') or 0)+(config['process'].get('arc_off_time_s') or 0)
                current=pts[-1]
            duration+=engine.travel_length(current,parking[r])/config['process']['travel_speed_mm_s']
            durations.append(duration)
        proxy+=max(durations,default=0.)+collision.get(k,0.)/max(1,len(config['robots']))
    indices={k:tuple(i for i,t in enumerate(tasks) if t.layer==k) for k in sorted(layers) if any(t.layer==k for t in tasks)}
    static=_static_features(tasks,layers,config,top)
    return ScenarioCache(CACHE_SCHEMA,str(source),hashes,config,top,tuple(tasks),static,
                         tuple(indices),indices,layers,float(engine.clock),float(proxy),tuple(engine.decisions),
                         layer_durations,collision,greedy_rows,tuple(engine.waits),time.perf_counter()-started)


def load_or_build_scenario(source,cache_dir,rebuild=False):
    source=canonical_path(source); cache_dir=canonical_path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256((str(source)+'|'+digest(source/'config.yaml')+'|'+digest(source/'target.stl')).encode()).hexdigest()[:20]
    path=cache_dir/f'{key}.pkl'
    if path.exists() and not rebuild:
        with path.open('rb') as f:
            cached=pickle.load(f)
        if cached.schema_version==CACHE_SCHEMA and cached.source_sha256=={n:digest(source/n) for n in ('config.yaml','target.stl')}:
            return cached,path,True
    cached=build_scenario(source)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        pickle.dump(cached,f,pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    metadata={'schema_version':CACHE_SCHEMA,'source':str(source),'source_sha256':cached.source_sha256,
              'tasks':len(cached.tasks),'layers':len(cached.layers),'greedy_makespan_s':cached.greedy_makespan_s,
              'greedy_proxy_makespan_s':cached.greedy_proxy_makespan_s,
              'build_seconds':cached.build_seconds,'cache_file':str(path)}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return cached,path,False
