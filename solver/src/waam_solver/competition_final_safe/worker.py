"""Workers never share live mutable geometry, plans, or optimizer globals.

Pickles are internal same-invocation handoffs only, never loaded from scenario
inputs or previous experiment outputs. Each worker runs in a new interpreter.
"""
import copy, hashlib, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely import get_precision, set_precision
from .common import save, digest

def serialized_hash(value):
    # Hash values, not pickle memo/reference layout or ndarray ownership flags.
    # Serialization may legally change aliasing while preserving all semantics.
    import dataclasses
    h=hashlib.sha256()
    def put(x):
        if isinstance(x,np.ndarray):
            h.update(b'array:'+x.dtype.str.encode()+str(x.shape).encode()+b':')
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,np.generic):put(x.item())
        elif isinstance(x,dict):
            h.update(b'dict:'+str(len(x)).encode()+b':')
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):put(k);put(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(b'seq:'+str(len(x)).encode()+b':')
            for v in x:put(v)
        elif dataclasses.is_dataclass(x):
            h.update(type(x).__name__.encode()+b':')
            for f in dataclasses.fields(x):put(f.name);put(getattr(x,f.name))
        elif isinstance(x,(str,bytes,int,float,bool,type(None))):
            data=x if isinstance(x,bytes) else repr(x).encode()
            h.update(type(x).__name__.encode()+str(len(data)).encode()+b':'+data)
        else:raise TypeError(f'Unspecified semantic hash type: {type(x)}')
    put(value)
    return h.hexdigest()

def hashes(scenario, trajectory=None):
    return {'geometry': serialized_hash((scenario.config, {k:(g.wkb,float(get_precision(g))) for k,g in scenario.layer_geometry.items()}, scenario.scores)),
            'tasks': serialized_hash(scenario.tasks),
            'trajectory': serialized_hash(trajectory) if trajectory is not None else None}

def grids(layers):
    return {k:float(get_precision(g)) for k,g in layers.items()}

def restore_grids(layers, precision):
    # WKB/pickle/deepcopy discard GEOS precision metadata. Coordinates alone
    # are not a semantically complete representation of a static polygon.
    return {k:set_precision(g,precision[k],mode='pointwise') if precision[k] else g for k,g in layers.items()}

def clone_scenario(s):
    result=copy.deepcopy(s)
    result.layer_geometry=restore_grids(result.layer_geometry,grids(s.layer_geometry))
    for task in result.tasks:task.points.setflags(write=False)
    return result

def clone_static(value):
    layers,mesh=copy.deepcopy(value)
    return restore_grids(layers,grids(value[0])),mesh

def proxy(scenario):
    p = scenario.config['process']
    distance = sum(t.length for t in scenario.tasks)
    travel = 0.
    for tasks in scenario.by_layer.values():
        for a, b in zip(tasks, tasks[1:]):
            travel += min(float(np.linalg.norm(a.points[-1] - b.points[0])),
                          float(np.linalg.norm(a.points[-1] - b.points[-1])))
    arc = len(scenario.tasks) * (p['arc_on_time_s'] + p['arc_off_time_s'])
    deposition = distance / p['deposition_speed_mm_s']
    travel_s = travel / p['travel_speed_mm_s']
    return {'task_count': len(scenario.tasks), 'arc_events': 2 * len(scenario.tasks),
            'segment_count': sum(len(t.points)-1 for t in scenario.tasks),
            'fragmentation_per_layer': len(scenario.tasks)/max(1, len(scenario.layers)),
            'D_s': deposition, 'mandatory_arc_s': arc, 'travel_proxy_s': travel_s,
            'manufacturing_proxy_s': deposition + arc + travel_s,
            'scope': 'robot-sum ordering proxy, NOT exact parallel makespan or feasibility certificate'}

def write_bundle(path, data):
    data=dict(data)
    if 'scenario' in data:data['scenario_precision']=grids(data['scenario'].layer_geometry)
    if 'slice' in data:data['slice_precision']=grids(data['slice'][0])
    with Path(path).open('xb') as f:
        pickle.dump(data, f, protocol=5)

def load_bundle(path):
    with Path(path).open('rb') as f:
        data=pickle.load(f)
    if 'scenario' in data:
        data['scenario'].layer_geometry=restore_grids(data['scenario'].layer_geometry,data['scenario_precision'])
        for task in data['scenario'].tasks:task.points.setflags(write=False)
    if 'slice' in data:
        data['slice']=(restore_grids(data['slice'][0],data['slice_precision']),data['slice'][1])
    return data

def frozen_worker(source, output, mode, seed, start):
    import run_solver_final as frozen
    static = {}; captured = {}; latest_plan = {}
    old_slice = frozen.scenario.slice_layers
    old_cold = frozen.scenario.cold_scenario
    old_greedy = frozen.scheduling.greedy_plan
    old_optimize = frozen.scheduling.optimize
    old_assemble = frozen.base.assemble
    def slicing(*args, **kwargs):
        value = old_slice(*args, **kwargs)
        static['slice'] = value
        return value
    def cold(*args, **kwargs):
        s = old_cold(*args, **kwargs)
        captured['scenario'] = s
        return s
    def greedy(*args, **kwargs):
        plan = old_greedy(*args, **kwargs)
        latest_plan['plan'] = plan
        return plan
    def optimize(*args, **kwargs):
        value = old_optimize(*args, **kwargs)
        latest_plan['plan'] = value[0]
        return value
    def assemble(s, e, r):
        value = old_assemble(s, e, r)
        bundle = {'scenario': s, 'plan': latest_plan['plan'], 'results': r, 'trajectory': value}
        if 'baseline' not in captured:
            captured['baseline'] = bundle
        captured['final'] = bundle
        return value
    frozen.scenario.slice_layers = slicing
    frozen.scenario.cold_scenario = cold
    frozen.scheduling.greedy_plan = greedy
    frozen.scheduling.optimize = optimize
    frozen.base.assemble = assemble
    frozen.START = start
    sys.argv = ['run_solver_final.py', '--scenario', str(source), '--output', str(output),
                '--mode', mode, '--seed', str(seed), '--validate']
    rc = frozen.main()
    if static:
        write_bundle(output/'static.pickle', static)
    if 'final' in captured:
        from .common import read
        result = read(output/'solver_result.json')
        bundle = captured['final']
        if result['final']['makespan_s'] != bundle['trajectory']['makespan_s']:
            bundle = captured['baseline']
        assert abs(result['final']['makespan_s']-bundle['trajectory']['makespan_s']) < 1e-7
        write_bundle(output/'frozen_bundle.pickle', bundle)
        save(output/'isolation.json', {**hashes(bundle['scenario'], bundle['trajectory']),
             'csv_sha256': digest(output/'trajectory.csv'), 'proxy': proxy(bundle['scenario']),
             'micro_eligible':micro_activation(bundle['scenario'])})
    return rc

def micro_layer_eligible(scenario, k):
    from ..geometry import polygons
    b = scenario.config['process']['bead_width_mm']
    threshold = scenario.config['shape_validation']['minimum_overall_iou']
    target=scenario.layer_geometry[k]
    # Match the existing Plus micro optimizer's actual robust-IoU return gate,
    # rather than spending time on already-robust bead-scale components.
    return (scenario.scores[k]['iou'] < threshold + .005 and not target.is_empty
            and all(p.area/b**2 < 9 for p in polygons(target)))

def micro_activation(scenario):
    return any(micro_layer_eligible(scenario,k) for k in scenario.layers)

def micro_copy(scenario):
    from ..competition_final_plus.geometry import micro_paths
    from ..shape_validation import aggregate
    from ..geometry import polygons
    from ..tasks import generate_tasks
    from shapely import normalize
    s = clone_scenario(scenario); paths = {}; scores = {}; memo = {}
    begin = time.perf_counter(); b = s.config['process']['bead_width_mm']
    for k in s.layers:
        original = [t.points[:, :2].copy() for t in s.by_layer[k]]
        target = s.layer_geometry[k]
        if micro_layer_eligible(s,k) and time.perf_counter()-begin < 30.:
            key = normalize(target.simplify(0.,preserve_topology=True)).wkb
            if key not in memo:
                memo[key] = micro_paths(target, original, s.config, budget=8., minimize=False)
            q, score, _ = memo[key]
            paths[k] = [p.copy() for p in q]; scores[k] = copy.deepcopy(score)
        else:
            paths[k] = original; scores[k] = copy.deepcopy(s.scores[k])
    if not aggregate(scores, s.config)['passed']:
        raise ValueError('INDEPENDENT_MICRO_FULL_SHAPE_GATE_FAILED')
    tasks = generate_tasks(s.layer_geometry, paths, s.config)
    by = defaultdict(list)
    for t in tasks: by[t.layer].append(t)
    s.tasks = tuple(tasks); s.by_layer = dict(by); s.layers = tuple(sorted(by)); s.scores = scores
    s.geometry_seconds = time.perf_counter()-begin
    return s

def plus_worker(kind, source, output, mode, seed, frozen_dir, start):
    from ..competition_final_plus import cli as plus
    from ..competition_final import scheduling
    from .common import read
    bundle_path = frozen_dir/'frozen_bundle.pickle'
    # Avoid loading/hashing a large trajectory merely to reject Micro. This
    # metadata is created by candidate A in this invocation, not an input cache.
    if kind=='micro_rescue' and (frozen_dir/'isolation.json').exists() and not read(frozen_dir/'isolation.json')['micro_eligible']:
        output.mkdir(parents=True)
        save(output/'failure.json',{'status':'SKIPPED_MICRO_SCREEN','reason':'All bead-scale layers already exceed the existing Plus micro robust-IoU threshold'})
        return 0
    bundle = load_bundle(bundle_path) if bundle_path.exists() else None
    before = hashes(bundle['scenario'], bundle['trajectory']) if bundle else None
    if bundle:
        original=read(frozen_dir/'isolation.json')
        assert all(before[k]==original[k] for k in before), 'Serialization changed frozen candidate semantics'
    cold_original = plus.geometry.cold_scenario
    schedule_original = plus.schedule
    if kind == 'final_plus_core' and bundle is None:
        # B has identical geometry to A; repeating the known full-shape failure
        # is wasteful. This does not suppress the independent C rescue.
        output.mkdir(parents=True)
        save(output/'failure.json', {'status':'SKIPPED_IDENTICAL_FAILED_GEOMETRY', 'reason':'No feasible frozen geometry bundle'})
        return 2
    if kind == 'micro_rescue' and bundle and not micro_activation(bundle['scenario']):
        output.mkdir(parents=True)
        save(output/'failure.json', {'status':'SKIPPED_MICRO_SCREEN', 'reason':'No low-margin bead-scale geometry'})
        return 0
    # Immutable static slicing is reused only inside this invocation. Workers
    # unpickle independent objects, so even third-party mutation cannot touch A.
    static_path = frozen_dir/'static.pickle'
    if static_path.exists():
        static = load_bundle(static_path)['slice']
        def slicing(*args, **kwargs):
            value=clone_static(static)
            return value if kwargs.get('return_mesh') else value[0]
        plus.geometry.frozen.slice_layers = slicing
        from ..competition_v4 import scenario as v4
        v4.slice_layers = slicing
    def cold(*args, **kwargs):
        if kind == 'final_plus_core':
            s = clone_scenario(bundle['scenario'])
        elif bundle:
            s = micro_copy(bundle['scenario'])
        else:
            s = cold_original(*args, micro=True)
        output.mkdir(parents=True, exist_ok=True)
        save(output/'manufacturing_proxy.json', proxy(s))
        return s
    used = False
    def schedule(s, m, seed=11):
        nonlocal used
        if kind == 'final_plus_core' and not used:
            used = True
            e = scheduling.Evaluator(s, certified_timeline=True)
            return {'scenario':s, 'evaluator':e, 'plan':copy.deepcopy(bundle['plan']),
                    'results':copy.deepcopy(bundle['results']), 'trajectory':copy.deepcopy(bundle['trajectory']),
                    'search':{'reused_independent_frozen_schedule':True}}
        return schedule_original(s, m, seed)
    plus.geometry.cold_scenario = cold
    plus.schedule = schedule
    sys.argv = ['run_solver_final_plus.py', '--scenario', str(source), '--output', str(output),
                '--mode', mode, '--seed', str(seed), '--validate']
    if kind == 'final_plus_core': sys.argv.append('--no-micro')
    rc = plus.main(start)
    if bundle:
        after = hashes(bundle['scenario'], bundle['trajectory'])
        assert before == after, 'Candidate mutation crossed isolation boundary'
        save(output/'isolation.json', {'before':before, 'after':after, 'unchanged':before==after})
    return rc

def main(argv, start):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['frozen_final','final_plus_core','micro_rescue'])
    p.add_argument('--scenario', required=True); p.add_argument('--output', required=True)
    p.add_argument('--mode', required=True); p.add_argument('--seed', type=int, default=11)
    p.add_argument('--frozen-dir')
    a = p.parse_args(argv)
    source, output = Path(a.scenario), Path(a.output)
    if a.kind == 'frozen_final': return frozen_worker(source, output, a.mode, a.seed, start)
    return plus_worker(a.kind, source, output, a.mode, a.seed, Path(a.frozen_dir), start)
