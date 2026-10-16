"""Exact four-way portfolio, with independent frozen fallback artifacts."""
import time,math,hashlib,traceback
from pathlib import Path
import run_solver_final as frozen
from ..competition_final import scheduling
from ..competition_v3.scheduling import assemble
from ..competition_v3.scenario import lower_bound
from ..competition_v3.metrics import measure,peak_memory_mb
from ..rl_v2.core import dump_plan
from ..shape_validation import aggregate
from ..exporter import export_csv
from . import geometry,conflicts
from .portfolio import layerwise

base=frozen.base
def schedule(scenario,mode,seed=11):
    e=scheduling.Evaluator(scenario,certified_timeline=True);p=scheduling.greedy_plan(scenario,e)
    r={k:e.layer(k,p[k]) for k in scenario.layers};m=sum(x['duration'] for x in r.values());lb=lower_bound(scenario)
    sample=sorted(set((scenario.layers[0],scenario.layers[len(scenario.layers)//2],scenario.layers[-1])))
    density=sum(e.conflict_graph(k).sample_density(6) for k in sample)/len(sample)
    gap=max(0.,m/max(lb['global_layer_precedence_s'],1.)-1.)
    complexity=(len(scenario.tasks)**.5+len(scenario.layers)**.5)*(1+min(gap,2))*(.5+.5*density)
    budget={'fast':0.,'balanced':min(30.,max(4.,complexity*.25)),'best':min(120.,max(15.,complexity))}[mode]
    budget*=min(1.,(60/max(60.,scenario.geometry_seconds))**.5)
    info={'optimization_seconds':0.}
    if budget:p,r,info=scheduling.optimize(e,p,r,algorithm='lns',budget=budget,seed=seed)
    return {'scenario':scenario,'evaluator':e,'plan':p,'results':r,'search':info,'trajectory':assemble(scenario,e,r)}

def main(start):
    parser=base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['fast','balanced','best'],default='balanced')
    parser.add_argument('--validate',action='store_true');parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--no-micro',action='store_true');parser.add_argument('--no-pruning',action='store_true')
    parser.add_argument('--no-conflicts',action='store_true');args=parser.parse_args()
    source=Path(args.scenario).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(f'Refusing overwrite: {output}')
    for name in ('config.yaml','target.stl'):
        if not (source/name).is_file():raise FileNotFoundError(source/name)
    output.mkdir(parents=True)
    for name in ('config.yaml','target.stl'):base.shutil.copy2(source/name,output/name)
    try:
        scenario=geometry.cold_scenario(source,micro=not args.no_micro)
    except (ValueError,base.InputValidationError) as exc:
        base.save(output/'geometry_final_plus.json',geometry.AUDIT)
        base.save(output/'failure.json',{'status':'CONTROLLED_INPUT_OR_GEOMETRY_FAILURE','reason':str(exc),'trajectory_available':False,'cold_elapsed_s':time.perf_counter()-start})
        return 2
    print('Geometry ready',len(scenario.tasks),len(scenario.layers),flush=True)
    incumbent=schedule(scenario,args.mode,args.seed);candidates={'frozen_or_micro_recovered':incumbent};failures=[];pruning={}
    base_dir=output/'baseline';base_dir.mkdir()
    for name in ('config.yaml','target.stl'):base.shutil.copy2(source/name,base_dir/name)
    export_csv(incumbent['trajectory']['rows'],base_dir/'trajectory.csv')
    print('Frozen candidate',incumbent['trajectory']['makespan_s'],flush=True)
    if args.mode!='fast' and not args.no_pruning:
        try:
            variant,pruning=geometry.minimized(scenario,20. if args.mode=='balanced' else 35.)
            if variant:
                # Rebuild/replay only modified layers. Unchanged layers keep
                # frozen results; global barriers make the delta additive.
                active=set(pruning['changed_layers']);variant.layers=tuple(k for k in variant.layers if k in active)
                variant.tasks=tuple(t for t in variant.tasks if t.layer in active)
                variant.by_layer={k:variant.by_layer[k] for k in variant.layers};variant.scores={k:variant.scores[k] for k in variant.layers}
                candidate=schedule(variant,'fast',args.seed)
                mixed,layers=layerwise(incumbent,candidate)
                if layers:candidates['geometry']=mixed
                pruning['exact_selected_layers']=layers
                pruning['affected_baseline_makespan_s']=sum(incumbent['results'][k]['duration'] for k in active)
                pruning['affected_new_makespan_s']=candidate['trajectory']['makespan_s']
                print('Layerwise selected',len(layers),mixed['trajectory']['makespan_s'],flush=True)
        except Exception as exc:failures.append({'phase':'pruning','error':str(exc),'traceback':traceback.format_exc()})
    if args.mode!='fast' and not args.no_conflicts:
        for label in list(candidates):
            candidate=candidates[label];r=candidate['results'];share=sum(x['collision'] for x in r.values())/max(1.,candidate['trajectory']['makespan_s'])
            if share<.08:continue
            budget=(6. if args.mode=='balanced' else 25.)*min(1.5,.5+share*2)
            try:
                p,r,d=conflicts.optimize(candidate['evaluator'],candidate['plan'],r,budget=budget,width=2 if args.mode=='balanced' else 3,depth=2 if args.mode=='balanced' else 3)
                candidates['combined' if label.startswith('geometry') else 'conflict']={**candidate,'plan':p,'results':r,'search':d,
                    'trajectory':assemble(candidate['scenario'],candidate['evaluator'],r)}
            except Exception as exc:failures.append({'phase':'conflicts','error':str(exc),'traceback':traceback.format_exc()})
    selected=min(candidates,key=lambda x:candidates[x]['trajectory']['makespan_s']);winner=candidates[selected]
    export_csv(winner['trajectory']['rows'],output/'trajectory.csv')
    chosen=winner['scenario'];result={'status':'GENERATED_NOT_OFFICIALLY_VALIDATED','mode':args.mode,'source':str(source),'output':str(output),
        'selected':selected,'tasks':len(chosen.tasks),'layers':len(chosen.layers),'final':measure(winner['trajectory'],chosen),
        'baseline':measure(incumbent['trajectory'],scenario),'plan':dump_plan(winner['plan']),'geometry_scores':chosen.scores,
        'shape_prevalidation':aggregate(chosen.scores,chosen.config),'fallback':failures,
        'portfolio':{label:{'makespan_s':c['trajectory']['makespan_s'],'tasks':len(c['scenario'].tasks),'search':c['search']} for label,c in candidates.items()},
        'runtime':{'cold_total_solver_s':time.perf_counter()-start,'geometry_s':scenario.geometry_seconds,'peak_working_set_mb':peak_memory_mb()},
        'input_policy':'Only supplied config/STL. No model IDs, stored plans, persisted geometry lookup, or neural network.'}
    base.save(output/'geometry_final_plus.json',{'initial':geometry.AUDIT,'pruning':pruning})
    base.save(output/'execution_source_sha256.json',{str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [base.ROOT/'run_solver_final_plus.py',*sorted((base.ROOT/'src').rglob('*.py'))]})
    base.save(output/'solver_result.json',result)
    if args.validate:
        official=frozen.validate_with_history(output);seconds=official['wall_s']
        if not official['passed'] and winner is not incumbent:
            reference=frozen.validate_with_history(base_dir);seconds+=reference['wall_s'];result['baseline_official']=reference
            if reference['passed']:
                base.shutil.copy2(output/'trajectory.csv',output/'rejected_trajectory.csv');base.shutil.copy2(base_dir/'trajectory.csv',output/'trajectory.csv')
                official=frozen.validate_with_history(output);seconds+=official['wall_s']
                result['selected']='official_frozen_fallback';result['final']=result['baseline'];result['tasks']=len(scenario.tasks)
                result['plan']=dump_plan(incumbent['plan']);result['geometry_scores']=scenario.scores
                result['shape_prevalidation']=aggregate(scenario.scores,scenario.config)
        result['official']=official;result['status']='PASS' if official['passed'] else 'FAIL';result['runtime']['validator_external_wall_s']=seconds
    base.save(output/'solver_result.json',result)
    print(result['status'],result['final']['makespan_s'],result['runtime']['cold_total_solver_s'],str(output),flush=True)
    return 0 if result['status']!='FAIL' else 2
