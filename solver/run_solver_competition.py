"""Unknown-job entry point. No model IDs, cached trajectories, or shape lookup."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONUTF8']='1'
import sys
sys.dont_write_bytecode=True
import time
START=time.perf_counter()
from rl_runtime import bootstrap
ROOT=bootstrap()
sys.path.insert(0,str(ROOT.parent/'WAAM_Validator/src'))
import argparse
import json
import shutil
import subprocess
import traceback
import hashlib
import math
from pathlib import Path
from datetime import datetime
from waam_solver.competition_v3.scenario import cold_scenario, lower_bound
from waam_solver.competition_v3.scheduling import Evaluator,greedy_plan,assemble,partition,support_schedule
from waam_solver.competition_v3.search import optimize
from waam_solver.competition_v3.metrics import measure,peak_memory_mb
from waam_solver.rl_v2.core import dump_plan
from waam_solver.exporter import export_csv
from waam_solver.shape_validation import aggregate
from waam_validator.errors import InputValidationError

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8'); temporary.replace(path)

def validate(job):
    exe=ROOT.parent/'WAAM_Validator/.venv/Scripts/waam-validator.exe'
    result={}; start=time.perf_counter()
    for command in ('check','run'):
        argv=[str(exe),command,str(job)]
        if command=='run': argv+=['--output',str(job/'validation')]
        begin=time.perf_counter()
        proc=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8',errors='replace',env=os.environ.copy())
        (job/f'validator_{command}.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
        result[command]={'returncode':proc.returncode,'status':'PASS' if proc.returncode==0 else 'FAIL','wall_s':time.perf_counter()-begin}
    summary=job/'validation/summary.json'
    if summary.exists(): result['summary']=json.loads(summary.read_text(encoding='utf-8'))
    result['wall_s']=time.perf_counter()-start
    result['passed']=result['check']['returncode']==0 and result['run']['returncode']==0 and result.get('summary',{}).get('status')=='PASS'
    return result

def solve(args):
    deployment=ROOT/'checkpoints/competition_v3/deployment.json'
    automatic_algorithm=args.algorithm=='auto'
    if automatic_algorithm: args.algorithm='portfolio'
    if args.engine=='auto' and deployment.exists() and args.algorithm!='stable':
        settings=json.loads(deployment.read_text(encoding='utf-8'))
        if automatic_algorithm: args.algorithm=settings.get('algorithm','portfolio')
        args.timeline=args.timeline or settings.get('timeline',False)
        args.certified_timeline=args.certified_timeline or settings.get('certified_timeline',False)
        args.geometry_memo=args.geometry_memo or settings.get('geometry_memo',False)
        if automatic_algorithm and not args.ranker and settings.get('ranker'): args.ranker=str(ROOT/settings['ranker'])
    elif args.engine=='timeline': args.timeline=True
    if args.algorithm in ('learned','ppo','hybrid') and not args.ranker:
        raise ValueError('Learned comparison requires --ranker; refusing a mislabeled neural result')
    source=Path(os.path.abspath(args.scenario)); output=Path(os.path.abspath(args.output)) if args.output else ROOT/'outputs/competition_v3'/datetime.now().strftime('job_%Y%m%d_%H%M%S_%f')
    if output.exists(): raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    # Validate mandatory input paths before creating a new result directory.
    for name in ('config.yaml','target.stl'):
        if not (source/name).is_file(): raise FileNotFoundError(source/name)
    output.mkdir(parents=True)
    for name in ('config.yaml','target.stl'): shutil.copy2(source/name,output/name)
    print(f'cold geometry: {source}',flush=True)
    try:
        scenario=cold_scenario(source,memoize=args.geometry_memo); lb=lower_bound(scenario)
    except (ValueError,InputValidationError) as exc:
        save(output/'failure.json',{'status':'CONTROLLED_INPUT_OR_GEOMETRY_FAILURE','reason':str(exc),
             'cold_elapsed_s':time.perf_counter()-START,'trajectory_available':False})
        print(f'Cannot construct feasible geometry: {exc}',flush=True)
        return 2
    shape_prevalidation=aggregate(scenario.scores,scenario.config)
    print(f'tasks={len(scenario.tasks)} layers={len(scenario.layers)} geometry={scenario.geometry_seconds:.3f}s',flush=True)
    begin=time.perf_counter()
    evaluator=Evaluator(scenario,local_height=not args.full_height and args.algorithm!='stable' and not args.local_support,
                        primitive_mm=args.primitive_mm,timeline=args.timeline,certified_timeline=args.certified_timeline)
    plan=greedy_plan(scenario,evaluator)
    if args.cluster_size: plan=partition(plan,args.cluster_size)
    results={}
    for index,layer in enumerate(scenario.layers):
        results[layer]=evaluator.layer(layer,plan[layer])
        if index%25==0: print(f'baseline layer {index+1}/{len(scenario.layers)}',flush=True)
    baseline=assemble(scenario,evaluator,results); scheduling=time.perf_counter()-begin
    baseline_exact=evaluator.exact_seconds
    baseline_dir=output/'baseline'; baseline_dir.mkdir()
    for name in ('config.yaml','target.stl'): shutil.copy2(source/name,baseline_dir/name)
    export_start=time.perf_counter(); export_csv(baseline['rows'],baseline_dir/'trajectory.csv')
    shutil.copy2(baseline_dir/'trajectory.csv',output/'trajectory.csv')
    export_seconds=time.perf_counter()-export_start
    save(output/'solver_progress.json',{'stage':'baseline_persisted','makespan_s':baseline['makespan_s'],
         'shape_prevalidation':shape_prevalidation,'officially_validated':False,'cold_elapsed_s':time.perf_counter()-START})
    print(f'baseline persisted: M={baseline["makespan_s"]:.6f}s',flush=True)
    final=baseline; search={'records':[],'neural_seconds':0.,'optimization_seconds':0.}; fallback=[]
    # Soft budget bounds the optimization loop, never truncates geometry or a
    # safety evaluation halfway through. A single exact evaluation may overrun.
    gap=max(0.,baseline['makespan_s']/max(lb['global_layer_precedence_s'],1.)-1.)
    sample_layers=sorted(set((scenario.layers[0],scenario.layers[len(scenario.layers)//2],scenario.layers[-1])))
    conflict_density=sum(evaluator.conflict_graph(k).sample_density(6) for k in sample_layers)/len(sample_layers)
    complexity=(len(scenario.tasks)**.5+len(scenario.layers)**.5)*(1+min(gap,2))*(.5+.5*conflict_density)
    adaptive={'fast':0.,'balanced':min(30.,max(4.,complexity*.25)), 'best':min(120.,max(15.,complexity))}[args.mode]
    budget=args.budget if args.budget is not None else adaptive
    if args.budget is None: budget*=min(1.,(60/max(60.,scenario.geometry_seconds))**.5)
    if not shape_prevalidation['passed']: budget=0.
    if args.algorithm in ('stable','greedy'): budget=0.
    ranker=None
    preflight_ood=None
    loading_begin=time.perf_counter()
    if args.ranker and budget>0:
        metadata_path=Path(args.ranker).with_name('training.json')
        if metadata_path.exists():
            training=json.loads(metadata_path.read_text(encoding='utf-8'))
            z=abs((math.log1p(len(scenario.layers))-training['mean'][19])/training['std'][19])
            if z>training['ood_limit']: preflight_ood={'reason':'layer-count feature outside training support','z':z,'limit':training['ood_limit']}
        if preflight_ood is None:
            from waam_solver.competition_v3.learning import Ranker
            ranker=Ranker.load(Path(args.ranker))
    neural_loading=time.perf_counter()-loading_begin
    try:
        if budget>0 and args.algorithm not in ('stable','greedy'):
            if args.algorithm=='teacher':
                search={'records':[],'neural_seconds':0.,'optimization_seconds':0.,'stages':[]}
                for stage in ('lns','sa','critical','reduced'):
                    plan,results,part=optimize(evaluator,plan,results,algorithm=stage,budget=budget/4,
                                              seed=args.seed,max_steps=args.steps)
                    for record in part['records']: record['context']=f'{stage}_{record["context"]}'
                    search['records'].extend(part.pop('records')); search['optimization_seconds']+=part['optimization_seconds']
                    search['stages'].append(part)
            else:
                plan,results,search=optimize(evaluator,plan,results,algorithm=args.algorithm,budget=budget,
                                            seed=args.seed,max_steps=args.steps,ranker=ranker)
            final=assemble(scenario,evaluator,results)
        if args.local_support:
            begin=time.perf_counter(); candidate=support_schedule(scenario,plan)
            search['support_seconds']=time.perf_counter()-begin
            search['support_candidate_makespan_s']=candidate['makespan_s']
            if candidate['makespan_s']<final['makespan_s']: final=candidate
    except Exception as exc:
        fallback.append({'phase':'optimization','error':repr(exc),'traceback':traceback.format_exc()}); final=baseline
    export_start=time.perf_counter()
    export_csv(final['rows'],output/'trajectory.next.csv'); (output/'trajectory.next.csv').replace(output/'trajectory.csv')
    export_seconds+=time.perf_counter()-export_start
    result={'source':str(source),'output':str(output),'mode':args.mode,'algorithm':args.algorithm,
            'input_policy':'config+STL only; no persistent scenario cache','seed':args.seed,
            'options':vars(args),'tasks':len(scenario.tasks),'layers':len(scenario.layers),
            'feasible_task_counts':{r['id']:sum(r['id'] in t.robots for t in scenario.tasks) for r in scenario.config['robots']},
            'baseline':measure(baseline,scenario),'final':measure(final,scenario),'lower_bound':lb,
            'plan':dump_plan(plan),'search':search,'fallback':fallback,'geometry_scores':scenario.scores,
            'shape_prevalidation':shape_prevalidation,'sampled_potential_conflict_density':conflict_density,
            'within_input_geometry_memo_hits':scenario.geometry_memo_hits,
            'runtime':{'cold_total_solver_s':time.perf_counter()-START,'geometry_s':scenario.geometry_seconds,
                       'baseline_scheduling_s':scheduling,'optimization_s':search['optimization_seconds'],
                       'neural_inference_s':search['neural_seconds'],'exact_evaluation_s':evaluator.exact_seconds,
                       'neural_loading_s':neural_loading,'baseline_exact_evaluation_s':baseline_exact,
                       'candidate_exact_evaluation_s':evaluator.exact_seconds-baseline_exact,
                       'local_support_s':search.get('support_seconds',0.),
                       'export_s':export_seconds,'peak_working_set_mb':peak_memory_mb(),
                       'timing_note':'exact_evaluation is nested in scheduling/optimization; do not sum overlapping categories'},
            'status':'GENERATED_NOT_OFFICIALLY_VALIDATED','soft_optimization_budget_s':budget}
    if ranker is not None:
        result['learned_routing']={'score_calls':ranker.calls,'ood_fallback_calls':ranker.ood_calls,'kind':ranker.kind}
    elif preflight_ood is not None: result['learned_routing']={'neural_loaded':False,'preflight_ood':preflight_ood}
    # Reproducibility metadata only. These digests never enter geometry, model
    # features, algorithm branching, assignment, or cache lookup.
    result['execution_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),*sorted((ROOT/'src/waam_solver/competition_v3').glob('*.py'))]}
    save(output/'solver_result.json',result)
    if args.validate:
        official=validate(output); result['official']=official
        validator_seconds=official['wall_s']
        if not official['passed'] and final is not baseline:
            baseline_official=validate(baseline_dir); result['baseline_official']=baseline_official
            validator_seconds+=baseline_official['wall_s']
            if baseline_official['passed']:
                shutil.copy2(output/'trajectory.csv',output/'rejected_trajectory.csv')
                shutil.copy2(baseline_dir/'trajectory.csv',output/'trajectory.csv')
                result['fallback'].append({'phase':'official','error':'advanced schedule failed; selected officially PASS baseline'})
                result['final']=result['baseline']; result['official']=validate(output)
                validator_seconds+=result['official']['wall_s']
        result['status']='PASS' if result['official']['passed'] else 'FAIL'
        result['runtime']['validator_external_wall_s']=validator_seconds
    save(output/'solver_result.json',result)
    print(json.dumps({'status':result['status'],'makespan_s':result['final']['makespan_s'],
                      'cold_solver_s':result['runtime']['cold_total_solver_s'],'output':str(output)},ensure_ascii=False),flush=True)
    return 0 if result['status']!='FAIL' else 2

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',required=True); p.add_argument('--output'); p.add_argument('--mode',choices=['fast','balanced','best'],default='balanced')
    p.add_argument('--algorithm',choices=['auto','stable','greedy','critical','local','sa','lns','reduced','learned','ppo','hybrid','portfolio','teacher'],default='auto')
    p.add_argument('--budget',type=float); p.add_argument('--steps',type=int); p.add_argument('--seed',type=int,default=11)
    p.add_argument('--primitive-mm',type=float,default=10.); p.add_argument('--cluster-size',type=int,default=0)
    p.add_argument('--full-height',action='store_true'); p.add_argument('--local-support',action='store_true')
    p.add_argument('--timeline',action='store_true')
    p.add_argument('--certified-timeline',action='store_true')
    p.add_argument('--engine',choices=['auto','reservations','timeline'],default='auto')
    p.add_argument('--geometry-memo',action='store_true')
    p.add_argument('--ranker'); p.add_argument('--validate',action='store_true')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.primitive_mm<=0 or args.cluster_size<0 or (args.budget is not None and args.budget<0): raise SystemExit('Invalid budget/granularity')
    raise SystemExit(solve(args))
