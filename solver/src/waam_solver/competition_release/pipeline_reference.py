"""Literal original-A protection plus bounded-work Release improvements."""
import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from waam_validator.config.loader import load_config
from waam_validator.errors import InputValidationError
from ..geometry import slice_layers
from ..exporter import export_csv
from ..shape_validation import aggregate
from ..competition_v3.metrics import measure
from ..rl_v2.core import dump_plan
from . import core, micro, reference
from .common import Timers, digest, read, save, scenario_hash
from .settings import MODES, DEFAULT_MODE, VERSION


def official(job, directory):
    """Untouched installed Validator, never internal predictions labelled PASS."""
    result={};start=perf_counter()
    logs={}
    for action in ('check','run'):
        argv=[sys.executable,'-B','-m','waam_validator.cli',action,str(job)]
        if action=='run':argv+=['--output',str(directory)]
        begin=perf_counter()
        p=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8',errors='replace',env=os.environ.copy())
        log=p.stdout+'\n'+p.stderr
        (job/f'validator_{action}.log').write_text(log,encoding='utf-8')
        logs[action]=log
        result[action]={'returncode':p.returncode,'status':'PASS' if p.returncode==0 else 'FAIL','wall_s':perf_counter()-begin}
    # Validator owns creation of --output and refuses a pre-existing directory.
    # Preserve attempt logs only after both CLI commands have returned.
    directory.mkdir(parents=True,exist_ok=True)
    for action,log in logs.items():(directory/f'validator_{action}.log').write_text(log,encoding='utf-8')
    if (directory/'summary.json').exists():result['summary']=read(directory/'summary.json')
    result['wall_s']=perf_counter()-start
    result['passed']=all(result[a]['returncode']==0 for a in ('check','run')) and result.get('summary',{}).get('status')=='PASS'
    return result


def persist(candidate,label,output):
    dest=output/'candidates'/label;dest.mkdir(parents=True)
    export_csv(candidate['trajectory']['rows'],dest/'trajectory.csv')
    metrics=measure(candidate['trajectory'],candidate['scenario'])
    row={'label':label,'makespan_s':metrics['makespan_s'],'metrics':metrics,
         'tasks':len(candidate['scenario'].tasks),'layers':len(candidate['scenario'].layers),
         'shape':aggregate(candidate['scenario'].scores,candidate['scenario'].config),
         'scenario_sha256':scenario_hash(candidate['scenario']),
         'trajectory_sha256':digest(dest/'trajectory.csv'),
         'path':str(dest.relative_to(output)),'search':candidate.get('search',{})}
    save(dest/'candidate.json',row);save(dest/'plan.json',dump_plan(candidate['plan']))
    return row


def main(start):
    imports_complete=perf_counter()-start
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=tuple(MODES),default=DEFAULT_MODE);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--validate',action='store_true',help='Official validation always enabled')
    p.add_argument('--wall-limit',type=float,default=0.,help='Emergency process-tree compute cap; final fallback validation may exceed it')
    a=p.parse_args();source=Path(a.scenario).resolve();output=Path(a.output).resolve()
    if output.exists():raise FileExistsError(f'Refusing overwrite: {output}')
    for name in ('config.yaml','target.stl'):
        if not (source/name).is_file():raise FileNotFoundError(source/name)
    output.mkdir(parents=True);timers=Timers();settings=MODES[a.mode];rows=[];baseline=None;budget_exhausted=False
    reference_validation=0.;reference_result={};reference_info={}
    def expired():return bool(a.wall_limit and perf_counter()-start>=a.wall_limit)
    inputs={}
    with timers.stage('input_copy_and_hash'):
        for name in ('config.yaml','target.stl'):
            shutil.copy2(source/name,output/name);inputs[name]=digest(output/name)
    save(output/'run_options.json',{'mode':a.mode,'seed':a.seed,'settings':settings,'version':VERSION,'input_sha256':inputs})
    try:
        with timers.stage('config_loading'):
            config=load_config(output/'config.yaml').model_dump()
        with timers.stage('original_A_total_wall'):
            bundle,prepared,reference_result,reference_info=reference.generate(output,output,a.mode,a.seed)
            reference_validation=reference_info['candidate_validation_s']
        timers.seconds['original_A_algorithm_and_process']=timers.seconds.pop('original_A_total_wall')-reference_validation
        scenario=bundle['scenario'] if bundle else None
        if scenario is not None:
            baseline={**bundle,'evaluator':core.scheduling.Evaluator(scenario,certified_timeline=True),
                      'search':reference_result.get('search',{})}
            with timers.stage('baseline_export_and_isolation'):
                rows.append(persist(baseline,'A',output));baseline_hash=scenario_hash(scenario)
                a_csv=rows[0]['trajectory_sha256']
            print('Independent A',rows[-1]['makespan_s'],flush=True)
            save(output/'candidate_progress.json',rows)
            if not expired():
                try:
                    improved,detail=core.improve(baseline,settings,a.seed,timers)
                    save(output/'core_detail.json',detail)
                    with timers.stage('core_export'):
                        rows.append(persist(improved,'Core',output))
                    assert scenario_hash(scenario)==baseline_hash, 'A geometry/tasks changed'
                    assert digest(output/'candidates/A/trajectory.csv')==a_csv, 'A CSV changed'
                except Exception as exc:
                    save(output/'core_failure.json',{'reason':repr(exc),'traceback':traceback.format_exc(),'baseline_retained':True})
            else:budget_exhausted=True
        else:
            if not expired():
                with timers.stage('micro_recovery'):
                    if prepared is None:
                        prepared=slice_layers(output/'target.stl',config,return_mesh=True)
                    recovered,detail=micro.recover(prepared,config,{},
                                                   settings['micro_evaluations_per_component'])
                    save(output/'micro_detail.json',detail)
                if recovered is not None:
                    with timers.stage('micro_schedule'):
                        candidate=core.schedule(recovered,settings['baseline_steps'],a.seed)
                    with timers.stage('micro_export'):
                        rows.append(persist(candidate,'Micro',output))
            else:budget_exhausted=True
        budget_exhausted=budget_exhausted or expired()
    except (ValueError,InputValidationError) as exc:
        save(output/'failure.json',{'status':'GEOMETRY_GATE_FAILED','reason':str(exc),'trajectory_available':False})
    save(output/'candidate_progress.json',rows)
    # Normalize file bytes produced by one fixed exporter before deduplication.
    unique={}
    for row in rows:
        if row['shape']['passed']:unique.setdefault(row['trajectory_sha256'],row)
    options=sorted(unique.values(),key=lambda r:(r['makespan_s'],-r['shape']['iou'],r['label']))
    attempts=[];official_s=reference_validation;selected=None;csv_ready=None;validation=None
    for row in options:
        with timers.stage('selected_csv_export'):
            shutil.copy2(output/row['path']/'trajectory.csv',output/'trajectory.csv')
            ready=perf_counter()-start
        validation=official(output,output/'validation_attempts'/f'{len(attempts):03d}')
        official_s+=validation['wall_s']
        attempts.append({'label':row['label'],'official':validation,'csv_ready_s':ready})
        if validation['passed']:
            selected=row;csv_ready=ready;break
    status='PASS' if selected else 'FAIL'
    candidate_validation=reference_validation+sum(x['official']['wall_s'] for x in attempts[:-1]) if selected else official_s
    final_validation=validation['wall_s'] if selected else 0.
    end=perf_counter()
    final=selected['metrics'] if selected else None
    guarantee=selected is not None and (not baseline or selected['makespan_s']<=rows[0]['makespan_s']+1e-7)
    result={'status':status,'mode':a.mode,'seed':a.seed,'version':VERSION,'settings':settings,
            'reference_A':reference_info,'reference_A_official':reference_result.get('official'),
            'candidates':rows,'unique_candidates':len(options),'selected':selected['label'] if selected else None,
            'final':final,'official':validation,'attempts':attempts,'candidate_zero_unchanged':True,
            'baseline_guarantee':{'passed':guarantee,'A_internal_makespan_s':rows[0]['makespan_s'] if baseline else None,
              'scope':'Same invocation literal unmodified original Final A via the existing Safe worker; original A is independently officially validated and immutable. No historical-best guarantee.',
              'A_official_passed':reference_result.get('official',{}).get('passed',False)},
            'budget_exhausted':budget_exhausted,
            'determinism_scope':'Core/Micro fixed work; original A retains legacy wall-budget stops. Repeat hashes must be measured, not assumed.',
            'runtime':{'T_to_final_csv':csv_ready,'T_algorithm_excluding_validator':end-start-official_s,
                       'T_candidate_validation':candidate_validation,'T_final_validation':final_validation,
                       'T_total_with_validation':end-start,'exclusive_stages':timers.seconds,
                       'process_start_import_s':imports_complete,
                       'unattributed_after_import_s':end-start-official_s-sum(timers.seconds.values())-imports_complete,
                       'note':'All candidate selection and preprocessing included in T_to_final_csv. Timers are exclusive. Entrypoint hard-caps the worker tree; mandatory fallback validation is separate.'},
            'input_sha256':inputs,'trajectory_sha256':digest(output/'trajectory.csv') if selected else None,
            'environment':{'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('numpy','shapely','trimesh','polars','pydantic','PyYAML')}}}
    if not selected:
        preceding=read(output/'failure.json') if (output/'failure.json').exists() else None
        micro_detail=read(output/'micro_detail.json') if (output/'micro_detail.json').exists() else None
        original_failure=read(output/'reference_A/failure.json') if (output/'reference_A/failure.json').exists() else None
        save(output/'failure.json',{'status':'SEARCH_EXHAUSTED' if not attempts else 'OFFICIAL_VALIDATION_FAILED',
             'not_an_impossibility_certificate':True,'attempts':attempts,'preceding_error':preceding,
             'original_A_failure':original_failure,'micro_result':micro_detail})
    save(output/'solver_result.json',result)
    print(status,final['makespan_s'] if final else None, 'CSV wall',csv_ready,str(output),flush=True)
    return 0 if selected else 2
