"""Emergency process watchdog. Mandatory validation may exceed its compute cap."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from .common import read,save


def run(entry, argv, limit, start):
    output=Path(argv[argv.index('--output')+1]).resolve()
    if output.exists():raise FileExistsError(output)
    command=[sys.executable,'-B',str(entry),'--_worker',*argv]
    if os.name=='nt':
        from .winjob import OwnedTree
        child=OwnedTree(command)
    else:
        child=subprocess.Popen(command,start_new_session=True)
    try:return child.wait(timeout=max(0.,limit-(perf_counter()-start)))
    except subprocess.TimeoutExpired:
        if os.name=='nt':child.kill()
        else:
            import signal
            os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=10.)
    finally:
        if os.name=='nt':child.close()
    output.mkdir(parents=True,exist_ok=True)
    save(output/'watchdog_shutdown.json',{'mechanism':'owned_windows_job' if os.name=='nt' else 'owned_process_group',
         'worker_tree_stopped':True,'elapsed_to_stop_s':perf_counter()-start,'compute_cap_s':limit})
    progress=output/'candidate_progress.json'
    rows=read(progress) if progress.exists() else []
    if rows or any((output/p).is_file() for p in ('reference_A/trajectory.csv','reference_A/baseline/trajectory.csv')):
        from .pipeline_reference import official
    attempts=[]
    original=read(output/'reference_A/solver_result.json') if (output/'reference_A/solver_result.json').exists() else None
    measured_validation=original.get('runtime',{}).get('validator_external_wall_s',0.) if original else 0.
    def runtime(ready,final_validation):
        total=perf_counter()-start
        return {'T_to_final_csv':ready,'T_final_validation':final_validation,
                'T_candidate_validation_completed_measured':measured_validation-final_validation,
                'T_total_with_validation':total,
                'T_algorithm_plus_unclassified_interrupted_validation':total-measured_validation,
                'note':'All elapsed costs remain in CSV readiness. Validation interrupted before an atomic result is unclassified, not subtracted or hidden.'}
    for index,row in enumerate(sorted(rows,key=lambda r:(r['makespan_s'],r['label']))):
        shutil.copy2(output/row['path']/'trajectory.csv',output/'trajectory.csv')
        ready=perf_counter()-start
        result=official(output,output/'watchdog_validation'/str(index))
        measured_validation+=result['wall_s'];attempts.append({'label':row['label'],'official':result})
        if result['passed']:
            record={'status':'PASS','selected':row['label'],'final':row['metrics'],'official':result,
                    'candidates':rows,'budget_exhausted':True,'reproducibility_guaranteed':False,
                    'runtime':runtime(ready,result['wall_s']),'attempts':attempts,
                    'completed_candidate_search_counts':[r.get('search') for r in rows],
                    'interrupted_stage_evaluations':None,
                    'watchdog_note':'Compute worker stopped; independent persisted incumbent verified afterwards. Mandatory safety validation is outside the emergency compute cap.'}
            save(output/'solver_result.json',record);print('WATCHDOG PASS',row['makespan_s'],flush=True);return 0
    # The original anytime worker can have persisted a baseline before it has
    # finished search/validation. It is not presumed feasible: verify it now.
    for index,path in enumerate((output/'reference_A/trajectory.csv',output/'reference_A/baseline/trajectory.csv')):
        if not path.is_file():continue
        shutil.copy2(path,output/'trajectory.csv');ready=perf_counter()-start
        result=official(output,output/'watchdog_original_A'/str(index))
        measured_validation+=result['wall_s'];attempts.append({'label':str(path),'official':result})
        if result['passed']:
            m=result['summary']['schedule']['makespan_s']
            save(output/'solver_result.json',{'status':'PASS','selected':'watchdog_persisted_original_A',
                 'final':{'makespan_s':m},'official':result,'budget_exhausted':True,
                 'reproducibility_guaranteed':False,
                 'runtime':runtime(ready,result['wall_s']),'attempts':attempts,
                 'interrupted_stage_evaluations':None})
            print('WATCHDOG VERIFIED ORIGINAL A',m,flush=True);return 0
    save(output/'failure.json',{'status':'WATCHDOG_NO_VERIFIED_INCUMBENT','budget_exhausted':True,
                              'reason':'No complete candidate survived or all persisted candidates failed official validation.'})
    save(output/'solver_result.json',{'status':'FAIL','budget_exhausted':True,'attempts':attempts,
          'runtime':runtime(None,0.),'interrupted_stage_evaluations':None})
    return 2
