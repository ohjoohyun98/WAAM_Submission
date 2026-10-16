"""Invoke the existing Safe A worker unchanged in an independent process.

This deliberately keeps literal original-A protection. Its wall-budget behavior
must be disclosed separately from the deterministic Release improvement budget.
Only this invocation's own pickle handoff is loaded; never user input pickles.
"""
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from .common import read

ROOT=Path(__file__).resolve().parents[3]


def generate(source,output,mode,seed):
    dest=output/'reference_A'
    command=[sys.executable,'-B',str(ROOT/'run_solver_final_safe.py'),'_worker','frozen_final',
             '--scenario',str(source),'--output',str(dest),'--mode',mode,'--seed',str(seed)]
    started=perf_counter()
    with (output/'reference_A.log').open('x',encoding='utf-8') as log:
        rc=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT).returncode
    from ..competition_final_safe.worker import load_bundle
    static=load_bundle(dest/'static.pickle')['slice'] if (dest/'static.pickle').exists() else None
    bundle=load_bundle(dest/'frozen_bundle.pickle') if (dest/'frozen_bundle.pickle').exists() else None
    result=read(dest/'solver_result.json') if (dest/'solver_result.json').exists() else {}
    return bundle,static,result,{'wall_s':perf_counter()-started,'returncode':rc,
                                  'candidate_validation_s':result.get('runtime',{}).get('validator_external_wall_s',0.)}
