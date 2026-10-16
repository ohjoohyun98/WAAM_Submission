"""Single evaluator command; internal modules are not alternative submissions."""
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
ENVIRONMENT=ROOT/'WAAM_Validator'/'.venv'
PYTHON=(ENVIRONMENT/'Scripts'/'python.exe' if os.name=='nt' else ENVIRONMENT/'bin'/'python')
if __name__=='__main__':
    if not PYTHON.exists():raise SystemExit(f'First run: {sys.executable} setup_environment.py')
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME'):env.pop(name,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONHASHSEED='11')
    raise SystemExit(subprocess.call([str(PYTHON),'-B',str(ROOT/'solver/run_solver_release.py'),*sys.argv[1:]],env=env))
