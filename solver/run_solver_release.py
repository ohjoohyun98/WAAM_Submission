"""Competition Release entry point; no dependency on saved result directories."""
import os
import sys
import time
from pathlib import Path

START=time.perf_counter()
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONUTF8']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
# Source-tree development layout, also used by the isolated reproduction package.
VALIDATOR=ROOT.parent/'WAAM_Validator'
if (VALIDATOR/'src').is_dir():sys.path.insert(0,str(VALIDATOR/'src'))

if __name__=='__main__':
    if '--_worker' in sys.argv:
        sys.argv.remove('--_worker')
    elif '--wall-limit' in sys.argv:
        limit=float(sys.argv[sys.argv.index('--wall-limit')+1])
        if limit>0:
            from waam_solver.competition_release.watchdog import run
            raise SystemExit(run(Path(__file__),sys.argv[1:],limit,START))
    from waam_solver.competition_release.pipeline_reference import main
    raise SystemExit(main(START))
