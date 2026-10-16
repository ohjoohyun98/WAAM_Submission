"""Deployment-safe portfolio entry point. Frozen implementations remain untouched."""
import os, sys, time
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUTF8'] = '1'
sys.dont_write_bytecode = True
START = time.perf_counter()
from rl_runtime import bootstrap
ROOT = bootstrap()
sys.path.insert(0, str(ROOT.parent / 'WAAM_Validator/src'))
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        from waam_solver.competition_final_safe.worker import main
        raise SystemExit(main(sys.argv[2:], START))
    elif len(sys.argv) > 1 and sys.argv[1] == 'audit':
        from waam_solver.competition_final_safe.workflow import main
        main(sys.argv[2:])
    else:
        from waam_solver.competition_final_safe.cli import main
        raise SystemExit(main(START))
