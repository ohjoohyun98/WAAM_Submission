"""Cold final competition entry point; no model IDs or persistent input cache."""
import os,sys,time,hashlib
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUTF8']='1';sys.dont_write_bytecode=True
START=time.perf_counter()
import run_solver_competition as base
from waam_solver.competition_final import scenario
from waam_solver.competition_final import scheduling
SOURCE_PATHS=[base.Path(__file__),*sorted((base.ROOT/'src/waam_solver/competition_final').glob('*.py'))]
SOURCE_HASH={str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS}
ORIGINAL_VALIDATE=base.validate

def validate_with_history(job):
    result=ORIGINAL_VALIDATE(job)
    # Preserve an advanced candidate's FAIL evidence even when the proven
    # baseline is later selected and the top-level validation is rerun.
    count=1
    while (job/'validation_attempts'/f'{count:03d}').exists():count+=1
    dest=job/'validation_attempts'/f'{count:03d}';dest.mkdir(parents=True)
    for name in ('validator_check.log','validator_run.log','validation/summary.json','validation/validation_inputs.json'):
        if (job/name).exists():base.shutil.copy2(job/name,dest/base.Path(name).name)
    base.save(dest/'result.json',result)
    if not result['passed'] and (job/'trajectory.csv').exists():base.shutil.copy2(job/'trajectory.csv',dest/'trajectory.csv')
    return result

def main():
    parser=base.parser();parser.add_argument('--no-path-minimization',action='store_true')
    parser.add_argument('--no-spatial',action='store_true')
    parser.add_argument('--no-certificate-cache',action='store_true')
    args=parser.parse_args()
    # Explicit final defaults: do not depend on a mutable v3 deployment file.
    if args.algorithm=='auto':args.algorithm='lns'
    if args.engine=='auto':
        args.engine='reservations';args.certified_timeline=True;args.geometry_memo=True
    if args.primitive_mm<=0 or args.cluster_size<0 or (args.budget is not None and args.budget<0):raise ValueError('Invalid budget/granularity')
    if not args.output:args.output=str(base.ROOT/'outputs/competition_final'/base.datetime.now().strftime('job_%Y%m%d_%H%M%S_%f'))
    base.cold_scenario=scenario.cold_scenario;scenario.MINIMIZE=not args.no_path_minimization;base.START=START
    scheduling.SPATIAL=not args.no_spatial;base.greedy_plan=scheduling.greedy_plan;base.optimize=scheduling.optimize;base.Evaluator=scheduling.Evaluator
    scheduling.CACHE_CERTIFICATES=not args.no_certificate_cache
    base.validate=validate_with_history
    output=base.Path(args.output).resolve();existed=output.exists()
    try:return base.solve(args)
    finally:
        if not existed and output.exists():
            base.save(output/'geometry_final.json',scenario.LAST_AUDIT)
            base.save(output/'scheduling_final.json',scheduling.AUDIT)
            base.save(output/'final_source.json',SOURCE_HASH)
if __name__=='__main__':raise SystemExit(main())
