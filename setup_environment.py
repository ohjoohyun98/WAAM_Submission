"""Create the isolated runtime used by the submission on any supported OS."""
import os
from pathlib import Path
import subprocess
import sys


def environment_python(environment: Path) -> Path:
    if os.name == 'nt':
        return environment / 'Scripts' / 'python.exe'
    return environment / 'bin' / 'python'


if __name__ == '__main__':
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Use Python 3.11 on Windows, macOS, or Linux.')
    root = Path(__file__).resolve().parent
    environment = root / 'WAAM_Validator' / '.venv'
    python = environment_python(environment)
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME'):env.pop(name,None)
    env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    def run(*args):subprocess.run(list(map(str,args)),cwd=root,env=env,check=True)
    run(sys.executable,'-m','venv',environment)
    # The unchanged Validator pyproject requires modern PEP 639 build tools.
    # Pin them in this NEW environment rather than use an unpinned build island.
    run(python,'-m','pip','install','--no-cache-dir','setuptools==80.9.0','wheel==0.45.1')
    run(python,'-m','pip','install','--no-cache-dir','-r',root/'requirements.txt')
    run(python,'-m','pip','install','--no-cache-dir','--no-build-isolation','--no-deps',root/'WAAM_Validator')
    run(python,'-m','pip','check')
    launcher = 'python' if os.name == 'nt' else 'python3.11'
    print(f'Ready: {launcher} run_solver.py --scenario INPUT --output NEW_OUTPUT --validate')
