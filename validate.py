"""Convenience launcher for the unchanged official Validator."""
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    environment = ROOT / 'WAAM_Validator' / '.venv'
    python = (
        environment / 'Scripts' / 'python.exe'
        if os.name == 'nt'
        else environment / 'bin' / 'python'
    )
    if not python.is_file():
        raise SystemExit(f'First run: {sys.executable} setup_environment.py')
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    raise SystemExit(subprocess.call(
        [str(python), '-B', '-m', 'waam_validator.cli', *sys.argv[1:]], env=env))
