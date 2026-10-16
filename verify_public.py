"""Run the official Validator against all six submitted public models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODELS = tuple(f"{number:02d}" for number in range(1, 7))


def run_validator(*arguments: str) -> None:
    command = [sys.executable, str(ROOT / "validate.py"), *arguments]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate public_models/01 through public_models/06."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run input checks without the full Validator simulation.",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = ROOT / "verification_results" / run_id

    for model in MODELS:
        scenario = ROOT / "public_models" / model
        print(f"[{model}] Validator check", flush=True)
        run_validator("check", str(scenario))
        if not args.check_only:
            output = result_root / model
            print(f"[{model}] Validator run", flush=True)
            run_validator("run", str(scenario), "--output", str(output))

    action = "checks" if args.check_only else "checks and runs"
    print(f"PASS: all six public model {action} completed successfully.")
    if not args.check_only:
        print(f"Results: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
