"""Typer command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .dashboard.app import run_validator_ui
from .dashboard.data import DashboardDataError
from .errors import ComputationError, WaamValidatorError
from .pipeline import check_input, run_validation
from .reporting.writers import render_console_summary, render_error_block

app = typer.Typer(
    name="waam-validator",
    help="Validate three-robot WAAM trajectory jobs.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    job_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing the three fixed input files."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Use an exact new or empty output directory."),
    ] = None,
    headless: Annotated[
        bool,
        typer.Option("--headless", help="Deprecated compatibility option; core output is lean."),
    ] = False,
    replay: Annotated[
        bool,
        typer.Option("--replay", help="Deprecated; generate Replay from the result screen."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print exactly one summary JSON object to stdout."),
    ] = False,
) -> None:
    """Run the complete validation pipeline."""
    if headless:
        typer.echo(
            "Warning: --headless is deprecated; Validation now always writes the lean core bundle.",
            err=True,
        )
    if replay:
        typer.echo(
            "Warning: --replay is deprecated; generate replay.html from the result screen.",
            err=True,
        )
    try:
        result = run_validation(job_dir, output)
    except WaamValidatorError as exc:
        typer.echo(render_error_block(exc, job_dir.expanduser().resolve()), err=True)
        raise typer.Exit(exc.exit_code) from None
    except Exception as exc:  # Defensive CLI boundary.
        wrapped = ComputationError("INTERNAL_CALCULATION_ERROR", str(exc))
        typer.echo(render_error_block(wrapped, job_dir.expanduser().resolve()), err=True)
        raise typer.Exit(wrapped.exit_code) from None
    if json_output:
        typer.echo(
            json.dumps(
                result.summary_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    else:
        typer.echo(render_console_summary(result))
    raise typer.Exit(0 if result.status == "PASS" else 1)


@app.command()
def check(
    job_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing the three fixed input files."),
    ],
) -> None:
    """Validate input structure and parsing without running the simulation."""
    try:
        row_count, _, _ = check_input(job_dir)
    except WaamValidatorError as exc:
        typer.echo("WAAM INPUT CHECK : FAIL")
        typer.echo(f"Code    : {exc.code}")
        typer.echo(f"Message : {exc.message}")
        raise typer.Exit(exc.exit_code) from None
    typer.echo("WAAM INPUT CHECK : PASS")
    typer.echo("config.yaml    : OK")
    typer.echo(f"trajectory.csv : OK (3 robots, {row_count} rows)")
    typer.echo("target.stl     : OK")


@app.command()
def ui(
    job_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing config.yaml, trajectory.csv, and target.stl."),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Local interface to bind."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="Local WAAM Validator UI port."),
    ] = 8050,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not open the default browser."),
    ] = False,
) -> None:
    """Open the single-job WAAM Validator interface."""
    try:
        run_validator_ui(
            job_dir,
            host=host,
            port=port,
            open_browser=not no_browser,
        )
    except DashboardDataError as exc:
        typer.echo(f"WAAM UI ERROR: {exc}", err=True)
        raise typer.Exit(2) from None
    except OSError as exc:
        typer.echo(f"WAAM UI ERROR: {exc}", err=True)
        raise typer.Exit(4) from None


if __name__ == "__main__":
    app()
