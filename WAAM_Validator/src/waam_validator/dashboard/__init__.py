"""Local single-job WAAM Validator web interface."""

from .app import create_validator_app, run_validator_ui

__all__ = ["create_validator_app", "run_validator_ui"]
