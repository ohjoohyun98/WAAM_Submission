"""Explicit work budgets, chosen before the untouched evaluation."""
MODES = {
    'fast': {'baseline_steps': 0, 'core_prune_evaluations': 48, 'critical_evaluations': 4,
             'core_steps': 0, 'micro_evaluations_per_component': 500, 'unique_prune_layers': 8},
    'balanced': {'baseline_steps': 16, 'core_prune_evaluations': 192, 'critical_evaluations': 12,
                 'core_steps': 16, 'micro_evaluations_per_component': 900, 'unique_prune_layers': 16},
    'best': {'baseline_steps': 48, 'core_prune_evaluations': 288, 'critical_evaluations': 48,
             'core_steps': 32, 'micro_evaluations_per_component': 1800, 'unique_prune_layers': 32},
}
DEFAULT_MODE = 'balanced'
SEED = 11
VERSION = 'release-1'
