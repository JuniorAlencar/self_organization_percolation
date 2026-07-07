"""Compatibility wrapper for cluster multi-run helpers.

When running ``python/run_multi.py``, Python resolves ``src`` to this folder.
Delegate to the project-root implementation so both entrypoints use the same
logic.
"""

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_impl = import_module("run_multi_functions")

run_multi_rho_array = _impl.run_multi_rho_array
get_missing_run_parameters = _impl.get_missing_run_parameters
log_existing_samples_for_parameters = _impl.log_existing_samples_for_parameters
cleanup_logged_output_for_parameters = _impl.cleanup_logged_output_for_parameters
snapshot_existing_outputs_for_parameters = _impl.snapshot_existing_outputs_for_parameters
write_existing_output_snapshots = _impl.write_existing_output_snapshots
custom_range = _impl.custom_range

__all__ = [
    "run_multi_rho_array",
    "get_missing_run_parameters",
    "log_existing_samples_for_parameters",
    "cleanup_logged_output_for_parameters",
    "snapshot_existing_outputs_for_parameters",
    "write_existing_output_snapshots",
    "custom_range",
]
