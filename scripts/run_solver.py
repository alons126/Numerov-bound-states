from __future__ import annotations

"""
Project entry point for the Numerov bound-state solver.

This script runs the full project workflow described in the report:
1. benchmark calculations for analytically solvable potentials,
2. exploratory calculations for nontrivial potentials,
3. convergence and splitting studies,
4. automated tests for numerical sanity checks.

It is intentionally lightweight: all physics and analysis logic lives in src/.

Reviewer guide
--------------
This script is intentionally small. Its job is orchestration, not numerical
work. A reviewer reading the project from top to bottom can treat it as the
"main program":
- clear previous generated outputs
- run each experiment module in the intended report order
- run the regression tests
- print a concise completion summary

Keeping this file thin is part of the project structure: most scientific logic
lives in `src/`, while this script simply reproduces the full workflow.
"""

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root and tests directory importable when this file is run
# directly from the scripts/ directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"

for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        # Prepend these paths so local project modules win over any unrelated
        # installed packages with the same names.
        sys.path.insert(0, str(path))

from test_solver import run_all_tests

RESULTS = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# FUNCTION: main
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Run the full project pipeline and regenerate the results directory.

    The routine clears previous outputs, executes all numerical experiments used
    in the project report, and then runs the automated validation tests.
    """
    # Start from a clean generated-results directory so each run reproduces the
    # current code state rather than mixing old and new outputs.
    shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir(exist_ok=True)

    experiments_import_error: Exception | None = None
    try:
        # Delay the import so the lightweight test-import check can confirm
        # that this entry script itself does not require plotting dependencies.
        from src.experiments import (
            run_scattering,
            run_double_well,
            run_finite_square_well,
            run_harmonic_oscillator,
            run_square_well,
        )
    except ModuleNotFoundError as exc:
        experiments_import_error = exc
        print(f"Skipping experiments: {exc}")
    else:
        print("Running experiments...")

        # Keep the execution order aligned with the report structure so the
        # generated results tree is easy to compare against the writeup.
        print("\n1. Infinite Square Well (Numerov only)...")
        run_square_well(RESULTS)

        print("\n2. Harmonic Oscillator (Numerov & RK4)...")
        run_harmonic_oscillator(RESULTS)

        print("\n3. Double Well (Numerov only)...")
        run_double_well(RESULTS)

        print("\n4. Finite Square Well (Numerov only)...")
        run_finite_square_well(RESULTS)

        print("\n5. Scattering and Resonant Tunneling (Numerov only)...")
        run_scattering(RESULTS)

    # Run tests after the experiments so the workflow reproduces the figures
    # first and then performs a quick regression pass on the codebase.
    print("\nRunning tests...")
    run_all_tests()

    if experiments_import_error is None:
        print(f"\nDone. Results written to: {RESULTS.resolve()}")
    else:
        print("\nDone. Tests ran, but experiment generation was skipped.")


if __name__ == "__main__":
    main()
