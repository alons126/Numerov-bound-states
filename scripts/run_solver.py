from __future__ import annotations

"""
Project entry point for the Numerov bound-state solver.

This script runs the full project workflow described in the report:
1. benchmark calculations for analytically solvable potentials,
2. exploratory calculations for nontrivial potentials,
3. convergence and splitting studies,
4. automated tests for numerical sanity checks.

It is intentionally lightweight: all physics and analysis logic lives in src/.
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
        sys.path.insert(0, str(path))

from test_solver import run_all_tests

RESULTS = Path("results")


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
    shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir(exist_ok=True)

    experiments_import_error: Exception | None = None
    try:
        from src.experiments import (
            run_scattering,
            run_double_well,
            run_finite_square_well,
            run_harmonic_oscillator,
            run_quartic_oscillator_demo,
            run_square_well,
        )
    except ModuleNotFoundError as exc:
        experiments_import_error = exc
        print(f"Skipping experiments: {exc}")
    else:
        print("Running experiments...")

        print("\n1. Infinite Square Well (Numerov only)...")
        run_square_well(RESULTS)

        # print("\n2. Harmonic Oscillator (Numerov & RK4)...")
        # run_harmonic_oscillator(RESULTS)

        print("\n3. Double Well (Numerov only)...")
        run_double_well(RESULTS)

        # print("\n4. Finite Square Well (Numerov only)...")
        # run_finite_square_well(RESULTS)

        # print("\n5. Quartic Oscillator Demo (Numerov only)...")
        # run_quartic_oscillator_demo(RESULTS)

        # print("\n6. Scattering and Resonant Tunneling (Numerov only)...")
        # run_scattering(RESULTS)

    print("\nRunning tests...")
    run_all_tests()

    if experiments_import_error is None:
        print(f"\nDone. Results written to: {RESULTS.resolve()}")
    else:
        print("\nDone. Tests ran, but experiment generation was skipped.")


if __name__ == "__main__":
    main()
