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

from src.experiments import (
    run_double_well,
    run_finite_square_well,
    run_harmonic_oscillator,
    run_quartic_oscillator_demo,
    run_square_well,
)
from test_solver import run_all_tests

RESULTS = Path("results")


def main() -> None:
    """
    Run the full project pipeline and regenerate the results directory.

    The routine clears previous outputs, executes all numerical experiments used
    in the project report, and then runs the automated validation tests.
    """
    shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir(exist_ok=True)

    print("Running experiments...")

    print("\n1. Infinite Square Well...")
    run_square_well(RESULTS)

    print("\n2. Harmonic Oscillator...")
    run_harmonic_oscillator(RESULTS)

    print("\n3. Double Well...")
    run_double_well(RESULTS)

    print("\n4. Finite Square Well...")
    run_finite_square_well(RESULTS)

    print("\n5. Quartic Oscillator Demo...")
    run_quartic_oscillator_demo(RESULTS)

    print("\nRunning tests...")
    run_all_tests()

    print(f"\nDone. Results written to: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
