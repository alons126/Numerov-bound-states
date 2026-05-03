from __future__ import annotations

import shutil
import sys
from pathlib import Path

"""
Project entry point for the Numerov bound-state solver.

This script runs the full project workflow described in the report:
1. Benchmark calculations for analytically solvable potentials
2. Exploratory calculations for nontrivial potentials
3. Convergence and splitting studies
4. Automated tests for numerical sanity checks

This script is intentionally small. Its job is orchestration, not numerical
work. A reviewer reading the project from top to bottom can treat it as the
"main program":
- Clear previous generated outputs
- Run each experiment module in the intended report order
- Run the regression tests
- Print a concise completion summary

Keeping this file thin is part of the project structure: most scientific logic
lives in `src/`, while this script simply reproduces the full workflow.
"""

# Make the project root and tests directory importable when this file is run
# directly from the scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"

for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        # Prepend these paths so local project modules win over any unrelated
        # installed packages with the same names.
        sys.path.insert(0, str(path))

from test_solver import run_all_tests

RESULTS = PROJECT_ROOT / "results"


# ===========================================================================
# FUNCTION: main
# ===========================================================================
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
            run_quartic_double_well,
            run_finite_square_well,
            run_harmonic_oscillator,
            run_square_well,
        )
    except ModuleNotFoundError as exc:
        # If the experiments module fails to import, we can still run the tests
        # and confirm that the core solver logic is sound, but we won't be able
        # to reproduce the figures or results from the report.
        experiments_import_error = exc
        print(f"Skipping experiments: {exc}")
    else:
        # If the import succeeded, run the full experiment suite.
        print("Running experiments...")

        # Keep the execution order aligned with the report structure so the
        # generated results tree is easy to compare against the writeup.
        print("\n1. Infinite Square Well (Numerov only)")
        print("------------------------------------------------------------------\n")
        run_square_well(RESULTS)

        print("\n2. Harmonic Oscillator (Numerov & RK4)")
        print("------------------------------------------------------------------\n")
        run_harmonic_oscillator(RESULTS)

        print("\n3. Quartic double Well (Numerov only)")
        print("------------------------------------------------------------------\n")
        run_quartic_double_well(RESULTS)

        print("\n4. Finite Square Well (Numerov only)")
        print("------------------------------------------------------------------\n")
        run_finite_square_well(RESULTS)

        print("\n5. Scattering and Resonant Tunneling (Numerov only)")
        print("------------------------------------------------------------------\n")
        run_scattering(RESULTS)

    # Run tests after the experiments so the workflow reproduces the figures
    # first and then performs a quick regression pass on the codebase.
    print("\nRunning tests")
    print("------------------------------------------------------------------\n")
    run_all_tests()

    if experiments_import_error is None:
        # If the experiments ran successfully, print the path to the generated results.
        print(f"\nDone. Results written to: {RESULTS.resolve()}")
    else:
        # If the experiments were skipped due to an import error, print a concise summary
        # clarifying that the tests ran but the experiment generation was skipped.
        print("\nDone. Tests ran, but experiment generation was skipped.")


if __name__ == "__main__":
    main()
