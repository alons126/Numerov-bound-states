from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"

for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.experiments import (
    run_double_well,
    run_extra_potential,
    run_harmonic_oscillator,
    run_quartic_oscillator_demo,
    run_square_well,
)
from test_solver import run_all_tests


RESULTS = Path("results")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    run_square_well(RESULTS)
    run_harmonic_oscillator(RESULTS)
    run_double_well(RESULTS)
    run_extra_potential(RESULTS)
    run_quartic_oscillator_demo(RESULTS)
    run_all_tests()
    print(f"\nDone. Results written to: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
