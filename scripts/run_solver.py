from __future__ import annotations

import shutil
import sys
from pathlib import Path

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
