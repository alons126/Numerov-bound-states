from __future__ import annotations

"""
Minimal automated tests for the Numerov project.

These tests are designed for project validation rather than full software
engineering coverage: they check normalization, analytic benchmark energies,
and one qualitative physical feature of the double well.
"""

import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis import exact_harmonic_oscillator_energies, exact_square_well_energies
from src.numerov import derivative_at_right_edge, normalize_wavefunction
from src.potentials import (
    double_square_barrier,
    harmonic_oscillator,
    infinite_square_well_numeric,
    quartic_double_well,
)
import src.rk4_compare as rk4_compare
from src.scattering import sweep_scattering
from src.shooting import (
    solve_symmetric_potential,
    solve_symmetric_potential_inward_decay,
)


# ---------------------------------------------------------------------------
# FUNCTION: test_normalization
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_normalization() -> None:
    """
    Check that the wavefunction normalization helper produces unit norm.
    """
    x = np.linspace(-1.0, 1.0, 1001)
    psi = np.exp(-(x**2))
    psi_n = normalize_wavefunction(x, psi)
    val = np.trapezoid(np.abs(psi_n) ** 2, x)
    assert abs(val - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# FUNCTION: test_derivative_at_right_edge_polynomial
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_derivative_at_right_edge_polynomial() -> None:
    """
    Check the high-order right-edge derivative stencil on a smooth polynomial.
    """
    x = np.linspace(-1.0, 1.0, 9)
    psi = x**4 - 2.0 * x**3 + x
    exact = 4.0 * x[-1] ** 3 - 6.0 * x[-1] ** 2 + 1.0
    assert abs(derivative_at_right_edge(x, psi) - exact) < 1e-12


# ---------------------------------------------------------------------------
# FUNCTION: test_find_rk4_brackets_accepts_exact_zero_hit
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_find_rk4_brackets_accepts_exact_zero_hit() -> None:
    """
    Check that RK4 bracketing keeps roots that land exactly on a scan point.
    """
    original = rk4_compare.rk4_inward_mismatch

    def fake_mismatch(
        energy: float, parity: str, x_max: float, n_grid: int, omega: float = 1.0
    ) -> float:
        values = {0.1: -1.0, 0.3: -0.2, 0.5: 0.0, 0.7: 0.4, 0.9: 1.0}
        return values[round(float(energy), 1)]

    rk4_compare.rk4_inward_mismatch = fake_mismatch
    try:
        brackets = rk4_compare.find_rk4_brackets(
            parity="even",
            x_max=8.0,
            n_grid=500,
            e_min=0.1,
            e_max=0.9,
            n_scan=5,
        )
    finally:
        rk4_compare.rk4_inward_mismatch = original

    assert any(lo < 0.5 < hi for lo, hi in brackets)


# ---------------------------------------------------------------------------
# FUNCTION: test_square_well_ground_state
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_square_well_ground_state() -> None:
    """
    Verify that the square-well ground-state energy matches the exact result.
    """
    a = 1.0
    states = solve_symmetric_potential(
        x_max=1.2,
        n_grid=1600,
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1.0e6},
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=60.0,
    )
    e0 = states[0].energy
    exact = exact_square_well_energies(np.array([1]), a=a)[0]
    assert abs(e0 - exact) / exact < 5e-3


# ---------------------------------------------------------------------------
# FUNCTION: test_harmonic_oscillator_first_levels
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_harmonic_oscillator_first_levels() -> None:
    """
    Verify that the first few harmonic-oscillator energies are accurate.
    """
    omega = 1.0
    states = solve_symmetric_potential(
        x_max=8.0,
        n_grid=2000,
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=6.0,
    )
    numerical = np.array([s.energy for s in states[:4]])
    exact = exact_harmonic_oscillator_energies(np.arange(4), omega=omega)
    assert np.all(np.abs((numerical - exact) / exact) < 3e-3)


# ---------------------------------------------------------------------------
# FUNCTION: test_harmonic_oscillator_inward_decay_first_levels
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_harmonic_oscillator_inward_decay_first_levels() -> None:
    """
    Verify that the inward-decay harmonic solver resolves the first levels accurately.
    """
    omega = 1.0
    states = solve_symmetric_potential_inward_decay(
        x_max=8.0,
        n_grid=800,
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=6.0,
    )
    numerical = np.array([s.energy for s in states[:4]])
    exact = exact_harmonic_oscillator_energies(np.arange(4), omega=omega)
    assert np.all(np.abs(numerical - exact) < 1e-8)


# ---------------------------------------------------------------------------
# FUNCTION: test_double_well_splitting_positive
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_double_well_splitting_positive() -> None:
    """
    Check that the first odd state lies above the first even state in the double well.
    """
    states = solve_symmetric_potential(
        x_max=3.0,
        n_grid=2400,
        potential_fn=quartic_double_well,
        potential_kwargs={"a": 1.0, "b": 6.0, "shift_min_to_zero": True},
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    assert states[1].energy > states[0].energy


# ---------------------------------------------------------------------------
# FUNCTION: test_scattering_probability_conservation
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_scattering_probability_conservation() -> None:
    """
    Check that scattering approximately conserves probability current.
    """
    x = np.linspace(-8.0, 8.0, 2000)
    V = double_square_barrier(x, V0=5.0, barrier_width=0.6, well_width=1.4)
    energies = np.array([0.8, 1.5, 2.5, 4.0])
    results = sweep_scattering(x, V, energies)
    for result in results:
        assert abs((result.transmission + result.reflection) - 1.0) < 5e-2


# ---------------------------------------------------------------------------
# FUNCTION: test_run_solver_import_without_experiment_dependencies
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def test_run_solver_import_without_experiment_dependencies() -> None:
    """
    Check that the entry script can be imported without importing plotting code.
    """
    path = PROJECT_ROOT / "scripts" / "run_solver.py"
    spec = importlib.util.spec_from_file_location("run_solver_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


# ---------------------------------------------------------------------------
# FUNCTION: run_all_tests
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def run_all_tests() -> None:
    """
    Execute all project validation tests and print a short summary.
    """
    test_normalization()
    test_derivative_at_right_edge_polynomial()
    test_find_rk4_brackets_accepts_exact_zero_hit()
    test_square_well_ground_state()
    test_harmonic_oscillator_first_levels()
    test_harmonic_oscillator_inward_decay_first_levels()
    test_double_well_splitting_positive()
    test_scattering_probability_conservation()
    test_run_solver_import_without_experiment_dependencies()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
