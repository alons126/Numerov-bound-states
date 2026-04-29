from __future__ import annotations

"""
Minimal automated tests for the Numerov project.

These tests are designed for project validation rather than full software
engineering coverage: they check normalization, analytic benchmark energies,
and one qualitative physical feature of the double well.

Reviewer guide
--------------
These tests are lightweight by software-engineering standards, but they are
targeted at the numerical claims made in the report. The goal is to catch the
kinds of regressions that would silently invalidate the scientific conclusions.

The suite checks:
- normalization and derivative-stencil correctness
- analytic benchmark accuracy for the square well and harmonic oscillator
- expected convergence order where the implementation should recover it
- quartic-double-well details such as the analytic minimum shift, box-size
  sensitivity, parity-specific mismatch behavior, and low-state residuals
- scattering probability conservation
- the top-level script import path behavior

One especially important regression test is the square-well convergence-order
check. It protects the higher-order startup logic in `initial_conditions()`
from quietly degrading in the future.
"""

import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis import (
    convergence_vs_grid,
    convergence_vs_grid_successive,
    estimate_convergence_slopes,
    exact_harmonic_oscillator_energies,
    exact_square_well_energies,
)
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
    sample_boundary_mismatch,
    solve_symmetric_potential,
    solve_symmetric_potential_inward_decay,
)


# ---------------------------------------------------------------------------
# FUNCTION: test_normalization
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
# FUNCTION: test_square_well_convergence_order
# ---------------------------------------------------------------------------
def test_square_well_convergence_order() -> None:
    """
    Check that square-well energies show near-fourth-order grid convergence.
    """
    a = 1.0
    exact = exact_square_well_energies(np.arange(1, 4), a=a)
    conv = convergence_vs_grid(
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1.0e6},
        x_max=a,
        grid_sizes=[50, 80, 120, 180],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=60.0,
        reference_energies=exact,
    )
    slopes = estimate_convergence_slopes(conv["h"], conv["energy_errors"])

    for row in slopes:
        assert row["convergence_exponent_p"] > 3.8


# ---------------------------------------------------------------------------
# FUNCTION: test_square_well_convergence_includes_four_requested_states
# ---------------------------------------------------------------------------
def test_square_well_convergence_includes_four_requested_states() -> None:
    """
    Check that the infinite-well convergence study tracks all four solved states.
    """
    a = 1.0
    exact = exact_square_well_energies(np.arange(1, 5), a=a)
    conv = convergence_vs_grid(
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1.0e6},
        x_max=a,
        grid_sizes=[50, 80, 120, 180],
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=80.0,
        reference_energies=exact,
    )

    assert conv["energy_errors"].shape == (4, 4)


# ---------------------------------------------------------------------------
# FUNCTION: test_harmonic_oscillator_first_levels
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
# FUNCTION: test_quartic_double_well_exact_shifted_minimum
# ---------------------------------------------------------------------------
def test_quartic_double_well_exact_shifted_minimum() -> None:
    """
    Check that the shifted quartic double well uses the analytic minimum.
    """
    b = 6.0
    x_min = np.sqrt(b / 2.0)
    values = quartic_double_well(
        np.array([0.0, x_min]), a=1.0, b=b, shift_min_to_zero=True
    )
    assert abs(values[1]) < 1e-12
    assert abs(values[0] - 9.0) < 1e-12


# ---------------------------------------------------------------------------
# FUNCTION: test_double_well_larger_box_improves_low_lying_energies
# ---------------------------------------------------------------------------
def test_double_well_larger_box_improves_low_lying_energies() -> None:
    """
    Check that enlarging the double-well box reduces truncation error.
    """
    kwargs = {"a": 1.0, "b": 6.0, "shift_min_to_zero": True}
    states_small = solve_symmetric_potential(
        x_max=3.0,
        n_grid=1500,
        potential_fn=quartic_double_well,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    states_large = solve_symmetric_potential(
        x_max=4.0,
        n_grid=2000,
        potential_fn=quartic_double_well,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    states_ref = solve_symmetric_potential(
        x_max=5.0,
        n_grid=2500,
        potential_fn=quartic_double_well,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )

    err_small = np.max(
        [abs(states_small[i].energy - states_ref[i].energy) for i in range(4)]
    )
    err_large = np.max(
        [abs(states_large[i].energy - states_ref[i].energy) for i in range(4)]
    )

    assert err_large < 1e-4
    assert err_large < 0.1 * err_small


# ---------------------------------------------------------------------------
# FUNCTION: test_double_well_same_box_grid_errors_decrease
# ---------------------------------------------------------------------------
def test_double_well_same_box_grid_errors_decrease() -> None:
    """
    Check that double-well successive refinement errors decrease on a fixed box.
    """
    kwargs = {"a": 1.0, "b": 6.0, "shift_min_to_zero": True}
    conv = convergence_vs_grid_successive(
        potential_fn=quartic_double_well,
        potential_kwargs=kwargs,
        x_max=4.0,
        grid_sizes=[600, 1000, 1600, 2200, 3000],
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )

    for state_index in range(conv["energy_errors"].shape[1]):
        errors = conv["energy_errors"][:, state_index]
        assert np.all(np.diff(errors) < 0.0)


# ---------------------------------------------------------------------------
# FUNCTION: test_double_well_successive_convergence_order
# ---------------------------------------------------------------------------
def test_double_well_successive_convergence_order() -> None:
    """
    Check that the low double-well states recover near-fourth-order convergence.
    """
    kwargs = {"a": 1.0, "b": 6.0, "shift_min_to_zero": True}
    conv = convergence_vs_grid_successive(
        potential_fn=quartic_double_well,
        potential_kwargs=kwargs,
        x_max=4.0,
        grid_sizes=[600, 1000, 1600, 2200, 3000],
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    slopes = estimate_convergence_slopes(conv["h"], conv["energy_errors"])
    assert slopes[0]["convergence_exponent_p"] > 3.5
    assert slopes[1]["convergence_exponent_p"] > 3.5
    assert slopes[2]["convergence_exponent_p"] > 3.5


# ---------------------------------------------------------------------------
# FUNCTION: test_double_well_low_state_mismatches_are_polished
# ---------------------------------------------------------------------------
def test_double_well_low_state_mismatches_are_polished() -> None:
    """
    Check that the low double-well roots are polished beyond the bisection width floor.
    """
    kwargs = {"a": 1.0, "b": 6.0, "shift_min_to_zero": True}
    states = solve_symmetric_potential(
        x_max=4.0,
        n_grid=2000,
        potential_fn=quartic_double_well,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    assert abs(states[0].mismatch) < 1e-5
    assert abs(states[1].mismatch) < 1e-5


# ---------------------------------------------------------------------------
# FUNCTION: test_double_well_even_odd_mismatch_scans_differ
# ---------------------------------------------------------------------------
def test_double_well_even_odd_mismatch_scans_differ() -> None:
    """
    Check that even and odd double-well mismatch scans are parity-specific.
    """
    x_half = np.linspace(0.0, 4.0, 800)
    V_half = quartic_double_well(x_half, a=1.0, b=6.0, shift_min_to_zero=True)
    energies_even, mismatches_even = sample_boundary_mismatch(
        x_half, V_half, parity="even", e_min=1.0, e_max=8.5, n_scan=400
    )
    energies_odd, mismatches_odd = sample_boundary_mismatch(
        x_half, V_half, parity="odd", e_min=1.0, e_max=8.5, n_scan=400
    )
    assert np.allclose(energies_even, energies_odd)
    assert not np.allclose(mismatches_even, mismatches_odd)


# ---------------------------------------------------------------------------
# FUNCTION: test_scattering_probability_conservation
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
# ---------------------------------------------------------------------------
def run_all_tests() -> None:
    """
    Execute all project validation tests and print a short summary.
    """
    test_normalization()
    test_derivative_at_right_edge_polynomial()
    test_find_rk4_brackets_accepts_exact_zero_hit()
    test_square_well_ground_state()
    test_square_well_convergence_order()
    test_harmonic_oscillator_first_levels()
    test_harmonic_oscillator_inward_decay_first_levels()
    test_double_well_splitting_positive()
    test_scattering_probability_conservation()
    test_run_solver_import_without_experiment_dependencies()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
