from __future__ import annotations

"""
Minimal automated tests for the Numerov project.

These tests are designed for project validation rather than full software
engineering coverage: they check normalization, analytic benchmark energies,
and one qualitative physical feature of the double well.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.analysis import exact_harmonic_oscillator_energies, exact_square_well_energies
from src.numerov import normalize_wavefunction
from src.potentials import (
    double_square_barrier,
    harmonic_oscillator,
    infinite_square_well_numeric,
    quartic_double_well,
)
from src.scattering import sweep_scattering
from src.shooting import solve_symmetric_potential


def test_normalization() -> None:
    """
    Check that the wavefunction normalization helper produces unit norm.
    """
    x = np.linspace(-1.0, 1.0, 1001)
    psi = np.exp(-(x**2))
    psi_n = normalize_wavefunction(x, psi)
    val = np.trapezoid(np.abs(psi_n) ** 2, x)
    assert abs(val - 1.0) < 1e-10


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
def run_all_tests() -> None:
    """
    Execute all project validation tests and print a short summary.
    """
    test_normalization()
    test_square_well_ground_state()
    test_harmonic_oscillator_first_levels()
    test_double_well_splitting_positive()
    test_scattering_probability_conservation()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
