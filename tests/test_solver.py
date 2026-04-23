from __future__ import annotations

import numpy as np

from src.analysis import exact_harmonic_oscillator_energies, exact_square_well_energies
from src.numerov import normalize_wavefunction
from src.potentials import harmonic_oscillator, infinite_square_well_numeric
from src.shooting import solve_symmetric_potential


def test_normalization() -> None:
    x = np.linspace(-1.0, 1.0, 1001)
    psi = np.exp(-x**2)
    psi_n = normalize_wavefunction(x, psi)
    val = np.trapezoid(np.abs(psi_n) ** 2, x)
    assert abs(val - 1.0) < 1e-10


def test_square_well_ground_state() -> None:
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


def run_all_tests() -> None:
    test_normalization()
    test_square_well_ground_state()
    test_harmonic_oscillator_first_levels()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()