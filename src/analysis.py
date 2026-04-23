from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.shooting import solve_symmetric_potential


def exact_square_well_energies(n_values: np.ndarray, a: float = 1.0) -> np.ndarray:
    """
    Infinite square well on [-a, a] with V=0 inside and psi(±a)=0.
    In units hbar=m=1:
        E_n = (n*pi/(2a))^2 / 2
    """
    n_values = np.asarray(n_values, dtype=float)
    return (n_values * np.pi / (2.0 * a)) ** 2 / 2.0


def exact_harmonic_oscillator_energies(n_values: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """
    Harmonic oscillator with frequency omega.
    In units hbar=m=1:
        E_n = hbar * omega * (n + 0.5)
    """
    n_values = np.asarray(n_values, dtype=float)
    return omega * (n_values + 0.5)


def relative_error(numerical: np.ndarray, exact: np.ndarray) -> np.ndarray:
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return np.abs((numerical - exact) / exact)


def save_energy_table(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convergence_vs_grid(
    potential_fn,
    potential_kwargs: dict,
    x_max: float,
    grid_sizes: list[int],
    exact_energies: np.ndarray,
    n_even: int,
    n_odd: int,
    e_min: float,
    e_max: float,
) -> dict[str, np.ndarray]:
    hs = []
    energy_errors = []

    n_states = len(exact_energies)

    for n_grid in grid_sizes:
        states = solve_symmetric_potential(
            x_max=x_max,
            n_grid=n_grid,
            potential_fn=potential_fn,
            potential_kwargs=potential_kwargs,
            n_even=n_even,
            n_odd=n_odd,
            e_min=e_min,
            e_max=e_max,
        )
        energies = np.array([s.energy for s in states[:n_states]])
        hs.append(x_max / (n_grid - 1))
        energy_errors.append(np.abs(energies - exact_energies))

    return {
        "h": np.array(hs),
        "energy_errors": np.array(energy_errors),
    }


def convergence_vs_box_size(
    potential_fn,
    potential_kwargs: dict,
    x_max_values: list[float],
    n_grid: int,
    exact_energies: np.ndarray,
    n_even: int,
    n_odd: int,
    e_min: float,
    e_max: float,
) -> dict[str, np.ndarray]:
    xs = []
    energy_errors = []

    n_states = len(exact_energies)

    for x_max in x_max_values:
        states = solve_symmetric_potential(
            x_max=x_max,
            n_grid=n_grid,
            potential_fn=potential_fn,
            potential_kwargs=potential_kwargs,
            n_even=n_even,
            n_odd=n_odd,
            e_min=e_min,
            e_max=e_max,
        )
        energies = np.array([s.energy for s in states[:n_states]])
        xs.append(x_max)
        energy_errors.append(np.abs(energies - exact_energies))

    return {
        "x_max": np.array(xs),
        "energy_errors": np.array(energy_errors),
    }