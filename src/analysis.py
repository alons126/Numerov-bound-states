from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.shooting import StateSolution, solve_symmetric_potential


def exact_square_well_energies(n_values: np.ndarray, a: float = 1.0) -> np.ndarray:
    n_values = np.asarray(n_values, dtype=float)
    return (n_values * np.pi / (2.0 * a)) ** 2 / 2.0


def exact_harmonic_oscillator_energies(
    n_values: np.ndarray, omega: float = 1.0
) -> np.ndarray:
    n_values = np.asarray(n_values, dtype=float)
    return omega * (n_values + 0.5)


def relative_error(numerical: np.ndarray, exact: np.ndarray) -> np.ndarray:
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return np.abs((numerical - exact) / exact)


def save_csv_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def energies_from_states(
    states: list[StateSolution], n_states: int | None = None
) -> np.ndarray:
    if n_states is None:
        return np.array([s.energy for s in states], dtype=float)
    return np.array([s.energy for s in states[:n_states]], dtype=float)


def convergence_vs_grid(
    potential_fn,
    potential_kwargs: dict,
    x_max: float,
    grid_sizes: list[int],
    n_even: int,
    n_odd: int,
    e_min: float,
    e_max: float,
    reference_energies: np.ndarray,
) -> dict[str, np.ndarray]:
    hs = []
    errors = []
    n_states = len(reference_energies)

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
        
        energies = energies_from_states(states, n_states=n_states)
        hs.append(x_max / (n_grid - 1))
        errors.append(np.abs(energies - reference_energies))

    return {"h": np.array(hs), "energy_errors": np.array(errors)}


def convergence_vs_box_size(
    potential_fn,
    potential_kwargs: dict,
    x_max_values: list[float],
    n_grid: int,
    n_even: int,
    n_odd: int,
    e_min: float,
    e_max: float,
    reference_energies: np.ndarray,
) -> dict[str, np.ndarray]:
    xs = []
    errors = []
    n_states = len(reference_energies)

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
        
        energies = energies_from_states(states, n_states=n_states)
        xs.append(x_max)
        errors.append(np.abs(energies - reference_energies))

    return {"x_max": np.array(xs), "energy_errors": np.array(errors)}


def splitting_vs_parameter(
    potential_fn,
    base_kwargs: dict,
    varied_param: str,
    varied_values: list[float],
    x_max: float,
    n_grid: int,
    e_min: float,
    e_max: float,
) -> list[dict]:
    rows: list[dict] = []
    
    for value in varied_values:
        kwargs = dict(base_kwargs)
        kwargs[varied_param] = value
        states = solve_symmetric_potential(
            x_max=x_max,
            n_grid=n_grid,
            potential_fn=potential_fn,
            potential_kwargs=kwargs,
            n_even=2,
            n_odd=2,
            e_min=e_min,
            e_max=e_max,
        )
        
        e0 = states[0].energy
        e1 = states[1].energy
        
        rows.append(
            {
                varied_param: value,
                "E0": e0,
                "E1": e1,
                "splitting": e1 - e0,
            }
        )
        
    return rows
