from __future__ import annotations

"""
Analysis utilities for validation and convergence studies.

This module collects functions that turn raw solver output into quantities used
in the report: exact benchmark spectra, error tables, convergence trends, and
double-well splitting data.
"""

import csv
from pathlib import Path

import numpy as np

from src.shooting import StateSolution, solve_symmetric_potential


def exact_square_well_energies(n_values: np.ndarray, a: float = 1.0) -> np.ndarray:
    """
    Exact energies for an infinite square well on [-a, a].

    Parameters
    ----------
    n_values : ndarray
        Quantum numbers n = 1, 2, 3, ...
    a : float, optional
        Half-width of the well.

    Returns
    -------
    ndarray
        Exact eigenvalues in the dimensionless units used by the project.
    """
    n_values = np.asarray(n_values, dtype=float)
    return (n_values * np.pi / (2.0 * a)) ** 2 / 2.0


def exact_harmonic_oscillator_energies(
    n_values: np.ndarray,
    omega: float = 1.0,
) -> np.ndarray:
    """
    Exact harmonic-oscillator spectrum E_n = omega (n + 1/2).

    Parameters
    ----------
    n_values : ndarray
        State indices n = 0, 1, 2, ...
    omega : float, optional
        Oscillator frequency.

    Returns
    -------
    ndarray
        Exact eigenvalues.
    """
    n_values = np.asarray(n_values, dtype=float)
    return omega * (n_values + 0.5)


def relative_error(numerical: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """
    Compute componentwise relative error.

    Parameters
    ----------
    numerical : ndarray
        Computed values.
    exact : ndarray
        Reference values.

    Returns
    -------
    ndarray
        Relative errors |numerical - exact| / |exact|.
    """
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return np.abs((numerical - exact) / exact)


def estimate_convergence_slopes(
    xvals: np.ndarray,
    errors: np.ndarray,
    state_offset: int = 0,
) -> list[dict]:
    """
    Estimate convergence exponents from a log-log error curve.

    If the error behaves approximately like error = C h^p, then fitting
    log(error) as a linear function of log(h) gives the slope p. This is useful
    for reporting the observed order of convergence.

    Parameters
    ----------
    xvals : ndarray
        Grid spacings h or another positive refinement parameter.
    errors : ndarray
        Error array with one column per state.
    state_offset : int, optional
        Offset added to the state index in the returned rows.

    Returns
    -------
    list[dict]
        Rows containing state_index and the fitted exponent p.
    """
    xvals = np.asarray(xvals, dtype=float)
    errors = np.asarray(errors, dtype=float)

    rows: list[dict] = []
    for i in range(errors.shape[1]):
        valid = np.isfinite(xvals) & np.isfinite(errors[:, i]) & (xvals > 0.0) & (errors[:, i] > 0.0)

        if np.count_nonzero(valid) < 2:
            slope = np.nan
            intercept = np.nan
        else:
            slope, intercept = np.polyfit(np.log(xvals[valid]), np.log(errors[valid, i]), 1)

        rows.append(
            {
                "state_index": i + state_offset,
                "convergence_exponent_p": slope,
                "log_prefactor": intercept,
            }
        )

    return rows


def save_csv_rows(path: str | Path, rows: list[dict]) -> None:
    """
    Save a list of dictionaries as a CSV table.

    Parameters
    ----------
    path : str or Path
        Output path.
    rows : list[dict]
        Table rows with matching keys.

    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def energies_from_states(
    states: list[StateSolution],
    n_states: int | None = None,
) -> np.ndarray:
    """
    Extract the first n state energies from a list of StateSolution objects.

    Parameters
    ----------
    states : list[StateSolution]
        Solver output.
    n_states : int, optional
        Number of energies to extract. If omitted, use all states.

    Returns
    -------
    ndarray
        Extracted energy values.
    """
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
    solver_fn=solve_symmetric_potential,
) -> dict[str, np.ndarray]:
    """
    Study eigenvalue convergence as the grid is refined.

    Parameters
    ----------
    potential_fn : callable
        Potential function.
    potential_kwargs : dict
        Keyword arguments for the potential.
    x_max : float
        Computational half-domain size.
    grid_sizes : list[int]
        Grid resolutions to test.
    n_even, n_odd : int
        Number of even and odd states requested.
    e_min, e_max : float
        Energy search interval.
    reference_energies : ndarray
        Exact or high-quality reference energies.

    Returns
    -------
    dict[str, ndarray]
        Dictionary with grid spacing h and energy error arrays.
    """
    hs = []
    errors = []
    n_states = len(reference_energies)

    for n_grid in grid_sizes:
        states = solver_fn(
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
    solver_fn=solve_symmetric_potential,
) -> dict[str, np.ndarray]:
    """
    Study eigenvalue convergence as the computational box size changes.

    Parameters
    ----------
    potential_fn : callable
        Potential function.
    potential_kwargs : dict
        Keyword arguments for the potential.
    x_max_values : list[float]
        Domain sizes to test.
    n_grid : int
        Fixed grid size used for the box-size study.
    n_even, n_odd : int
        Number of even and odd states requested.
    e_min, e_max : float
        Energy search interval.
    reference_energies : ndarray
        Exact or high-quality reference energies.

    Returns
    -------
    dict[str, ndarray]
        Dictionary with x_max values and energy error arrays.
    """
    xs = []
    errors = []
    n_states = len(reference_energies)

    for x_max in x_max_values:
        states = solver_fn(
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
    """
    Measure double-well ground-state splitting while varying one parameter.

    Parameters
    ----------
    potential_fn : callable
        Potential function to study.
    base_kwargs : dict
        Baseline potential parameters.
    varied_param : str
        Name of the parameter to sweep.
    varied_values : list[float]
        Parameter values used in the sweep.
    x_max : float
        Computational half-domain size.
    n_grid : int
        Number of half-domain grid points.
    e_min, e_max : float
        Energy search interval.

    Returns
    -------
    list[dict]
        Rows containing the varied parameter, E0, E1, and the splitting.
    """
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
