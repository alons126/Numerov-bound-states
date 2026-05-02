from __future__ import annotations

"""
Analysis utilities for validation and convergence studies.

This module collects functions that turn raw solver output into quantities used
in the report: exact benchmark spectra, error tables, convergence trends, and
double-well splitting data.

This module is the scientific validation layer. It is what turns the project
from "the code runs" into "the numerical claims are checked."

The functions here fall into three groups:
- Exact benchmark helpers for problems with known spectra
- Convergence helpers for studying discretization and truncation error
- Parameter-sweep helpers for extracting physical trends such as double-well
  tunneling splitting
"""

import csv
from pathlib import Path

import numpy as np

# Import the solver's state container for type annotations and the default
# outward-shooting routine used by the convergence/parameter-sweep helpers.
from src.shooting import StateSolution, solve_symmetric_potential_outward_shooting


# ---------------------------------------------------------------------------
# FUNCTION: exact_square_well_energies
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FUNCTION: exact_harmonic_oscillator_energies
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FUNCTION: relative_error
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FUNCTION: estimate_convergence_slopes
# ---------------------------------------------------------------------------
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
        # Ignore non-positive or non-finite entries before taking logarithms.
        valid = (
            np.isfinite(xvals)
            & np.isfinite(errors[:, i])
            & (xvals > 0.0)
            & (errors[:, i] > 0.0)
        )

        if np.count_nonzero(valid) < 2:
            slope = np.nan
            intercept = np.nan
        else:
            slope, intercept = np.polyfit(
                np.log(xvals[valid]), np.log(errors[valid, i]), 1
            )

        rows.append(
            {
                "state_index": i + state_offset,
                "convergence_exponent_p": slope,
                "log_prefactor": intercept,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# FUNCTION: save_csv_rows
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FUNCTION: energies_from_states
# ---------------------------------------------------------------------------
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

    if len(states) < n_states:
        raise ValueError(
            f"Requested {n_states} state energies, but solver returned only "
            f"{len(states)} states."
        )

    # States are already sorted by energy by the solver, so slicing keeps the
    # physically lowest levels in order.
    return np.array([s.energy for s in states[:n_states]], dtype=float)


# ---------------------------------------------------------------------------
# FUNCTION: convergence_vs_grid
# ---------------------------------------------------------------------------
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
    solver_fn=solve_symmetric_potential_outward_shooting,
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
        # Each loop entry reruns the full eigenvalue search on a new mesh, then
        # compares the resulting low-lying spectrum to the supplied reference.
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


# ---------------------------------------------------------------------------
# FUNCTION: convergence_vs_grid_successive
# ---------------------------------------------------------------------------
def convergence_vs_grid_successive(
    potential_fn,
    potential_kwargs: dict,
    x_max: float,
    grid_sizes: list[int],
    n_even: int,
    n_odd: int,
    e_min: float,
    e_max: float,
    solver_fn=solve_symmetric_potential_outward_shooting,
) -> dict[str, np.ndarray]:
    """
    Study grid convergence using successive refinements on a fixed box.

    Instead of comparing every grid to one finite reference calculation, this
    compares each grid to the next finer grid. That avoids contamination from a
    finite-reference floor when no exact spectrum is available.

    Parameters
    ----------
    potential_fn : callable
        Potential function.
    potential_kwargs : dict
        Keyword arguments for the potential.
    x_max : float
        Computational half-domain size.
    grid_sizes : list[int]
        Grid resolutions to test, ordered from coarse to fine.
    n_even, n_odd : int
        Number of even and odd states requested.
    e_min, e_max : float
        Energy search interval.
    solver_fn : callable, optional
        Solver function used to compute states.

    Returns
    -------
    dict[str, ndarray]
        Dictionary with coarse-grid spacings h and successive-difference arrays.
    """

    if len(grid_sizes) < 2:
        raise ValueError("Need at least two grid sizes for successive convergence.")

    hs = []
    energies_by_grid = []
    n_states = n_even + n_odd

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
        energies_by_grid.append(energies_from_states(states, n_states=n_states))
        hs.append(x_max / (n_grid - 1))

    energies_arr = np.array(energies_by_grid, dtype=float)
    # Successive differences act as a stand-in error estimate when no exact
    # spectrum is available, as in the quartic double well.
    successive_errors = np.abs(energies_arr[:-1] - energies_arr[1:])

    return {"h": np.array(hs[:-1]), "energy_errors": successive_errors}


# ---------------------------------------------------------------------------
# FUNCTION: convergence_vs_box_size
# ---------------------------------------------------------------------------
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
    solver_fn=solve_symmetric_potential_outward_shooting,
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
        # Here n_grid stays fixed, so changing x_max changes both the physical
        # box size and the spacing h. This helper is mainly for simpler box
        # sensitivity checks, not clean fixed-h studies.
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


# ---------------------------------------------------------------------------
# FUNCTION: convergence_vs_box_size_fixed_spacing
# ---------------------------------------------------------------------------
def convergence_vs_box_size_fixed_spacing(
    potential_fn,
    potential_kwargs: dict,
    x_max_values: list[float],
    target_h: float,
    n_even: int,
    n_odd: int,
    e_min: float,
    e_max: float,
    reference_energies: np.ndarray,
    solver_fn=solve_symmetric_potential_outward_shooting,
) -> dict[str, np.ndarray]:
    """
    Study box-size convergence while keeping grid spacing approximately fixed.

    This is useful for the harmonic oscillator. If x_max is changed while n_grid
    is held fixed, then the grid spacing h changes too, so the plot mixes
    boundary truncation error with discretization error. Here n_grid is adjusted
    for each x_max so that h remains nearly constant and the box-size study
    mainly measures the finite-domain cutoff effect.

    Parameters
    ----------
    potential_fn : callable
        Potential function.
    potential_kwargs : dict
        Keyword arguments for the potential.
    x_max_values : list[float]
        Domain sizes to test.
    target_h : float
        Desired half-domain grid spacing.
    n_even, n_odd : int
        Number of even and odd states requested.
    e_min, e_max : float
        Energy search interval.
    reference_energies : ndarray
        Exact or high-quality reference energies.
    solver_fn : callable, optional
        Solver function used to compute states.

    Returns
    -------
    dict[str, ndarray]
        Dictionary with x_max values, actual h values, n_grid values, and
        energy error arrays.
    """

    xs = []
    hs = []
    grid_sizes = []
    errors = []
    n_states = len(reference_energies)

    for x_max in x_max_values:
        # Choose the closest integer grid that preserves the requested nominal
        # spacing, then record the actual spacing achieved after rounding.
        n_grid = int(round(x_max / target_h)) + 1
        n_grid = max(n_grid, 3)
        actual_h = x_max / (n_grid - 1)

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
        hs.append(actual_h)
        grid_sizes.append(n_grid)
        errors.append(np.abs(energies - reference_energies))

    return {
        "x_max": np.array(xs),
        "h": np.array(hs),
        "n_grid": np.array(grid_sizes),
        "energy_errors": np.array(errors),
    }


# ---------------------------------------------------------------------------
# FUNCTION: splitting_vs_parameter
# ---------------------------------------------------------------------------
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
        # Copy the baseline potential parameters, then overwrite only the one
        # being swept so each solve differs in a controlled way.
        kwargs = dict(base_kwargs)
        kwargs[varied_param] = value
        states = solve_symmetric_potential_outward_shooting(
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
        # In a symmetric double well, the lowest two states are the even/odd
        # tunneling pair, so their splitting is the key physical observable.
        rows.append(
            {
                varied_param: value,
                "E0": e0,
                "E1": e1,
                "splitting": e1 - e0,
            }
        )

    return rows
