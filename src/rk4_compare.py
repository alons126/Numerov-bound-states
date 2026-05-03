from __future__ import annotations

"""
Runge-Kutta comparison routines for the harmonic oscillator.

This module is separate from the main Numerov solver. It solves the same
harmonic-oscillator shooting problem with a fourth-order Runge-Kutta integrator
so the project can compare a general ODE method with the specialized Numerov
scheme.

This module exists to make one methodological comparison precise: how does the
specialized Numerov integrator compare with a standard general-purpose RK4
integrator on the same physical problem?

The harmonic oscillator is used because it has:
- an exact spectrum, so errors can be measured directly
- a smooth potential, so the comparison is not dominated by discontinuities
- a natural inward-shooting formulation, matching the stable Numerov treatment

RK4 is implemented as a first-order system for `[psi, psi']`, whereas Numerov
works directly with the second-order equation. That distinction is one reason
the derivative boundary treatment matters so much for Numerov: RK4 carries
`psi'` explicitly, while Numerov must reconstruct it when parity conditions
require a derivative mismatch.
"""

from dataclasses import dataclass

import numpy as np

# Keep this module free of plotting dependencies. The RK4 comparison routines
# only need NumPy plus the exact harmonic-oscillator benchmark helper, so
# importing this file does not pull in `matplotlib` or the experiments layer.
from src.analysis import exact_harmonic_oscillator_energies


# ===========================================================================
# DATA CLASS: RK4EnergyResult
# ===========================================================================
@dataclass
class RK4EnergyResult:
    """One RK4 eigenvalue result for the harmonic oscillator."""

    state_index: int
    parity: str
    energy: float
    exact_energy: float
    absolute_error: float
    relative_error: float


# ===========================================================================
# FUNCTION: _harmonic_rhs
# ===========================================================================
def _harmonic_rhs(x: float, y: np.ndarray, energy: float, omega: float) -> np.ndarray:
    """
    Right-hand side for the first-order Schrödinger system.

    The second-order equation psi'' = 2[V(x)-E] psi is written as
    y = [psi, phi], where phi = psi'.

    Parameters
    ----------
    x : float
        Spatial coordinate at which the harmonic-oscillator system is
        evaluated.
    y : ndarray
        Two-component state vector ``[psi, psi']``.
    energy : float
        Trial energy used in the Schrödinger equation.
    omega : float
        Harmonic-oscillator frequency in ``V(x) = 1/2 * omega^2 * x^2``.

    Returns
    -------
    ndarray
        First-order derivative vector for the RK4 update.
    """

    # RK4 evolves a first-order system, so the second component explicitly
    # stores phi = psi'. This is the key contrast with Numerov.
    psi, phi = y
    potential = 0.5 * omega**2 * x**2

    return np.array([phi, 2.0 * (potential - energy) * psi], dtype=float)


# ===========================================================================
# FUNCTION: RK4_step
# ===========================================================================
def RK4_step(
    x: float, y: np.ndarray, h: float, energy: float, omega: float
) -> np.ndarray:
    """
    Advance the first-order Schrödinger system by one RK4 step.

    Parameters
    ----------
    x : float
        Current spatial coordinate.
    y : ndarray
        Current two-component state vector ``[psi, psi']``.
    h : float
        Step size in ``x``. This may be negative during inward shooting.
    energy : float
        Trial energy used in the Schrödinger equation.
    omega : float
        Harmonic-oscillator frequency.

    Returns
    -------
    ndarray
        Updated state vector after one classical fourth-order Runge-Kutta step.
    """

    # Four slope evaluations give the classical fourth-order Runge-Kutta step.
    # Each intermediate stage samples the RHS at a different point inside the
    # current interval, then combines them into one O(h^5) local update.
    k1 = _harmonic_rhs(x, y, energy, omega)
    k2 = _harmonic_rhs(x + 0.5 * h, y + 0.5 * h * k1, energy, omega)
    k3 = _harmonic_rhs(x + 0.5 * h, y + 0.5 * h * k2, energy, omega)
    k4 = _harmonic_rhs(x + h, y + h * k3, energy, omega)

    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ===========================================================================
# FUNCTION: RK4_inward_mismatch
# ===========================================================================
def RK4_inward_mismatch(
    energy: float,
    parity: str,
    x_max: float,
    n_grid: int,
    omega: float = 1.0,
) -> float:
    """
    Compute the parity mismatch using inward RK4 shooting.

    The integration starts at x_max in the forbidden region with the asymptotic
    decaying condition psi'/psi ~= -sqrt(2[V(x_max)-E]) and proceeds inward to
    x=0. Even states require psi'(0)=0, while odd states require psi(0)=0.

    Parameters
    ----------
    energy : float
        Trial energy at which the inward-shooting mismatch is evaluated.
    parity : str
        Symmetry sector of the trial state, either ``"even"`` or ``"odd"``.
    x_max : float
        Outer truncation point of the half-domain.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    omega : float, default=1.0
        Harmonic-oscillator frequency.

    Returns
    -------
    float
        Origin mismatch used for root finding: ``psi'(0)`` for even states or
        ``psi(0)`` for odd states.
    """

    if parity not in {"even", "odd"}:
        raise ValueError("parity must be 'even' or 'odd'.")

    if n_grid < 3:
        raise ValueError("n_grid must be at least 3.")

    # The grid is decreasing because this is inward shooting: start in the
    # asymptotic forbidden region and integrate toward the symmetry point.
    x_values = np.linspace(x_max, 0.0, n_grid)
    h = x_values[1] - x_values[0]

    # Estimate the forbidden-region decay rate at the starting edge.
    potential_edge = 0.5 * omega**2 * x_max**2
    kappa = np.sqrt(max(2.0 * (potential_edge - energy), 1.0e-14))

    y = np.array([1.0, -kappa], dtype=float)
    for x_value in x_values[:-1]:
        # The grid spacing `h` is negative here because x decreases from x_max
        # toward the origin, so the standard RK4 step automatically marches
        # inward without any separate reversal logic.
        y = RK4_step(x_value, y, h, energy, omega)

    psi_at_zero, derivative_at_zero = y

    if parity == "even":
        return float(derivative_at_zero)

    return float(psi_at_zero)


# ===========================================================================
# FUNCTION: RK4_diagnostic_mismatch
# ===========================================================================
def RK4_diagnostic_mismatch(
    energy: float,
    parity: str,
    x_max: float,
    n_grid: int,
    omega: float = 1.0,
) -> float:
    """
    Compute a scale-invariant RK4 mismatch for diagnostic plots.

    The inward RK4 trial solution is initialized with an arbitrary tail
    amplitude, so the raw origin mismatch can become very large away from the
    roots. For plotting only, divide the mismatch by the half-domain ``L2``
    norm of the trial wavefunction. This keeps the zeros fixed while making
    the diagnostic curve easier to compare with the Numerov plots.
    """

    if parity not in {"even", "odd"}:
        raise ValueError("parity must be 'even' or 'odd'.")

    if n_grid < 3:
        raise ValueError("n_grid must be at least 3.")

    x_values = np.linspace(x_max, 0.0, n_grid)
    h = x_values[1] - x_values[0]
    potential_edge = 0.5 * omega**2 * x_max**2
    kappa = np.sqrt(max(2.0 * (potential_edge - energy), 1.0e-14))

    y = np.array([1.0, -kappa], dtype=float)
    psi_values = np.empty(n_grid, dtype=float)
    psi_values[0] = y[0]
    for i, x_value in enumerate(x_values[:-1], start=1):
        y = RK4_step(x_value, y, h, energy, omega)
        psi_values[i] = y[0]

    scale = max(
        float(np.sqrt(np.trapezoid(np.abs(psi_values[::-1]) ** 2, x_values[::-1]))),
        1.0e-300,
    )
    psi_at_zero, derivative_at_zero = y

    if parity == "even":
        return float(derivative_at_zero / scale)

    return float(psi_at_zero / scale)


# ===========================================================================
# FUNCTION: RK4_find_brackets
# ===========================================================================
def RK4_find_brackets(
    parity: str,
    x_max: float,
    n_grid: int,
    e_min: float,
    e_max: float,
    omega: float = 1.0,
    # Use a denser scan so RK4 brackets are as accurate as Numerov ones
    n_scan: int = 800,
) -> list[tuple[float, float]]:
    """
    Locate sign-changing energy brackets for RK4 inward shooting.

    Parameters
    ----------
    parity : str
        State parity whose inward-shooting mismatch is scanned, either
        ``"even"`` or ``"odd"``.
    x_max : float
        Outer truncation point of the half-domain.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    e_min : float
        Lower end of the trial-energy scan interval.
    e_max : float
        Upper end of the trial-energy scan interval.
    omega : float, default=1.0
        Harmonic-oscillator frequency.
    n_scan : int, default=800
        Number of sampled trial energies used to detect sign-changing
        intervals.

    Returns
    -------
    list[tuple[float, float]]
        Energy brackets containing mismatch zeros that can be refined by
        bisection.
    """

    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [RK4_inward_mismatch(e, parity, x_max, n_grid, omega) for e in energies],
        dtype=float,
    )

    brackets: list[tuple[float, float]] = []
    for i in range(len(energies) - 1):
        a = mismatches[i]
        b = mismatches[i + 1]

        if not np.isfinite(a) or not np.isfinite(b):
            continue

        if a == 0.0:
            # Preserve exact scan hits as tiny brackets for uniform handling by
            # the downstream bisection solver.
            eps = 1e-10 * max(1.0, abs(energies[i]))
            brackets.append((float(energies[i] - eps), float(energies[i] + eps)))
        elif np.signbit(a) != np.signbit(b):
            brackets.append((float(energies[i]), float(energies[i + 1])))

    return brackets


# ===========================================================================
# FUNCTION: RK4_sample_mismatch
# ===========================================================================
def RK4_sample_mismatch(
    parity: str,
    x_max: float,
    n_grid: int,
    e_min: float,
    e_max: float,
    omega: float = 1.0,
    n_scan: int = 1000,
    diagnostic_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the RK4 inward-shooting mismatch over an energy interval.

    Parameters
    ----------
    parity : str
        State parity whose mismatch curve is sampled, either ``"even"`` or
        ``"odd"``.
    x_max : float
        Outer truncation point of the half-domain.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    e_min : float
        Lower end of the trial-energy scan interval.
    e_max : float
        Upper end of the trial-energy scan interval.
    omega : float, default=1.0
        Harmonic-oscillator frequency.
    n_scan : int, default=1000
        Number of equally spaced trial energies used to sample the mismatch
        curve.
    diagnostic_scale : bool, default=False
        If ``True``, return a scale-invariant mismatch intended for plotting.

    Returns
    -------
    tuple[ndarray, ndarray]
        Pair ``(energies, mismatches)`` containing the sampled energies and the
        corresponding inward-shooting mismatch values.
    """

    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [
            RK4_diagnostic_mismatch(e, parity, x_max, n_grid, omega)
            if diagnostic_scale
            else RK4_inward_mismatch(e, parity, x_max, n_grid, omega)
            for e in energies
        ],
        dtype=float,
    )

    return energies, mismatches


# ===========================================================================
# FUNCTION: RK4_bisect_energy
# ===========================================================================
def RK4_bisect_energy(
    parity: str,
    bracket: tuple[float, float],
    x_max: float,
    n_grid: int,
    omega: float = 1.0,
    tol: float = 1.0e-12,
    max_iter: int = 120,
) -> float:
    """
    Refine one RK4 shooting bracket with bisection.

    Parameters
    ----------
    parity : str
        State parity whose mismatch root is being refined, either ``"even"``
        or ``"odd"``.
    bracket : tuple[float, float]
        Energy interval ``(lo, hi)`` that must already contain a sign change in
        the RK4 inward-shooting mismatch.
    x_max : float
        Outer truncation point of the half-domain.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    omega : float, default=1.0
        Harmonic-oscillator frequency.
    tol : float, default=1.0e-12
        Stopping tolerance applied to both the bracket width and mismatch
        magnitude.
    max_iter : int, default=120
        Maximum number of bisection iterations.

    Returns
    -------
    float
        Refined RK4 eigenvalue estimate.
    """

    lo, hi = bracket
    flo = RK4_inward_mismatch(lo, parity, x_max, n_grid, omega)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        fmid = RK4_inward_mismatch(mid, parity, x_max, n_grid, omega)

        if abs(hi - lo) < tol or abs(fmid) < tol:
            return float(mid)

        if np.signbit(flo) != np.signbit(fmid):
            hi = mid
        else:
            lo = mid
            flo = fmid

    return float(0.5 * (lo + hi))


# ===========================================================================
# FUNCTION: RK4_bisection_history
# ===========================================================================
def RK4_bisection_history(
    parity: str,
    bracket: tuple[float, float],
    x_max: float,
    n_grid: int,
    omega: float = 1.0,
    tol: float = 1.0e-12,
    max_iter: int = 80,
    diagnostic_scale: bool = False,
) -> list[dict]:
    """
    Record the RK4 inward-shooting bisection process for one bracket.

    Parameters
    ----------
    parity : str
        State parity whose mismatch root is being traced, either ``"even"``
        or ``"odd"``.
    bracket : tuple[float, float]
        Energy interval ``(lo, hi)`` that brackets a sign change in the RK4
        inward-shooting mismatch.
    x_max : float
        Outer truncation point of the half-domain.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    omega : float, default=1.0
        Harmonic-oscillator frequency.
    tol : float, default=1.0e-12
        Stopping tolerance applied to the midpoint mismatch and bracket width.
    max_iter : int, default=80
        Maximum number of recorded bisection iterations.
    diagnostic_scale : bool, default=False
        If ``True``, record a scale-invariant mismatch for plotting while the
        bisection itself still uses the raw mismatch signs.

    Returns
    -------
    list[dict]
        Per-iteration diagnostics containing the current bracket endpoints,
        midpoint, and midpoint mismatch.
    """

    lo, hi = bracket
    flo = RK4_inward_mismatch(lo, parity, x_max, n_grid, omega)

    history: list[dict] = []
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)

        fmid = RK4_inward_mismatch(mid, parity, x_max, n_grid, omega)
        history.append(
            {
                "iteration": iteration,
                "lo": lo,
                "hi": hi,
                "mid": mid,
                "mismatch_mid": (
                    RK4_diagnostic_mismatch(mid, parity, x_max, n_grid, omega)
                    if diagnostic_scale
                    else fmid
                ),
            }
        )

        if abs(hi - lo) < tol or abs(fmid) < tol:
            break

        if np.signbit(flo) != np.signbit(fmid):
            hi = mid
        else:
            lo = mid
            flo = fmid

    return history


# ===========================================================================
# FUNCTION: RK4_solve_harmonic_oscillator_energies
# ===========================================================================
def RK4_solve_harmonic_oscillator_energies(
    x_max: float,
    n_grid: int,
    n_states: int = 4,
    omega: float = 1.0,
    e_min: float = 0.1,
    e_max: float = 6.5,
) -> list[RK4EnergyResult]:
    """
    Compute the lowest harmonic-oscillator energies using RK4 shooting.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    n_states : int, default=4
        Number of lowest harmonic-oscillator states to compute.
    omega : float, default=1.0
        Harmonic-oscillator frequency.
    e_min : float, default=0.1
        Lower end of the trial-energy scan interval used for bracketing.
    e_max : float, default=6.5
        Upper end of the trial-energy scan interval used for bracketing.

    Returns
    -------
    list[RK4EnergyResult]
        Lowest RK4-computed harmonic-oscillator states together with exact
        energies and error diagnostics.
    """

    exact = exact_harmonic_oscillator_energies(np.arange(n_states), omega=omega)
    results: list[RK4EnergyResult] = []

    for parity, state_offset in [("even", 0), ("odd", 1)]:
        # Even and odd harmonic-oscillator states interleave in energy, so each
        # parity branch maps to every other global state index.
        brackets = RK4_find_brackets(
            parity=parity,
            x_max=x_max,
            n_grid=n_grid,
            e_min=e_min,
            e_max=e_max,
            omega=omega,
        )

        for local_index, bracket in enumerate(brackets):
            state_index = state_offset + 2 * local_index

            if state_index >= n_states:
                break

            energy = RK4_bisect_energy(
                parity=parity,
                bracket=bracket,
                x_max=x_max,
                n_grid=n_grid,
                omega=omega,
            )
            exact_energy = float(exact[state_index])
            absolute_error = abs(energy - exact_energy)
            results.append(
                RK4EnergyResult(
                    state_index=state_index,
                    parity=parity,
                    energy=energy,
                    exact_energy=exact_energy,
                    absolute_error=absolute_error,
                    relative_error=absolute_error / abs(exact_energy),
                )
            )

    results.sort(key=lambda row: row.state_index)

    if len(results) < n_states:
        raise RuntimeError(
            f"RK4 comparison found only {len(results)} states, needed {n_states}."
        )

    return results[:n_states]


# ===========================================================================
# FUNCTION: RK4_harmonic_convergence_vs_grid
# ===========================================================================
def RK4_harmonic_convergence_vs_grid(
    x_max: float,
    grid_sizes: list[int],
    n_states: int = 4,
    omega: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Return RK4 harmonic-oscillator energies and errors for several grids.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used in every solve.
    grid_sizes : list[int]
        Sequence of half-domain grid sizes used for the convergence study.
    n_states : int, default=4
        Number of lowest states to include for each grid.
    omega : float, default=1.0
        Harmonic-oscillator frequency.

    Returns
    -------
    dict[str, ndarray]
        Dictionary containing the grid spacings, computed energies, and
        absolute energy errors for each sampled grid.
    """

    h_values = []
    errors = []
    energies = []

    for n_grid in grid_sizes:
        # Use the same post-processed output format as the Numerov convergence
        # routines so the comparison layer can treat both methods uniformly.
        rows = RK4_solve_harmonic_oscillator_energies(
            x_max=x_max,
            n_grid=n_grid,
            n_states=n_states,
            omega=omega,
        )
        h_values.append(x_max / (n_grid - 1))
        energies.append([row.energy for row in rows])
        errors.append([row.absolute_error for row in rows])

    return {
        "h": np.array(h_values, dtype=float),
        "energies": np.array(energies, dtype=float),
        "energy_errors": np.array(errors, dtype=float),
    }


# ===========================================================================
# FUNCTION: RK4_harmonic_convergence_vs_box_size_fixed_spacing
# ===========================================================================
def RK4_harmonic_convergence_vs_box_size_fixed_spacing(
    x_max_values: list[float],
    target_h: float,
    n_states: int = 4,
    omega: float = 1.0,
    e_min: float = 0.1,
    e_max: float = 6.5,
) -> dict[str, np.ndarray]:
    """
    Return RK4 harmonic-oscillator errors versus box size at fixed spacing.

    Parameters
    ----------
    x_max_values : list[float]
        Sequence of outer truncation points used in the box-size study.
    target_h : float
        Desired grid spacing. Each solve uses the nearest integer grid size
        that approximates this spacing.
    n_states : int, default=4
        Number of lowest states to include for each box size.
    omega : float, default=1.0
        Harmonic-oscillator frequency.
    e_min : float, default=0.1
        Lower end of the trial-energy scan interval used for bracketing.
    e_max : float, default=6.5
        Upper end of the trial-energy scan interval used for bracketing.

    Returns
    -------
    dict[str, ndarray]
        Dictionary containing sampled box sizes, achieved spacings, grid
        sizes, computed energies, and absolute energy errors.
    """

    exact = exact_harmonic_oscillator_energies(np.arange(n_states), omega=omega)

    x_values = []
    h_values = []
    grid_sizes = []
    energies = []
    errors = []

    for x_max in x_max_values:
        # Match the requested spacing as closely as an integer grid allows.
        n_grid = int(round(x_max / target_h)) + 1
        n_grid = max(n_grid, 3)
        actual_h = x_max / (n_grid - 1)

        rows = RK4_solve_harmonic_oscillator_energies(
            x_max=x_max,
            n_grid=n_grid,
            n_states=n_states,
            omega=omega,
            e_min=e_min,
            e_max=e_max,
        )
        row_energies = np.array([row.energy for row in rows], dtype=float)

        x_values.append(x_max)
        h_values.append(actual_h)
        grid_sizes.append(n_grid)
        energies.append(row_energies)
        errors.append(np.abs(row_energies - exact))

    return {
        "x_max": np.array(x_values, dtype=float),
        "h": np.array(h_values, dtype=float),
        "n_grid": np.array(grid_sizes, dtype=int),
        "energies": np.array(energies, dtype=float),
        "energy_errors": np.array(errors, dtype=float),
    }
