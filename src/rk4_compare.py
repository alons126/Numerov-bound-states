from __future__ import annotations

"""
Runge-Kutta comparison routines for the harmonic oscillator.

This module is separate from the main Numerov solver. It solves the same
harmonic-oscillator shooting problem with a fourth-order Runge-Kutta integrator
so the project can compare a general ODE method with the specialized Numerov
scheme.

Reviewer guide
--------------
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

from src.analysis import exact_harmonic_oscillator_energies


# ---------------------------------------------------------------------------
# DATA CLASS: RK4EnergyResult
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
@dataclass
class RK4EnergyResult:
    """One RK4 eigenvalue result for the harmonic oscillator."""

    state_index: int
    parity: str
    energy: float
    exact_energy: float
    absolute_error: float
    relative_error: float


# ---------------------------------------------------------------------------
# FUNCTION: _harmonic_rhs
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def _harmonic_rhs(x: float, y: np.ndarray, energy: float, omega: float) -> np.ndarray:
    """
    Right-hand side for the first-order Schrodinger system.

    The second-order equation psi'' = 2[V(x)-E] psi is written as
    y = [psi, phi], where phi = psi'.
    """
    # RK4 evolves a first-order system, so the second component explicitly
    # stores phi = psi'. This is the key contrast with Numerov.
    psi, phi = y
    potential = 0.5 * omega**2 * x**2
    return np.array([phi, 2.0 * (potential - energy) * psi], dtype=float)


# ---------------------------------------------------------------------------
# FUNCTION: rk4_step
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def rk4_step(
    x: float, y: np.ndarray, h: float, energy: float, omega: float
) -> np.ndarray:
    """Advance the first-order Schrodinger system by one RK4 step."""
    # Four slope evaluations give the classical fourth-order Runge-Kutta step.
    # Each intermediate stage samples the RHS at a different point inside the
    # current interval, then combines them into one O(h^5) local update.
    k1 = _harmonic_rhs(x, y, energy, omega)
    k2 = _harmonic_rhs(x + 0.5 * h, y + 0.5 * h * k1, energy, omega)
    k3 = _harmonic_rhs(x + 0.5 * h, y + 0.5 * h * k2, energy, omega)
    k4 = _harmonic_rhs(x + h, y + h * k3, energy, omega)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# FUNCTION: rk4_inward_mismatch
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def rk4_inward_mismatch(
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
        y = rk4_step(x_value, y, h, energy, omega)

    psi_at_zero, derivative_at_zero = y
    if parity == "even":
        return float(derivative_at_zero)
    return float(psi_at_zero)


# ---------------------------------------------------------------------------
# FUNCTION: find_rk4_brackets
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def find_rk4_brackets(
    parity: str,
    x_max: float,
    n_grid: int,
    e_min: float,
    e_max: float,
    omega: float = 1.0,
    # Use a denser scan so RK4 brackets are as accurate as Numerov ones
    n_scan: int = 800,
) -> list[tuple[float, float]]:
    """Locate sign-changing energy brackets for RK4 inward shooting."""
    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [rk4_inward_mismatch(e, parity, x_max, n_grid, omega) for e in energies],
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


# ---------------------------------------------------------------------------
# FUNCTION: sample_rk4_mismatch
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def sample_rk4_mismatch(
    parity: str,
    x_max: float,
    n_grid: int,
    e_min: float,
    e_max: float,
    omega: float = 1.0,
    n_scan: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the RK4 inward-shooting mismatch over an energy interval."""
    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [rk4_inward_mismatch(e, parity, x_max, n_grid, omega) for e in energies],
        dtype=float,
    )
    return energies, mismatches


# ---------------------------------------------------------------------------
# FUNCTION: bisect_rk4_energy
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def bisect_rk4_energy(
    parity: str,
    bracket: tuple[float, float],
    x_max: float,
    n_grid: int,
    omega: float = 1.0,
    tol: float = 1.0e-12,
    max_iter: int = 120,
) -> float:
    """Refine one RK4 shooting bracket with bisection."""
    lo, hi = bracket
    flo = rk4_inward_mismatch(lo, parity, x_max, n_grid, omega)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = rk4_inward_mismatch(mid, parity, x_max, n_grid, omega)

        if abs(hi - lo) < tol or abs(fmid) < tol:
            return float(mid)

        if np.signbit(flo) != np.signbit(fmid):
            hi = mid
        else:
            lo = mid
            flo = fmid

    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# FUNCTION: bisection_history_rk4
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def bisection_history_rk4(
    parity: str,
    bracket: tuple[float, float],
    x_max: float,
    n_grid: int,
    omega: float = 1.0,
    tol: float = 1.0e-12,
    max_iter: int = 80,
) -> list[dict]:
    """Record the RK4 inward-shooting bisection process for one bracket."""
    lo, hi = bracket
    flo = rk4_inward_mismatch(lo, parity, x_max, n_grid, omega)

    history: list[dict] = []
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = rk4_inward_mismatch(mid, parity, x_max, n_grid, omega)
        history.append(
            {
                "iteration": iteration,
                "lo": lo,
                "hi": hi,
                "mid": mid,
                "mismatch_mid": fmid,
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


# ---------------------------------------------------------------------------
# FUNCTION: solve_harmonic_oscillator_rk4_energies
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def solve_harmonic_oscillator_rk4_energies(
    x_max: float,
    n_grid: int,
    n_states: int = 4,
    omega: float = 1.0,
    e_min: float = 0.1,
    e_max: float = 6.5,
) -> list[RK4EnergyResult]:
    """Compute the lowest harmonic-oscillator energies using RK4 shooting."""
    exact = exact_harmonic_oscillator_energies(np.arange(n_states), omega=omega)
    results: list[RK4EnergyResult] = []

    for parity, state_offset in [("even", 0), ("odd", 1)]:
        # Even and odd harmonic-oscillator states interleave in energy, so each
        # parity branch maps to every other global state index.
        brackets = find_rk4_brackets(
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

            energy = bisect_rk4_energy(
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


# ---------------------------------------------------------------------------
# FUNCTION: rk4_harmonic_convergence_vs_grid
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def rk4_harmonic_convergence_vs_grid(
    x_max: float,
    grid_sizes: list[int],
    n_states: int = 4,
    omega: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return RK4 harmonic-oscillator energies and errors for several grids."""
    h_values = []
    errors = []
    energies = []

    for n_grid in grid_sizes:
        # Use the same post-processed output format as the Numerov convergence
        # routines so the comparison layer can treat both methods uniformly.
        rows = solve_harmonic_oscillator_rk4_energies(
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


# ---------------------------------------------------------------------------
# FUNCTION: rk4_harmonic_convergence_vs_box_size_fixed_spacing
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def rk4_harmonic_convergence_vs_box_size_fixed_spacing(
    x_max_values: list[float],
    target_h: float,
    n_states: int = 4,
    omega: float = 1.0,
    e_min: float = 0.1,
    e_max: float = 6.5,
) -> dict[str, np.ndarray]:
    """Return RK4 harmonic-oscillator errors versus box size at fixed spacing."""
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

        rows = solve_harmonic_oscillator_rk4_energies(
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
