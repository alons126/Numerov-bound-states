from __future__ import annotations

"""
Potential models used in the Numerov project.

All functions in this module accept a NumPy array x and return the potential
evaluated at the same locations. Keeping potentials separate from the solver
makes it easy to test multiple physical systems with the same numerical method.

This module is the "physics definition" layer of the project. The solvers in
`src/shooting.py`, `src/numerov.py`, and `src/scattering.py` do not know which
system is being solved; they only require sampled values of V(x). That
separation is intentional because it lets the same numerical machinery be
validated on analytic benchmarks and then reused for less trivial systems.

The potentials play distinct roles in the report:
- infinite square well: basic analytic validation of bound-state shooting and
  convergence-order checks
- harmonic oscillator: analytic validation plus inward-shooting stability study
- finite square well: finite-depth bound-state example showing that only a
  finite number of bound levels survive below the barrier
- quartic double well: tunneling splitting and convergence analysis
- square and double barriers: scattering and resonant-tunneling extension

The most important implementation choice here is in `quartic_double_well()`.
When `shift_min_to_zero=True`, the code subtracts the analytic minimum
`-b^2/(4a)` instead of the sampled grid minimum. That keeps the physical
potential fixed while the grid changes, which is necessary for meaningful
grid-convergence studies.
"""

import numpy as np


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FUNCTION: harmonic_oscillator
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def harmonic_oscillator(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """
    Harmonic-oscillator potential V(x) = 1/2 * omega^2 * x^2.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    omega : float, optional
        Oscillator frequency in dimensionless units.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """

    # Evaluate the analytic formula pointwise on the supplied grid.
    return 0.5 * omega**2 * x**2


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FUNCTION: infinite_square_well_numeric
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def infinite_square_well_numeric(
    x: np.ndarray,
    a: float = 1.0,
    wall_height: float = 1.0e6,
) -> np.ndarray:
    """
    Numerical approximation to an infinite square well.

    The well is zero on |x| <= a and replaced by a very large finite barrier
    outside the well so it can be treated with the same finite-grid solver.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    a : float, optional
        Half-width of the well.
    wall_height : float, optional
        Large barrier used to mimic an infinite wall.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """

    # Replace the ideal infinite wall by a very large finite value so the same
    # grid-based solver can still be used numerically.
    return np.where(np.abs(x) <= a, 0.0, wall_height)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FUNCTION: finite_square_well
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def finite_square_well(
    x: np.ndarray,
    V0: float = 20.0,
    a: float = 1.0,
) -> np.ndarray:
    """
    Finite square well with barrier height V0 outside |x| <= a.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    V0 : float, optional
        Barrier height outside the well.
    a : float, optional
        Half-width of the well.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """

    # Inside the well the potential is the chosen zero reference level; outside
    # it jumps to the barrier height V0.
    return np.where(np.abs(x) <= a, 0.0, V0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FUNCTION: quartic_double_well
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def quartic_double_well(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 6.0,
    shift_min_to_zero: bool = True,
) -> np.ndarray:
    """
    Quartic double-well potential V(x) = a x^4 - b * x^2.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    a : float, optional
        Quartic coefficient controlling overall growth at large |x|.
    b : float, optional
        Quadratic coefficient controlling the barrier / well separation.
    shift_min_to_zero : bool, optional
        If True, subtract the minimum value so the well bottoms sit at zero.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """

    # The quartic term confines the particle at large |x|, while the negative
    # quadratic term carves out the central barrier and two minima.
    v = a * x**4 - b * x**2

    if shift_min_to_zero:
        # Use the analytic minimum rather than the sampled grid minimum. The
        # latter drifts with h and contaminates convergence studies by changing
        # the potential itself as the grid is refined.
        if a <= 0.0:
            raise ValueError("quartic_double_well requires a > 0.")

        v_min = -(b**2) / (4.0 * a) if b > 0.0 else 0.0
        v = v - v_min

    return v


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FUNCTION: square_barrier
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def square_barrier(
    x: np.ndarray,
    V0: float = 5.0,
    width: float = 1.0,
    center: float = 0.0,
) -> np.ndarray:
    """
    Rectangular scattering barrier.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    V0 : float, optional
        Barrier height.
    width : float, optional
        Barrier width.
    center : float, optional
        Barrier center.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """

    half_width = 0.5 * width

    # Mark points inside the barrier interval with height V0 and leave the
    # asymptotic free regions at zero.
    return np.where(np.abs(x - center) <= half_width, V0, 0.0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FUNCTION: double_square_barrier
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def double_square_barrier(
    x: np.ndarray,
    V0: float = 5.0,
    barrier_width: float = 0.6,
    well_width: float = 1.2,
    center: float = 0.0,
) -> np.ndarray:
    """
    Symmetric double-barrier scattering potential.

    The potential is zero outside the barriers and in the central well. Two
    rectangular barriers of height V0 are separated by a well of width
    ``well_width``. This is the standard structure used to demonstrate resonant
    tunneling.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    V0 : float, optional
        Barrier height.
    barrier_width : float, optional
        Width of each barrier.
    well_width : float, optional
        Separation between the two barriers.
    center : float, optional
        Center of the full double-barrier structure.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """

    left_center = center - 0.5 * (well_width + barrier_width)
    right_center = center + 0.5 * (well_width + barrier_width)
    left_barrier = np.abs(x - left_center) <= 0.5 * barrier_width
    right_barrier = np.abs(x - right_center) <= 0.5 * barrier_width

    return np.where(left_barrier | right_barrier, V0, 0.0)
