from __future__ import annotations

"""
Potential models used in the Numerov project.

All functions in this module accept a NumPy array x and return the potential
evaluated at the same locations. Keeping potentials separate from the solver
makes it easy to test multiple physical systems with the same numerical method.
"""

import numpy as np


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
    return 0.5 * omega**2 * x**2


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
    return np.where(np.abs(x) <= a, 0.0, wall_height)


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
    return np.where(np.abs(x) <= a, 0.0, V0)


def quartic_double_well(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 6.0,
    shift_min_to_zero: bool = True,
) -> np.ndarray:
    """
    Quartic double-well potential V(x) = a x^4 - b x^2.

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
    v = a * x**4 - b * x**2
    if shift_min_to_zero:
        v = v - np.min(v)
    return v


def quartic_oscillator(
    x: np.ndarray,
    lam: float = 0.1,
) -> np.ndarray:
    """
    Anharmonic quartic oscillator V(x) = 1/2 x^2 + lam x^4.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    lam : float, optional
        Strength of the quartic correction.

    Returns
    -------
    ndarray
        Potential sampled on the grid.
    """
    return 0.5 * x**2 + lam * x**4
