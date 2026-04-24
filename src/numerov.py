from __future__ import annotations

"""
Low-level Numerov utilities.

This module contains the numerical building blocks for solving the 1D
time-independent Schrödinger equation in dimensionless form,

    psi''(x) = 2 [V(x) - E] psi(x),

using the Numerov method on a uniform grid.
"""

import numpy as np


def q_from_energy(V: np.ndarray, energy: float) -> np.ndarray:
    """
    Build the Numerov coefficient q(x) for a trial energy.

    Parameters
    ----------
    V : ndarray
        Potential evaluated on the spatial grid.
    energy : float
        Trial eigenvalue E.

    Returns
    -------
    ndarray
        Array q(x) such that psi'' = q psi.
    """
    return 2.0 * (V - energy)


def numerov_outward(
    x: np.ndarray,
    q: np.ndarray,
    psi0: float,
    psi1: float,
) -> np.ndarray:
    """
    Integrate a second-order ODE outward with the Numerov recurrence.

    The method assumes a uniform grid and an equation of the form

        y''(x) = q(x) y(x).

    Parameters
    ----------
    x : ndarray
        Uniform spatial grid.
    q : ndarray
        Coefficient array in y'' = q y.
    psi0 : float
        Initial value at the first grid point.
    psi1 : float
        Initial value at the second grid point.

    Returns
    -------
    ndarray
        Unnormalized solution on the supplied grid.

    Notes
    -----
    A simple dynamic rescaling step is included to reduce overflow when trial
    solutions grow exponentially for non-eigenvalue energies.
    """
    if x.ndim != 1 or q.ndim != 1 or len(x) != len(q):
        raise ValueError("x and q must be 1D arrays with the same length.")
    if len(x) < 3:
        raise ValueError("Need at least 3 grid points.")

    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, rtol=1e-12, atol=1e-14):
        raise ValueError("Numerov integration requires a uniform grid.")

    psi = np.zeros_like(x, dtype=float)
    psi[0] = psi0
    psi[1] = psi1

    h2 = h * h
    c = h2 / 12.0

    for n in range(1, len(x) - 1):
        # Standard Numerov update for y'' = q y.
        a = 1.0 - c * q[n + 1]
        b = 2.0 * (1.0 + 5.0 * c * q[n]) * psi[n]
        d = (1.0 - c * q[n - 1]) * psi[n - 1]
        psi[n + 1] = (b - d) / a

        # Keep trial solutions numerically manageable during bracket scans.
        if abs(psi[n + 1]) > 1e100:
            psi[: n + 2] /= 1e100

    return psi


def normalize_wavefunction(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Normalize a wavefunction safely on a discrete grid.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    psi : ndarray
        Unnormalized wavefunction values.

    Returns
    -------
    ndarray
        Wavefunction normalized so that integral |psi|^2 dx = 1.
    """
    scale = np.max(np.abs(psi))
    if scale == 0.0:
        raise ValueError("Cannot normalize a zero wavefunction.")

    psi_scaled = psi / scale
    norm = scale * np.sqrt(np.trapezoid(np.abs(psi_scaled) ** 2, x))

    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Wavefunction normalization failed.")

    return psi / norm


def derivative_at_right_edge(x: np.ndarray, psi: np.ndarray) -> float:
    """
    Estimate the derivative at the last grid point with a backward stencil.

    Parameters
    ----------
    x : ndarray
        Uniform spatial grid.
    psi : ndarray
        Function values on the grid.

    Returns
    -------
    float
        Derivative estimate at the right edge. Uses a fourth-order backward
        stencil when at least five grid points are available, otherwise falls
        back to the standard second-order formula.
    """
    if len(x) < 3:
        raise ValueError("Need at least 3 points for derivative.")

    h = x[1] - x[0]
    if len(x) >= 5:
        return (
            25.0 * psi[-1]
            - 48.0 * psi[-2]
            + 36.0 * psi[-3]
            - 16.0 * psi[-4]
            + 3.0 * psi[-5]
        ) / (12.0 * h)

    return (3.0 * psi[-1] - 4.0 * psi[-2] + psi[-3]) / (2.0 * h)
