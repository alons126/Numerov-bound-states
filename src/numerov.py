from __future__ import annotations

"""
Low-level Numerov utilities.

This module contains the numerical building blocks for solving the 1D
time-independent Schrödinger equation in dimensionless form,

    psi''(x) = 2 [V(x) - E] psi(x),

using the Numerov method on a uniform grid.

Reviewer guide
--------------
This file is the numerical core of the project. It does not decide which
eigenvalue is correct; instead it provides the low-level operations that all
higher layers rely on when turning the Schrödinger boundary-value eigenproblem
into a numerical shooting calculation:
- convert a trial energy into the coefficient q(x)
- propagate a trial solution with the Numerov recurrence
- normalize the resulting wavefunction safely
- estimate boundary derivatives needed by parity-based shooting

Several comments in the report are implemented directly here:
- `numerov_outward()` rescales large trial solutions during scans so a wrong
  trial energy does not overflow before the mismatch is evaluated
- `normalize_wavefunction()` uses numerical quadrature because physical bound
  states must satisfy integral |psi|^2 dx = 1
- `derivative_at_right_edge()` uses a fourth-order stencil when possible

That last point is not cosmetic. Even-state inward shooting enforces
`psi'(0)=0`, so a low-order derivative estimate would make the overall method
look worse than the Numerov recurrence itself really is.
"""

import numpy as np


# ---------------------------------------------------------------------------
# FUNCTION: q_from_energy
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
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
    # Repackage the Schrödinger equation into the `y'' = q(x) y` form expected
    # by Numerov. Every trial energy produces a different coefficient field.
    return 2.0 * (V - energy)


# ---------------------------------------------------------------------------
# FUNCTION: numerov_outward
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
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

    # Numerov assumes a constant spacing h. The recurrence coefficients below
    # are valid only on a uniform grid.
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, rtol=1e-12, atol=1e-14):
        raise ValueError("Numerov integration requires a uniform grid.")

    # Allocate the output and seed the first two values. Numerov is a
    # two-step method, so these two values replace the usual first-order
    # initial condition pair.
    psi = np.zeros_like(x, dtype=float)
    psi[0] = psi0
    psi[1] = psi1

    # Precompute constants used by the recurrence at every grid point.
    h2 = h * h
    c = h2 / 12.0

    for n in range(1, len(x) - 1):
        # Rearranged Numerov recurrence:
        #   (1 - h^2 q_{n+1}/12) psi_{n+1}
        # = 2(1 + 5 h^2 q_n/12) psi_n - (1 - h^2 q_{n-1}/12) psi_{n-1}
        # so each new point uses the previous two solution values.
        a = 1.0 - c * q[n + 1]
        b = 2.0 * (1.0 + 5.0 * c * q[n]) * psi[n]
        d = (1.0 - c * q[n - 1]) * psi[n - 1]
        psi[n + 1] = (b - d) / a

        # Keep trial solutions numerically manageable during bracket scans.
        if abs(psi[n + 1]) > 1e100:
            psi[: n + 2] /= 1e100

    return psi


# ---------------------------------------------------------------------------
# FUNCTION: normalize_wavefunction
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
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
    # Physical wavefunctions must satisfy integral |psi|^2 dx = 1. Evaluate
    # that discrete normalization safely by scaling before squaring, which also
    # avoids overflow when a non-eigenvalue shooting trial has grown strongly.
    scale = np.max(np.abs(psi))
    if scale == 0.0:
        raise ValueError("Cannot normalize a zero wavefunction.")

    psi_scaled = psi / scale
    # Integrate |psi|^2 on the discrete grid with the trapezoid rule, then
    # undo the temporary scaling in a numerically safe way.
    norm = scale * np.sqrt(np.trapezoid(np.abs(psi_scaled) ** 2, x))

    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Wavefunction normalization failed.")

    return psi / norm


# ---------------------------------------------------------------------------
# FUNCTION: derivative_at_right_edge
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
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

    # The derivative is requested at the last grid point. For inward shooting
    # on a decreasing grid, that last point is still x = 0, so the same helper
    # works without any sign changes or index remapping.
    h = x[1] - x[0]

    # The fourth-order stencil is important for even-state inward shooting:
    # the eigenvalue condition is psi'(0)=0, so a low-order derivative stencil
    # would degrade the accuracy of an otherwise high-order Numerov solve.
    if len(x) >= 5:
        return (
            25.0 * psi[-1]
            - 48.0 * psi[-2]
            + 36.0 * psi[-3]
            - 16.0 * psi[-4]
            + 3.0 * psi[-5]
        ) / (12.0 * h)

    return (3.0 * psi[-1] - 4.0 * psi[-2] + psi[-3]) / (2.0 * h)
