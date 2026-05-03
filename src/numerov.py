from __future__ import annotations

"""
Low-level Numerov utilities.

This module contains the numerical building blocks for solving the 1D
time-independent Schrödinger equation in dimensionless form,

    psi''(x) = 2 [V(x) - E] psi(x),

using the Numerov method on a uniform grid.

This file is the numerical core of the project. It does not decide which
eigenvalue is correct; instead it provides the low-level operations that all
higher layers rely on when turning the Schrödinger boundary-value eigenproblem
into a numerical shooting calculation:
- Convert a trial energy into the coefficient q(x)
- Propagate a trial solution with the Numerov recurrence
- Normalize the resulting wavefunction safely
- Estimate boundary derivatives needed by parity-based shooting

Several comments in the report are implemented directly here:
- `numerov_march()` rescales large trial solutions during scans so a wrong
  trial energy does not overflow before the mismatch is evaluated
- `normalize_wavefunction()` uses numerical quadrature because physical bound
  states must satisfy the normalization condition integral |psi|^2 dx = 1
- `derivative_at_right_edge()` uses a fourth-order stencil when possible

That last point is not cosmetic. Even-state inward shooting enforces
`psi'(0)=0`, so a low-order derivative estimate would make the overall method
look worse than the Numerov recurrence itself really is.
"""

import numpy as np


# ---------------------------------------------------------------------------
# FUNCTION: q_from_energy
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
# FUNCTION: numerov_march
# ---------------------------------------------------------------------------
def numerov_march(
    x: np.ndarray,
    q: np.ndarray,
    psi0: float,
    psi1: float,
) -> np.ndarray:
    """
    Integrate a second-order ODE by marching along the supplied grid order.

    The method assumes a uniform grid and an equation of the form

        y''(x) = q(x) y(x).

    The word "march" is intentional: this helper advances from ``x[0]`` and
    ``x[1]`` to later grid points in index order, regardless of whether the
    coordinates themselves are increasing or decreasing. On an ascending grid
    it is used for physical outward shooting from ``0`` toward ``x_max``; on
    a descending grid it is used for physical inward shooting from ``x_max``
    toward ``0``. The routine itself therefore has no built-in notion of
    "inward" or "outward" physics; the caller chooses that physical direction
    by how it orders the grid points.

    Parameters
    ----------
    x : ndarray
        Uniform spatial grid, either ascending or descending.
    q : ndarray
        Coefficient array in y'' = q y.
    psi0 : float
        Initial value at the first grid point ``x[0]``.
    psi1 : float
        Initial value at the second grid point ``x[1]``.

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
    # are valid only on a uniform grid, but h may be positive (ascending x)
    # or negative (descending x) because the method follows index order.
    h = x[1] - x[0]

    if not np.allclose(np.diff(x), h, rtol=1e-12, atol=1e-14):
        raise ValueError("Numerov integration requires a uniform grid.")

    # Allocate the output and seed the first two values. Numerov is a
    # two-step method, so these two values replace the usual first-order
    # initial condition pair and define where the index-order march starts.
    psi = np.zeros_like(x, dtype=float)
    psi[0] = psi0
    psi[1] = psi1

    # Precompute constants used by the recurrence at every grid point.
    h2 = h * h
    c = h2 / 12.0

    for n in range(1, len(x) - 1):
        # Rearranged Numerov recurrence:
        # (1 - h^2 * q_{n+1}/12) * psi_{n+1} =
        #   = 2 * (1 + 5 * h^2 * q_n/12) * psi_n -
        #     - (1 - h^2 * q_{n-1}/12) * psi_{n-1}
        # so each new point uses the previous two solution values. If x is
        # ascending this marches physically outward; if x is descending it
        # marches physically inward toward smaller coordinates.
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
# ---------------------------------------------------------------------------
def normalize_wavefunction(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Normalize a wavefunction safely on a discrete grid.

    Physical wavefunctions must satisfy the normalization condition
    integral |psi|^2 dx = 1. Evaluate that discrete normalization safely by
    scaling before squaring, which also avoids overflow when a non-eigenvalue
    shooting trial has grown strongly.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    psi : ndarray
        Unnormalized wavefunction values.

    Returns
    -------
    ndarray
        Wavefunction normalized so that the discrete approximation to
        integral |psi|^2 dx equals 1.
    """

    scale = np.max(np.abs(psi))

    if scale == 0.0:
        raise ValueError("Cannot normalize a zero wavefunction.")

    psi_scaled = psi / scale

    # Approximate the normalization integral ∫|psi|^2 dx from the sampled grid
    # values with the trapezoid rule. This is a simple, stable quadrature on
    # the same mesh used by the solver, and it is accurate enough here because
    # normalization is only a post-processing rescaling step.
    norm = scale * np.sqrt(np.trapezoid(np.abs(psi_scaled) ** 2, x))

    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Wavefunction normalization failed.")

    return psi / norm


# ---------------------------------------------------------------------------
# FUNCTION: derivative_at_right_edge
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
