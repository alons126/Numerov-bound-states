from __future__ import annotations

import numpy as np


def q_from_energy(V: np.ndarray, energy: float) -> np.ndarray:
    """Return q(x) in psi'' = q(x) psi."""
    return 2.0 * (V - energy)


def numerov_outward(x: np.ndarray, q: np.ndarray, psi0: float, psi1: float) -> np.ndarray:
    """
    Numerov integration for y'' = q(x) y on an equally spaced grid.
    """
    if x.ndim != 1 or q.ndim != 1 or len(x) != len(q):
        raise ValueError("x and q must be 1D arrays of equal length.")
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
        a = 1.0 - c * q[n + 1]
        b = 2.0 * (1.0 + 5.0 * c * q[n]) * psi[n]
        d = (1.0 - c * q[n - 1]) * psi[n - 1]
        psi[n + 1] = (b - d) / a

    return psi


def normalize_wavefunction(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero wavefunction.")
    return psi / norm


def derivative_at_right_edge(x: np.ndarray, psi: np.ndarray) -> float:
    """
    Second-order backward derivative at the last point.
    """
    if len(x) < 3:
        raise ValueError("Need at least 3 points for derivative.")
    h = x[1] - x[0]
    return (3.0 * psi[-1] - 4.0 * psi[-2] + psi[-3]) / (2.0 * h)
