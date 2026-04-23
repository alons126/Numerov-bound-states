from __future__ import annotations

import numpy as np


def harmonic_oscillator(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return 0.5 * omega**2 * x**2


def infinite_square_well_numeric(
    x: np.ndarray,
    a: float = 1.0,
    wall_height: float = 1.0e6,
) -> np.ndarray:
    return np.where(np.abs(x) <= a, 0.0, wall_height)


def finite_square_well(
    x: np.ndarray,
    V0: float = 20.0,
    a: float = 1.0,
) -> np.ndarray:
    return np.where(np.abs(x) <= a, 0.0, V0)


def quartic_double_well(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 6.0,
    shift_min_to_zero: bool = True,
) -> np.ndarray:
    v = a * x**4 - b * x**2
    if shift_min_to_zero:
        v = v - np.min(v)
    return v


def quartic_oscillator(
    x: np.ndarray,
    lam: float = 0.1,
) -> np.ndarray:
    return 0.5 * x**2 + lam * x**4
