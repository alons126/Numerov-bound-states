from __future__ import annotations

import numpy as np


def harmonic_oscillator(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    return 0.5 * omega**2 * x**2


def finite_square_well(x: np.ndarray, V0: float = 20.0, a: float = 1.0) -> np.ndarray:
    return np.where(np.abs(x) <= a, 0.0, V0)


def infinite_square_well_numeric(
    x: np.ndarray,
    a: float = 1.0,
    wall_height: float = 1.0e6,
) -> np.ndarray:
    return np.where(np.abs(x) <= a, 0.0, wall_height)


def double_well_quartic(x: np.ndarray, a: float = 1.0, b: float = 6.0) -> np.ndarray:
    return a * x**4 - b * x**2


def shifted_double_well_quartic(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 6.0,
) -> np.ndarray:
    v = double_well_quartic(x, a=a, b=b)
    return v - np.min(v)


def soft_box(x: np.ndarray, a: float = 1.0, k: float = 100.0) -> np.ndarray:
    z = np.maximum(np.abs(x) - a, 0.0)
    return k * z**2
