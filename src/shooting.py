from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numerov import derivative_at_right_edge, normalize_wavefunction, numerov_outward, q_from_energy


@dataclass
class StateSolution:
    energy: float
    parity: str
    x_full: np.ndarray
    psi_full: np.ndarray
    mismatch: float


def initial_conditions(x_half: np.ndarray, parity: str) -> tuple[float, float]:
    h = x_half[1] - x_half[0]
    if parity == "even":
        # psi(0)=1, psi'(0)=0 -> psi(h)=psi(0)+O(h^2)
        return 1.0, 1.0
    if parity == "odd":
        # psi(0)=0, psi'(0)=1 -> psi(h)=h+O(h^3)
        return 0.0, h
    raise ValueError("parity must be 'even' or 'odd'")


def half_domain_wavefunction(x_half: np.ndarray, V_half: np.ndarray, energy: float, parity: str) -> np.ndarray:
    q = q_from_energy(V_half, energy)
    psi0, psi1 = initial_conditions(x_half, parity)
    return numerov_outward(x_half, q, psi0=psi0, psi1=psi1)


def boundary_mismatch(
    x_half: np.ndarray,
    V_half: np.ndarray,
    energy: float,
    parity: str,
    mode: str = "value",
) -> float:
    psi = half_domain_wavefunction(x_half, V_half, energy, parity)
    if mode == "value":
        return float(psi[-1])
    if mode == "logder":
        denom = psi[-1]
        if abs(denom) < 1e-14:
            return np.sign(denom) * np.inf if denom != 0.0 else np.inf
        return float(derivative_at_right_edge(x_half, psi) / denom)
    raise ValueError("mode must be 'value' or 'logder'")


def find_brackets(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 2000,
) -> list[tuple[float, float]]:
    energies = np.linspace(e_min, e_max, n_scan)
    vals = np.array([boundary_mismatch(x_half, V_half, e, parity, mode="value") for e in energies])

    brackets: list[tuple[float, float]] = []
    for i in range(len(energies) - 1):
        a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if a == 0.0:
            eps = 1e-10 * max(1.0, abs(energies[i]))
            brackets.append((energies[i] - eps, energies[i] + eps))
        elif a * b < 0.0:
            brackets.append((energies[i], energies[i + 1]))
    return brackets


def bisect_energy(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, float]:
    lo, hi = bracket
    flo = boundary_mismatch(x_half, V_half, lo, parity, mode="value")
    fhi = boundary_mismatch(x_half, V_half, hi, parity, mode="value")

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if flo == 0.0:
        return lo, flo
    if fhi == 0.0:
        return hi, fhi
    if flo * fhi > 0.0:
        raise ValueError("Bisection requires a sign-changing bracket.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = boundary_mismatch(x_half, V_half, mid, parity, mode="value")

        if not np.isfinite(fmid):
            raise ValueError("Non-finite mismatch during bisection.")

        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid, fmid

        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    mid = 0.5 * (lo + hi)
    return mid, boundary_mismatch(x_half, V_half, mid, parity, mode="value")


def build_full_wavefunction(x_half: np.ndarray, psi_half: np.ndarray, parity: str) -> tuple[np.ndarray, np.ndarray]:
    if parity == "even":
        x_left = -x_half[:0:-1]
        psi_left = psi_half[:0:-1]
    elif parity == "odd":
        x_left = -x_half[:0:-1]
        psi_left = -psi_half[:0:-1]
    else:
        raise ValueError("parity must be 'even' or 'odd'")

    x_full = np.concatenate([x_left, x_half])
    psi_full = np.concatenate([psi_left, psi_half])
    return x_full, psi_full


def solve_state_from_bracket(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
) -> StateSolution:
    energy, mismatch = bisect_energy(x_half, V_half, parity, bracket, tol=tol)
    psi_half = half_domain_wavefunction(x_half, V_half, energy, parity)
    x_full, psi_full = build_full_wavefunction(x_half, psi_half, parity)
    psi_full = normalize_wavefunction(x_full, psi_full)
    return StateSolution(
        energy=energy,
        parity=parity,
        x_full=x_full,
        psi_full=psi_full,
        mismatch=mismatch,
    )


def solve_symmetric_potential(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict | None = None,
    n_even: int = 3,
    n_odd: int = 3,
    e_min: float | None = None,
    e_max: float | None = None,
    scan_points: int = 1200,
    tol: float = 1e-12,
) -> list[StateSolution]:
    if potential_kwargs is None:
        potential_kwargs = {}

    x_half = np.linspace(0.0, x_max, n_grid)
    V_half = potential_fn(x_half, **potential_kwargs)

    if e_min is None:
        e_min = float(np.min(V_half))
    if e_max is None:
        e_max = float(np.max(V_half))
        # For confined smooth potentials on a box, scan somewhat above the well minimum too.
        e_max = max(e_max, e_min + 30.0)

    solutions: list[StateSolution] = []

    for parity, n_needed in [("even", n_even), ("odd", n_odd)]:
        brackets = find_brackets(x_half, V_half, parity, e_min=e_min, e_max=e_max, n_scan=scan_points)
        if len(brackets) < n_needed:
            raise RuntimeError(
                f"Found only {len(brackets)} {parity} brackets, needed {n_needed}. "
                "Increase x_max, e_max, or scan_points."
            )
        for bracket in brackets[:n_needed]:
            solutions.append(solve_state_from_bracket(x_half, V_half, parity, bracket, tol=tol))

    solutions.sort(key=lambda s: s.energy)
    return solutions
