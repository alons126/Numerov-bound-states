from __future__ import annotations

"""
Parity-based shooting solver for symmetric 1D potentials.

For a symmetric potential, low-lying eigenstates can be separated into:
- even states, with psi'(0) = 0,
- odd states, with psi(0) = 0.

This module exploits that symmetry to solve the bound-state problem on the
half-domain x in [0, x_max], then reconstructs the full wavefunction.
"""

from dataclasses import dataclass

import numpy as np

from src.numerov import (
    derivative_at_right_edge,
    normalize_wavefunction,
    numerov_outward,
    q_from_energy,
)


@dataclass
class StateSolution:
    """
    Container for a single bound-state solution.

    Attributes
    ----------
    energy : float
        Computed eigenvalue.
    parity : str
        Either "even" or "odd".
    x_full : ndarray
        Full spatial grid on [-x_max, x_max].
    psi_full : ndarray
        Normalized full-domain wavefunction.
    mismatch : float
        Residual value of the boundary mismatch at the final eigenvalue.
    """

    energy: float
    parity: str
    x_full: np.ndarray
    psi_full: np.ndarray
    mismatch: float


def initial_conditions(
    x_half: np.ndarray, q_half: np.ndarray, parity: str
) -> tuple[float, float]:
    """
    Construct parity-consistent starting values near x = 0.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid beginning at x = 0.
    parity : str
        State parity, either "even" or "odd".

    Returns
    -------
    tuple[float, float]
        Starting values (psi0, psi1) for Numerov integration.
    """
    h = x_half[1] - x_half[0]
    q0 = q_half[0]

    if parity == "even":
        # psi(0) = 1 and psi'(0) = 0.
        # The second grid value includes the Taylor correction psi(h) = 1 + q(0)h^2/2 + O(h^4).
        return 1.0, 1.0 + 0.5 * q0 * h**2
    if parity == "odd":
        # psi(0) = 0 and psi'(0) = 1.
        # The second grid value includes psi(h) = h + q(0)h^3/6 + O(h^5).
        return 0.0, h + (q0 * h**3) / 6.0

    raise ValueError("parity must be 'even' or 'odd'")


def half_domain_wavefunction(
    x_half: np.ndarray,
    V_half: np.ndarray,
    energy: float,
    parity: str,
) -> np.ndarray:
    """
    Compute the half-domain trial wavefunction for a given energy.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid.
    V_half : ndarray
        Potential sampled on the half-domain.
    energy : float
        Trial energy.
    parity : str
        State parity.

    Returns
    -------
    ndarray
        Unnormalized half-domain wavefunction.
    """
    q = q_from_energy(V_half, energy)
    psi0, psi1 = initial_conditions(x_half, q, parity)
    return numerov_outward(x_half, q, psi0=psi0, psi1=psi1)


def boundary_mismatch(
    x_half: np.ndarray,
    V_half: np.ndarray,
    energy: float,
    parity: str,
    mode: str = "value",
) -> float:
    """
    Evaluate the mismatch used for eigenvalue shooting.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid.
    V_half : ndarray
        Half-domain potential.
    energy : float
        Trial energy.
    parity : str
        State parity.
    mode : str, optional
        Type of mismatch to compute. "value" uses psi(x_max), while
        "logder" uses psi'(x_max)/psi(x_max).

    Returns
    -------
    float
        Scalar mismatch value used in bracketing or diagnostics.
    """
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
    """
    Scan an energy interval and locate sign-changing brackets.

    Parameters
    ----------
    x_half, V_half : ndarray
        Half-domain grid and potential.
    parity : str
        Target parity sector.
    e_min, e_max : float
        Search interval in energy.
    n_scan : int, optional
        Number of scan points used to build trial energies.

    Returns
    -------
    list[tuple[float, float]]
        Brackets suitable for bisection.
    """
    energies = np.linspace(e_min, e_max, n_scan)
    vals = np.array(
        [boundary_mismatch(x_half, V_half, e, parity, mode="value") for e in energies]
    )

    brackets: list[tuple[float, float]] = []

    for i in range(len(energies) - 1):
        a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if a == 0.0:
            eps = 1e-10 * max(1.0, abs(energies[i]))
            brackets.append((energies[i] - eps, energies[i] + eps))
        elif np.signbit(a) != np.signbit(b):
            brackets.append((energies[i], energies[i + 1]))

    return brackets


def sample_boundary_mismatch(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the shooting mismatch over an energy interval.

    This is used only for diagnostics and visualization. The eigenvalues are
    located where the mismatch curve crosses zero.
    """
    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [boundary_mismatch(x_half, V_half, e, parity, mode="value") for e in energies],
        dtype=float,
    )
    return energies, mismatches


def bisection_history(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 80,
) -> list[dict]:
    """
    Record the bisection process for one eigenvalue bracket.

    Returns a list of dictionaries containing the lower edge, upper edge,
    midpoint, and mismatch at each bisection iteration.
    """
    lo, hi = bracket
    flo = boundary_mismatch(x_half, V_half, lo, parity, mode="value")
    fhi = boundary_mismatch(x_half, V_half, hi, parity, mode="value")

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if np.signbit(flo) == np.signbit(fhi):
        raise ValueError("Bisection history requires a sign-changing bracket.")

    history: list[dict] = []
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = boundary_mismatch(x_half, V_half, mid, parity, mode="value")
        history.append(
            {
                "iteration": iteration,
                "lo": lo,
                "hi": hi,
                "mid": mid,
                "mismatch_mid": fmid,
                "width": hi - lo,
            }
        )

        if abs(fmid) < tol or abs(hi - lo) < tol:
            break
        if np.signbit(flo) != np.signbit(fmid):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return history


def bisect_energy(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, float]:
    """
    Refine a sign-changing eigenvalue bracket with bisection.

    Parameters
    ----------
    x_half, V_half : ndarray
        Half-domain grid and potential.
    parity : str
        Target parity sector.
    bracket : tuple[float, float]
        Initial sign-changing energy interval.
    tol : float, optional
        Bisection termination tolerance.
    max_iter : int, optional
        Maximum number of bisection iterations.

    Returns
    -------
    tuple[float, float]
        Refined eigenvalue estimate and final mismatch.
    """
    lo, hi = bracket
    flo = boundary_mismatch(x_half, V_half, lo, parity, mode="value")
    fhi = boundary_mismatch(x_half, V_half, hi, parity, mode="value")

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if flo == 0.0:
        return lo, flo
    if fhi == 0.0:
        return hi, fhi
    if np.signbit(flo) == np.signbit(fhi):
        raise ValueError("Bisection requires a sign-changing bracket.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = boundary_mismatch(x_half, V_half, mid, parity, mode="value")

        if not np.isfinite(fmid):
            raise ValueError("Non-finite mismatch during bisection.")
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid, fmid

        if np.signbit(flo) != np.signbit(fmid):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    mid = 0.5 * (lo + hi)
    return mid, boundary_mismatch(x_half, V_half, mid, parity, mode="value")


def build_full_wavefunction(
    x_half: np.ndarray,
    psi_half: np.ndarray,
    parity: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the full wavefunction from its half-domain representation.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid.
    psi_half : ndarray
        Half-domain wavefunction.
    parity : str
        State parity.

    Returns
    -------
    tuple[ndarray, ndarray]
        Full-domain grid and full-domain wavefunction.
    """
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
    """
    Compute one bound state starting from a valid energy bracket.

    Parameters
    ----------
    x_half, V_half : ndarray
        Half-domain grid and potential.
    parity : str
        Target parity sector.
    bracket : tuple[float, float]
        Energy interval containing a single eigenvalue.
    tol : float, optional
        Root-finding tolerance.

    Returns
    -------
    StateSolution
        Structured solution object containing energy and wavefunction data.
    """
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


def inward_decay_initial_conditions(
    x_desc: np.ndarray,
    V_desc: np.ndarray,
    energy: float,
) -> tuple[float, float]:
    """
    Construct stable starting values at x_max for inward shooting.

    For confining potentials such as the harmonic oscillator, the physical
    bound-state solution decays in the forbidden region. Starting the integration
    at large x and integrating inward suppresses the unphysical growing tail.

    The asymptotic estimate uses psi'/psi ~= -sqrt(2(V-E)) at x_max.
    """
    dx = abs(x_desc[1] - x_desc[0])

    # In the forbidden region, a decaying bound-state tail satisfies roughly
    # psi'/psi = -kappa(x), where kappa(x) = sqrt(2[V(x)-E]).
    # Since we integrate inward from x_max to x_max-dx, the physical decaying
    # solution grows by exp(integral kappa dx) over the first step.
    # This is more accurate than a low-order Taylor start and makes the
    # Numerov/RK4 comparison fairer.
    kappa0 = np.sqrt(max(2.0 * (V_desc[0] - energy), 1.0e-14))
    kappa1 = np.sqrt(max(2.0 * (V_desc[1] - energy), 1.0e-14))
    exponent = 0.5 * (kappa0 + kappa1) * dx

    psi0 = 1.0
    psi1 = psi0 * np.exp(exponent)
    return psi0, psi1


def inward_decay_half_domain_wavefunction(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    energy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the decaying tail inward from x_max to x = 0.

    Returns
    -------
    tuple[ndarray, ndarray]
        A decreasing grid from x_max to 0 and the corresponding unnormalized
        inward-integrated wavefunction.
    """
    x_desc = np.linspace(x_max, 0.0, n_grid)
    V_desc = potential_fn(x_desc, **potential_kwargs)
    q_desc = q_from_energy(V_desc, energy)
    psi0, psi1 = inward_decay_initial_conditions(x_desc, V_desc, energy)
    psi_desc = numerov_outward(x_desc, q_desc, psi0=psi0, psi1=psi1)
    return x_desc, psi_desc


def inward_decay_boundary_mismatch(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    energy: float,
    parity: str,
) -> float:
    """
    Evaluate the parity mismatch at the origin for inward shooting.

    Even states should satisfy psi'(0) = 0.
    Odd states should satisfy psi(0) = 0.
    """
    x_desc, psi_desc = inward_decay_half_domain_wavefunction(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=potential_fn,
        potential_kwargs=potential_kwargs,
        energy=energy,
    )

    if parity == "even":
        return float(derivative_at_right_edge(x_desc, psi_desc))
    if parity == "odd":
        return float(psi_desc[-1])

    raise ValueError("parity must be 'even' or 'odd'")


def sample_inward_decay_mismatch(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the inward-shooting parity mismatch over an energy interval.
    """
    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [
            inward_decay_boundary_mismatch(
                x_max,
                n_grid,
                potential_fn,
                potential_kwargs,
                energy,
                parity,
            )
            for energy in energies
        ],
        dtype=float,
    )
    return energies, mismatches


def find_inward_decay_brackets(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 2000,
) -> list[tuple[float, float]]:
    """
    Locate sign-changing brackets for inward-shooting eigenvalue searches.
    """
    energies, vals = sample_inward_decay_mismatch(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        parity,
        e_min,
        e_max,
        n_scan=n_scan,
    )

    brackets: list[tuple[float, float]] = []
    for i in range(len(energies) - 1):
        a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if a == 0.0:
            eps = 1e-10 * max(1.0, abs(energies[i]))
            brackets.append((energies[i] - eps, energies[i] + eps))
        elif np.signbit(a) != np.signbit(b):
            brackets.append((energies[i], energies[i + 1]))

    return brackets


def bisect_energy_inward_decay(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, float]:
    """
    Refine an inward-shooting eigenvalue bracket with bisection.
    """
    lo, hi = bracket
    flo = inward_decay_boundary_mismatch(
        x_max, n_grid, potential_fn, potential_kwargs, lo, parity
    )
    fhi = inward_decay_boundary_mismatch(
        x_max, n_grid, potential_fn, potential_kwargs, hi, parity
    )

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if flo == 0.0:
        return lo, flo
    if fhi == 0.0:
        return hi, fhi
    if np.signbit(flo) == np.signbit(fhi):
        raise ValueError("Bisection requires a sign-changing bracket.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = inward_decay_boundary_mismatch(
            x_max,
            n_grid,
            potential_fn,
            potential_kwargs,
            mid,
            parity,
        )

        if not np.isfinite(fmid):
            raise ValueError("Non-finite mismatch during bisection.")
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid, fmid

        if np.signbit(flo) != np.signbit(fmid):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    mid = 0.5 * (lo + hi)
    return mid, inward_decay_boundary_mismatch(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        mid,
        parity,
    )


def bisection_history_inward_decay(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 80,
) -> list[dict]:
    """
    Record the inward-shooting bisection process for diagnostic plots.
    """
    lo, hi = bracket
    flo = inward_decay_boundary_mismatch(
        x_max, n_grid, potential_fn, potential_kwargs, lo, parity
    )

    history: list[dict] = []
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = inward_decay_boundary_mismatch(
            x_max,
            n_grid,
            potential_fn,
            potential_kwargs,
            mid,
            parity,
        )
        history.append(
            {
                "iteration": iteration,
                "lo": lo,
                "hi": hi,
                "mid": mid,
                "mismatch_mid": fmid,
            }
        )

        if abs(fmid) < tol or abs(hi - lo) < tol:
            break

        if np.signbit(flo) != np.signbit(fmid):
            hi = mid
        else:
            lo = mid
            flo = fmid

    return history


def solve_state_from_inward_decay_bracket(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
) -> StateSolution:
    """
    Compute one bound state using inward shooting from the decaying tail.
    """
    energy, mismatch = bisect_energy_inward_decay(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        parity,
        bracket,
        tol=tol,
    )

    x_desc, psi_desc = inward_decay_half_domain_wavefunction(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        energy,
    )

    x_half = x_desc[::-1]
    psi_half = psi_desc[::-1]
    x_full, psi_full = build_full_wavefunction(x_half, psi_half, parity)
    psi_full = normalize_wavefunction(x_full, psi_full)

    return StateSolution(
        energy=energy,
        parity=parity,
        x_full=x_full,
        psi_full=psi_full,
        mismatch=mismatch,
    )


def solve_symmetric_potential_inward_decay(
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
    """
    Solve symmetric confining potentials by shooting inward from x_max.

    This is especially useful for the harmonic oscillator. Outward shooting from
    the origin can pick up a numerically growing exponential tail in the forbidden
    region. Inward shooting starts from the decaying asymptotic side and imposes
    parity at the origin, which produces much cleaner wavefunctions.
    """
    if potential_kwargs is None:
        potential_kwargs = {}

    x_probe = np.linspace(0.0, x_max, n_grid)
    V_probe = potential_fn(x_probe, **potential_kwargs)

    if e_min is None:
        e_min = float(np.min(V_probe))
    if e_max is None:
        e_max = float(np.max(V_probe))
        e_max = max(e_max, e_min + 30.0)

    solutions: list[StateSolution] = []

    for parity, n_needed in [("even", n_even), ("odd", n_odd)]:
        brackets = find_inward_decay_brackets(
            x_max,
            n_grid,
            potential_fn,
            potential_kwargs,
            parity,
            e_min=e_min,
            e_max=e_max,
            n_scan=scan_points,
        )

        if len(brackets) < n_needed:
            raise RuntimeError(
                f"Found only {len(brackets)} {parity} inward-decay brackets, "
                f"needed {n_needed}. Increase x_max, e_max, or scan_points."
            )

        for bracket in brackets[:n_needed]:
            solutions.append(
                solve_state_from_inward_decay_bracket(
                    x_max,
                    n_grid,
                    potential_fn,
                    potential_kwargs,
                    parity,
                    bracket,
                    tol=tol,
                )
            )

    solutions.sort(key=lambda s: s.energy)
    return solutions


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
    """
    Solve for multiple bound states of a symmetric potential.

    Parameters
    ----------
    x_max : float
        Right boundary of the half-domain [0, x_max].
    n_grid : int
        Number of half-domain grid points.
    potential_fn : callable
        Potential function V(x, **kwargs).
    potential_kwargs : dict, optional
        Keyword arguments passed to the potential.
    n_even, n_odd : int, optional
        Number of even and odd states requested.
    e_min, e_max : float, optional
        Energy search interval. If omitted, values are inferred from the potential.
    scan_points : int, optional
        Number of scan points used to find sign-changing brackets.
    tol : float, optional
        Root-finding tolerance for the bisection step.

    Returns
    -------
    list[StateSolution]
        Bound states sorted by energy.
    """
    if potential_kwargs is None:
        potential_kwargs = {}

    x_half = np.linspace(0.0, x_max, n_grid)
    V_half = potential_fn(x_half, **potential_kwargs)

    if e_min is None:
        e_min = float(np.min(V_half))
    if e_max is None:
        e_max = float(np.max(V_half))
        e_max = max(e_max, e_min + 30.0)

    solutions: list[StateSolution] = []

    for parity, n_needed in [("even", n_even), ("odd", n_odd)]:
        brackets = find_brackets(
            x_half,
            V_half,
            parity,
            e_min=e_min,
            e_max=e_max,
            n_scan=scan_points,
        )

        if len(brackets) < n_needed:
            raise RuntimeError(
                f"Found only {len(brackets)} {parity} brackets, needed {n_needed}. "
                "Increase x_max, e_max, or scan_points."
            )

        for bracket in brackets[:n_needed]:
            solutions.append(
                solve_state_from_bracket(x_half, V_half, parity, bracket, tol=tol)
            )

    solutions.sort(key=lambda s: s.energy)
    return solutions
