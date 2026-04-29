from __future__ import annotations

"""
Parity-based shooting solver for symmetric 1D potentials.

For a symmetric potential, low-lying eigenstates can be separated into:
- even states, with psi'(0) = 0,
- odd states, with psi(0) = 0.

This module exploits that symmetry to solve the bound-state problem on the
half-domain x in [0, x_max], then reconstructs the full wavefunction.

Reviewer guide
--------------
This file is the bound-state eigenvalue solver. Conceptually, it turns the
Schrödinger boundary-value eigenproblem into a practical shooting procedure:
1. choose a trial energy E
2. integrate the corresponding trial wavefunction with Numerov
3. evaluate a boundary mismatch M(E)
4. bracket sign changes of M(E)
5. refine the roots to obtain eigenvalues

Two formulations are implemented because the project studies both bounded and
unbounded confining systems:
- outward shooting from x=0 for finite-domain or effectively boxed problems
- inward shooting from x_max for decaying-tail problems such as the harmonic
  oscillator, where outward shooting is numerically contaminated by the growing
  forbidden-region solution

Outward shooting case:
The initial conditions are imposed at the symmetry point x=0 using parity. For
an even state, psi(0)=1 and psi'(0)=0; for an odd state, psi(0)=0 and psi'(0)=1.
The trial wavefunction is then integrated outward to x_max. The mismatch is the
right-edge leakage, M(E)=psi_E(x_max). An allowed energy is found when this
mismatch vanishes. This is the natural setup when the outer boundary is known
or when the finite computational box is part of the approximation being tested.

Inward shooting case:
The initial conditions are imposed near x_max using the expected decaying tail of
a bound state in a forbidden region. The wavefunction is integrated inward toward
x=0, where parity supplies the boundary condition. For an even state the mismatch
is M(E)=psi'_E(0), while for an odd state it is M(E)=psi_E(0). This is more stable
for confining infinite-domain potentials such as the harmonic oscillator, because
it avoids growing-mode contamination during outward integration.

This file also contains the most delicate numerical fixes described in the
reviewer documents:
- `initial_conditions_outward_shooting()` includes higher-order Taylor startup terms, including
  the q''(0) contribution, so the first Numerov step does not spoil the
  observed fourth-order convergence
- `bisect_energy_outward_shooting()` finishes with a few safeguarded secant-style polishing
  steps inside the final sign-changing bracket
- `StateSolution.mismatch` stores the final residual diagnostic returned by the
  corresponding shooting formulation
"""

from dataclasses import dataclass

import numpy as np

from src.numerov import (
    derivative_at_right_edge,
    normalize_wavefunction,
    numerov_outward,
    q_from_energy,
)


# ---------------------------------------------------------------------------
# DATA CLASS: StateSolution
# ---------------------------------------------------------------------------
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
        Final diagnostic residual reported by the solver. For outward shooting
        this is the normalized boundary leakage psi(x_max); for inward
        shooting it is the final parity mismatch at the origin.
    """

    energy: float
    parity: str
    x_full: np.ndarray
    psi_full: np.ndarray
    mismatch: float


# ---------------------------------------------------------------------------
# FUNCTION: initial_conditions_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def initial_conditions_outward_shooting(
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

    # The Taylor expansion around x=0 supplies the first two Numerov values.
    # This is where parity enters the shooting calculation on the half-domain.
    h = x_half[1] - x_half[0]
    q0 = q_half[0]

    # For smooth symmetric potentials, q'(0)=0 but q''(0) generally does not
    # vanish. Including it prevents the startup step from degrading the
    # observed high-order convergence for center-penetrating states. Without
    # this term, the startup error can dominate the measured convergence rate
    # even when the Numerov recurrence itself is higher order.
    q2 = (q_half[2] - 2.0 * q_half[1] + q_half[0]) / (h * h)

    if parity == "even":
        # psi(0) = 1 and psi'(0) = 0.
        # Include the full h^4 term. For y'' = q(x) y, the even Taylor series
        # is y(h) = 1 + q0 h^2 / 2 + (q0^2 + q''(0)) h^4 / 24 + O(h^6).
        return 1.0, 1.0 + 0.5 * q0 * h**2 + ((q0**2 + q2) * h**4) / 24.0
    if parity == "odd":
        # psi(0) = 0 and psi'(0) = 1.
        # Include the full h^5 term. For odd states the series is
        # y(h) = h + q0 h^3 / 6 + (q0^2 + 3 q''(0)) h^5 / 120 + O(h^7).
        return 0.0, h + (q0 * h**3) / 6.0 + ((q0**2 + 3.0 * q2) * h**5) / 120.0

    raise ValueError("parity must be 'even' or 'odd'")


# ---------------------------------------------------------------------------
# FUNCTION: half_domain_wavefunction_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def half_domain_wavefunction_outward_shooting(
    x_half: np.ndarray,
    V_half: np.ndarray,
    energy: float,
    parity: str,
) -> np.ndarray:
    """
    Compute the half-domain trial wavefunction for a given energy.

    A trial energy does not automatically satisfy the boundary conditions.
    Shooting treats the ODE solve as a diagnostic: only special energies
    produce a wavefunction that also matches the far boundary.
    Build the Schrödinger coefficient for this energy, convert the parity
    condition into the first two grid values, then march to x_max.

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
    psi0, psi1 = initial_conditions_outward_shooting(x_half, q, parity)

    return numerov_outward(x_half, q, psi0=psi0, psi1=psi1)


# ---------------------------------------------------------------------------
# FUNCTION: boundary_mismatch_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def boundary_mismatch_outward_shooting(
    x_half: np.ndarray,
    V_half: np.ndarray,
    energy: float,
    parity: str,
    mode: str = "value",
) -> float:
    """
    Evaluate the mismatch used for eigenvalue shooting.

    The mismatch M(E) is the scalar quantity whose zero selects an allowed
    eigenvalue. This converts the boundary-value problem into root finding.

    This function is used for the outward-shooting case. The wavefunction is
    started at x=0 using parity, integrated outward to x_max, and then checked
    at the far boundary. In the standard "value" mode, the mismatch is
    M(E)=psi_E(x_max). Therefore a correct bound-state energy is one for which
    the trial wavefunction reaches the outer boundary with zero leakage.

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

    psi = half_domain_wavefunction_outward_shooting(x_half, V_half, energy, parity)

    if mode == "value":
        # For outward shooting on a boxed domain, a true eigenstate should have
        # negligible leakage at the far boundary.
        return float(psi[-1])
    if mode == "logder":
        denom = psi[-1]
        if abs(denom) < 1e-14:
            return np.sign(denom) * np.inf if denom != 0.0 else np.inf
        # The logarithmic derivative is a more scale-invariant alternative, but
        # most of the project diagnostics use the simpler boundary value itself.
        return float(derivative_at_right_edge(x_half, psi) / denom)

    raise ValueError("mode must be 'value' or 'logder'")


# ---------------------------------------------------------------------------
# FUNCTION: find_brackets_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def find_brackets_outward_shooting(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 2000,
) -> list[tuple[float, float]]:
    """
    Scan an energy interval and locate sign-changing brackets.

    Bracketing converts the eigenvalue problem into a set of scalar root
    searches. Every sign change in the mismatch curve marks one candidate
    eigenvalue interval, which is why the report treats root finding as a
    central ingredient rather than a post-processing detail.

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
        [
            boundary_mismatch_outward_shooting(x_half, V_half, e, parity, mode="value")
            for e in energies
        ]
    )

    brackets: list[tuple[float, float]] = []

    for i in range(len(energies) - 1):
        a, b = vals[i], vals[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if a == 0.0:
            # Keep exact zero hits by inflating them into a tiny bracket so the
            # downstream bisection code can treat all roots uniformly.
            eps = 1e-10 * max(1.0, abs(energies[i]))
            brackets.append((energies[i] - eps, energies[i] + eps))
        elif np.signbit(a) != np.signbit(b):
            # A sign change guarantees at least one root if the mismatch stays
            # continuous across the interval.
            brackets.append((energies[i], energies[i + 1]))

    return brackets


# ---------------------------------------------------------------------------
# FUNCTION: sample_boundary_mismatch
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
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
    located where the mismatch curve crosses zero, so plotting it makes the
    root-finding logic visible instead of leaving it as a black box.
    """

    energies = np.linspace(e_min, e_max, n_scan)
    mismatches = np.array(
        [
            boundary_mismatch_outward_shooting(x_half, V_half, e, parity, mode="value")
            for e in energies
        ],
        dtype=float,
    )

    return energies, mismatches


# ---------------------------------------------------------------------------
# FUNCTION: bisection_history
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
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
    flo = boundary_mismatch_outward_shooting(x_half, V_half, lo, parity, mode="value")
    fhi = boundary_mismatch_outward_shooting(x_half, V_half, hi, parity, mode="value")

    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise ValueError("Non-finite function value at bracket endpoints.")
    if np.signbit(flo) == np.signbit(fhi):
        raise ValueError("Bisection history requires a sign-changing bracket.")

    history: list[dict] = []
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = boundary_mismatch_outward_shooting(
            x_half, V_half, mid, parity, mode="value"
        )
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
            # Keep the half-interval that still brackets the zero.
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return history


# ---------------------------------------------------------------------------
# FUNCTION: bisect_energy_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def bisect_energy_outward_shooting(
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
    flo = boundary_mismatch_outward_shooting(x_half, V_half, lo, parity, mode="value")
    fhi = boundary_mismatch_outward_shooting(x_half, V_half, hi, parity, mode="value")

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
        fmid = boundary_mismatch_outward_shooting(
            x_half, V_half, mid, parity, mode="value"
        )

        if not np.isfinite(fmid):
            raise ValueError("Non-finite mismatch during bisection.")
        if abs(fmid) < tol:
            return mid, fmid
        if abs(hi - lo) < tol:
            if np.signbit(flo) != np.signbit(fmid):
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
            break

        if np.signbit(flo) != np.signbit(fmid):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    # Finish with a few safeguarded secant steps inside the last sign-changing
    # bracket to report a much better boundary residual without sacrificing robustness
    for _ in range(8):
        denom = fhi - flo
        if denom == 0.0 or not np.isfinite(denom):
            break
        # Secant interpolation proposes the root of the line through the two
        # remaining bracket endpoints, but the step is accepted only if it
        # stays inside the sign-changing interval.
        trial = lo - flo * (hi - lo) / denom
        if not (min(lo, hi) <= trial <= max(lo, hi)):
            break

        ftrial = boundary_mismatch_outward_shooting(
            x_half, V_half, trial, parity, mode="value"
        )
        if not np.isfinite(ftrial):
            break
        if abs(ftrial) < tol:
            return trial, ftrial

        if np.signbit(flo) != np.signbit(ftrial):
            hi, fhi = trial, ftrial
        else:
            lo, flo = trial, ftrial

    if abs(flo) <= abs(fhi):
        return lo, flo

    return hi, fhi


# ---------------------------------------------------------------------------
# FUNCTION: build_full_wavefunction
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
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
        # Even states mirror with the same sign across x = 0.
        x_left = -x_half[:0:-1]
        psi_left = psi_half[:0:-1]
    elif parity == "odd":
        # Odd states mirror with a sign flip so psi(0)=0 is preserved.
        x_left = -x_half[:0:-1]
        psi_left = -psi_half[:0:-1]
    else:
        raise ValueError("parity must be 'even' or 'odd'")

    x_full = np.concatenate([x_left, x_half])
    psi_full = np.concatenate([psi_left, psi_half])

    return x_full, psi_full


# ---------------------------------------------------------------------------
# FUNCTION: solve_state_from_bracket_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def solve_state_from_bracket_outward_shooting(
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

    # First determine the eigenvalue, then recompute the corresponding
    # wavefunction once at that energy for the final normalized output.
    energy, _raw_mismatch = bisect_energy_outward_shooting(
        x_half, V_half, parity, bracket, tol=tol
    )
    psi_half = half_domain_wavefunction_outward_shooting(x_half, V_half, energy, parity)
    x_full, psi_full = build_full_wavefunction(x_half, psi_half, parity)
    psi_full = normalize_wavefunction(x_full, psi_full)
    # Report the wall leakage after normalization so the diagnostic is tied to
    # a physical bound-state scale instead of the arbitrary amplitude of the
    # unnormalized shooting solution.
    mismatch = float(psi_full[-1])

    return StateSolution(
        energy=energy,
        parity=parity,
        x_full=x_full,
        psi_full=psi_full,
        mismatch=mismatch,
    )


# ---------------------------------------------------------------------------
# FUNCTION: initial_conditions_inward_shooting
# ---------------------------------------------------------------------------
def initial_conditions_inward_shooting(
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

    # In the forbidden region, a physical bound-state tail satisfies roughly
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


# ---------------------------------------------------------------------------
# FUNCTION: half_domain_wavefunction_inward_shooting
# ---------------------------------------------------------------------------
def half_domain_wavefunction_inward_shooting(
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
    # The grid is descending because Numerov still marches "forward" in index,
    # while physically we want to start at the asymptotic tail and move inward.
    x_desc = np.linspace(x_max, 0.0, n_grid)
    V_desc = potential_fn(x_desc, **potential_kwargs)
    q_desc = q_from_energy(V_desc, energy)
    psi0, psi1 = initial_conditions_inward_shooting(x_desc, V_desc, energy)
    psi_desc = numerov_outward(x_desc, q_desc, psi0=psi0, psi1=psi1)
    return x_desc, psi_desc


# ---------------------------------------------------------------------------
# FUNCTION: boundary_mismatch_inward_shooting
# ---------------------------------------------------------------------------
def boundary_mismatch_inward_shooting(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    energy: float,
    parity: str,
) -> float:
    """
    Evaluate the parity mismatch at the origin for inward shooting.

    This is the inward-shooting counterpart of `boundary_mismatch_outward_shooting()`. Instead of
    starting at x=0 and checking the value at x_max, the solver starts near
    x_max from the expected decaying tail and integrates inward to the origin.
    The origin condition is then used as the scalar root-finding function M(E).

    Even states should satisfy psi'(0) = 0, so M(E)=psi'_E(0).
    Odd states should satisfy psi(0) = 0, so M(E)=psi_E(0).

    For the harmonic oscillator, this origin mismatch is preferable to checking
    the far wall after outward integration because the unphysical growing tail
    contaminates outward shooting on large domains.
    """
    x_desc, psi_desc = half_domain_wavefunction_inward_shooting(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=potential_fn,
        potential_kwargs=potential_kwargs,
        energy=energy,
    )

    if parity == "even":
        # For even states the origin condition is on the derivative.
        return float(derivative_at_right_edge(x_desc, psi_desc))
    if parity == "odd":
        # For odd states the origin condition is on the value itself.
        return float(psi_desc[-1])

    raise ValueError("parity must be 'even' or 'odd'")


# ---------------------------------------------------------------------------
# FUNCTION: sample_inward_decay_mismatch
# ---------------------------------------------------------------------------
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
            boundary_mismatch_inward_shooting(
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


# ---------------------------------------------------------------------------
# FUNCTION: find_brackets_inward_shooting
# ---------------------------------------------------------------------------
def find_brackets_inward_shooting(
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


# ---------------------------------------------------------------------------
# FUNCTION: bisect_energy_inward_shooting
# ---------------------------------------------------------------------------
def bisect_energy_inward_shooting(
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
    flo = boundary_mismatch_inward_shooting(
        x_max, n_grid, potential_fn, potential_kwargs, lo, parity
    )
    fhi = boundary_mismatch_inward_shooting(
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
        fmid = boundary_mismatch_inward_shooting(
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
    # If the loop hit the iteration cap, return the midpoint of the last valid
    # bracket together with its mismatch.
    return mid, boundary_mismatch_inward_shooting(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        mid,
        parity,
    )


# ---------------------------------------------------------------------------
# FUNCTION: bisection_history_inward_decay
# ---------------------------------------------------------------------------
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
    flo = boundary_mismatch_inward_shooting(
        x_max, n_grid, potential_fn, potential_kwargs, lo, parity
    )

    history: list[dict] = []
    for iteration in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = boundary_mismatch_inward_shooting(
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


# ---------------------------------------------------------------------------
# FUNCTION: solve_state_from_inward_decay_bracket
# ---------------------------------------------------------------------------
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
    energy, mismatch = bisect_energy_inward_shooting(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        parity,
        bracket,
        tol=tol,
    )

    x_desc, psi_desc = half_domain_wavefunction_inward_shooting(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        energy,
    )

    # Reverse the inward solution so the rest of the code can reuse the same
    # full-wavefunction reconstruction logic as the outward solver.
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


# ---------------------------------------------------------------------------
# FUNCTION: solve_symmetric_potential_inward_shooting
# ---------------------------------------------------------------------------
def solve_symmetric_potential_inward_shooting(
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
    parity at the origin, which produces much cleaner wavefunctions and matches
    the physical requirement that bound states decay as |x| becomes large.
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

    # Solve each parity sector separately, then merge and sort the states by
    # energy. This is cleaner than solving on the full domain for every state.
    for parity, n_needed in [("even", n_even), ("odd", n_odd)]:
        # Symmetry lets each parity sector be searched independently, which is
        # simpler than trying to recover all states from a full-domain solve.
        brackets = find_brackets_inward_shooting(
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
            # Each bracket is assumed to isolate one state in this parity
            # sector, so it can be solved independently and appended.
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


# ---------------------------------------------------------------------------
# FUNCTION: solve_symmetric_potential_outward_shooting
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def solve_symmetric_potential_outward_shooting(
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
        Optional dictionary of named parameters forwarded to `potential_fn`
        as `potential_fn(x, **potential_kwargs)`, for example well depth,
        width, or oscillator frequency.
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

    Notes
    -----
    This is the standard outward half-domain solver used for bounded or
    effectively boxed problems, where checking the right-edge leakage is a
    stable way to enforce the second boundary condition.
    """

    # If there are no user-specified potential kwargs, use an empty dict to avoid
    # passing None to the potential function.
    if potential_kwargs is None:
        potential_kwargs = {}

    x_half = np.linspace(0.0, x_max, n_grid)
    V_half = potential_fn(x_half, **potential_kwargs)

    if e_min is None:
        # Bound states typically start near the minimum of the potential.
        e_min = float(np.min(V_half))
    if e_max is None:
        # The scan ceiling is chosen generously so several low-lying states fit
        # inside the search window without extra user tuning.
        e_max = float(np.max(V_half))
        e_max = max(e_max, e_min + 30.0)

    solutions: list[StateSolution] = []

    for parity, n_needed in [("even", n_even), ("odd", n_odd)]:
        # Solve even and odd sectors separately, then merge by energy.
        brackets = find_brackets_outward_shooting(
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
                solve_state_from_bracket_outward_shooting(
                    x_half, V_half, parity, bracket, tol=tol
                )
            )

    solutions.sort(key=lambda s: s.energy)

    return solutions
