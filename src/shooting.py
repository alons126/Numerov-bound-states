from __future__ import annotations

"""
Parity-based shooting solver for symmetric 1D potentials.

For a symmetric potential satisfying V(-x)=V(x), low-lying eigenstates can be
chosen to have definite parity:
- even states, with psi(-x)=psi(x) and therefore psi'(0)=0,
- odd states, with psi(-x)=-psi(x) and therefore psi(0)=0.

This module exploits that symmetry to solve the bound-state problem only on the
half-domain x in [0, x_max], then reconstruct the left half by reflection. The
half-domain reduction is exact for symmetric potentials; it is not an
approximation. It reduces the amount of integration work and replaces one side
of the boundary-value problem with exact parity conditions at x=0.

This file is the bound-state eigenvalue solver. Conceptually, it turns the
Schrödinger boundary-value eigenproblem into a practical shooting procedure:
1. Choose a trial energy E
2. Integrate the corresponding trial wavefunction with Numerov
3. Evaluate a boundary mismatch M(E)
4. Bracket sign changes of M(E)
5. Refine the roots to obtain eigenvalues

Two formulations are implemented because the project studies both bounded and
unbounded confining systems:
- Outward shooting from x=0 for finite-domain or effectively boxed problems
- Inward shooting from x_max for decaying-tail problems such as the harmonic
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
"""

from dataclasses import dataclass

import numpy as np

# The shooting code depends on the same Numerov framework as the bound-state
# solver, so we can reuse the same q_from_energy function to compute the effective
# q(x) = 2(V(x) - E) for the shooting problem. The other Numerov utilities are
# also used to build the initial conditions and normalize the final wavefunction,
# so we import them here as well for convenience.
from src.numerov import (
    derivative_at_right_edge,
    normalize_wavefunction,
    numerov_march,
    q_from_energy,
)


# ===========================================================================
# DATA CLASS: StateSolution
# ===========================================================================
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


# ===========================================================================
# FUNCTION: initial_conditions_outward_shooting
# ===========================================================================
def initial_conditions_outward_shooting(
    x_half: np.ndarray, q_half: np.ndarray, parity: str
) -> tuple[float, float]:
    """
    Construct parity-consistent starting values near x = 0.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid beginning at x = 0.
    q_half : ndarray
        Numerov coefficient sampled on the same half-domain grid, with
        ``q(x) = 2 [V(x) - E]`` so the Schrödinger equation is written as
        ``psi'' = q psi`` for the current trial energy. The first few entries
        are used to build the parity-consistent Taylor startup values near the
        origin.
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


# ===========================================================================
# FUNCTION: half_domain_wavefunction_outward_shooting
# ===========================================================================
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

    # Convert this trial energy into the Numerov coefficient q(x), then choose
    # the first two wavefunction values psi0 = psi(x_0) and psi1 = psi(x_1).
    # These parity-consistent startup values encode the origin condition and
    # let the outward Numerov recurrence start on the half-domain grid.
    q = q_from_energy(V_half, energy)
    psi0, psi1 = initial_conditions_outward_shooting(x_half, q, parity)

    # March the trial solution from x = 0 to x_max using the half-domain grid,
    # the Numerov coefficient q(x), and the two startup values fixed above.
    return numerov_march(x_half, q, psi0=psi0, psi1=psi1)


# ===========================================================================
# FUNCTION: boundary_mismatch_outward_shooting
# ===========================================================================
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

    # First build the trial half-domain wavefunction for this energy and
    # parity; the outward boundary mismatch is then read off from its behavior
    # at x_max.
    psi = half_domain_wavefunction_outward_shooting(x_half, V_half, energy, parity)

    if mode == "value":
        # For outward shooting on a boxed domain, a true eigenstate should have
        # negligible leakage at the far boundary.
        return float(psi[-1])

    if mode == "logder":
        # The logarithmic derivative is a more scale-invariant alternative, but
        # most of the project diagnostics use the simpler boundary value itself.
        denom = psi[-1]
        if abs(denom) < 1e-14:
            return np.sign(denom) * np.inf if denom != 0.0 else np.inf

        return float(derivative_at_right_edge(x_half, psi) / denom)

    raise ValueError("mode must be 'value' or 'logder'")


# ===========================================================================
# FUNCTION: diagnostic_mismatch_outward_shooting
# ===========================================================================
def diagnostic_mismatch_outward_shooting(
    x_half: np.ndarray,
    V_half: np.ndarray,
    energy: float,
    parity: str,
) -> float:
    """
    Evaluate a scale-invariant outward-shooting mismatch for plotting.

    For diagnostic plots, divide the wall mismatch by the half-domain
    ``L2`` norm of the trial wavefunction so the plotted curve reflects root
    structure rather than the arbitrary amplitude of the unnormalized trial
    solution.
    """

    psi = half_domain_wavefunction_outward_shooting(x_half, V_half, energy, parity)
    scale = max(float(np.sqrt(np.trapezoid(np.abs(psi) ** 2, x_half))), 1.0e-300)

    return float(psi[-1] / scale)


# ===========================================================================
# FUNCTION: find_brackets_outward_shooting
# ===========================================================================
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
    eigenvalue interval, which is why we treat root finding as a central
    ingredient rather than a post-processing detail.

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

    # Build a uniform trial-energy scan on [e_min, e_max], then evaluate the
    # outward-shooting mismatch M(E) at each sample so neighboring sign changes
    # can be turned into root brackets.
    energies = np.linspace(e_min, e_max, n_scan)
    vals = np.array(
        [
            boundary_mismatch_outward_shooting(x_half, V_half, e, parity, mode="value")
            for e in energies
        ]
    )

    brackets: list[tuple[float, float]] = []

    # Walk through neighboring scan points and turn either an exact zero hit or
    # a sign change in M(E) into a bracket for the later root-refinement step.
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


# ===========================================================================
# FUNCTION: sample_boundary_mismatch_outward_shooting
# ===========================================================================
def sample_boundary_mismatch_outward_shooting(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 1000,
    diagnostic_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the shooting mismatch over an energy interval.

    This is used only for diagnostics and visualization. The eigenvalues are
    located where the mismatch curve crosses zero, so plotting it makes the
    root-finding logic visible instead of leaving it as a black box.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid on which the outward Numerov shooting solve is run.
    V_half : ndarray
        Potential sampled on the same half-domain grid.
    parity : str
        State parity to enforce during shooting, either ``"even"`` or
        ``"odd"``.
    e_min : float
        Lower end of the trial-energy interval to scan.
    e_max : float
        Upper end of the trial-energy interval to scan.
    n_scan : int, default=1000
        Number of equally spaced trial energies used to sample the mismatch
        curve between ``e_min`` and ``e_max``.
    diagnostic_scale : bool, default=False
        If ``True``, divide the boundary mismatch by the half-domain ``L2``
        norm of the trial solution for plotting. This preserves the zeros
        while removing the arbitrary amplitude scale of the trial solution.

    Returns
    -------
    tuple[ndarray, ndarray]
        Pair ``(energies, mismatches)`` containing the sampled trial energies
        and the corresponding boundary mismatch values.
    """

    energies = np.linspace(e_min, e_max, n_scan)

    # Sample either the diagnostics-only scaled mismatch or the raw outward
    # solver mismatch at the same trial energies, depending on what the caller
    # wants to visualize.
    if diagnostic_scale:
        mismatches = np.array(
            [
                diagnostic_mismatch_outward_shooting(x_half, V_half, e, parity)
                for e in energies
            ],
            dtype=float,
        )
    else:
        mismatches = np.array(
            [
                boundary_mismatch_outward_shooting(
                    x_half, V_half, e, parity, mode="value"
                )
                for e in energies
            ],
            dtype=float,
        )

    return energies, mismatches


# ===========================================================================
# FUNCTION: bisection_history_outward_shooting
# ===========================================================================
def bisection_history_outward_shooting(
    x_half: np.ndarray,
    V_half: np.ndarray,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 80,
    diagnostic_scale: bool = False,
) -> list[dict]:
    """
    Record the bisection process for one eigenvalue bracket.

    Returns a list of dictionaries containing the lower edge, upper edge,
    midpoint, and mismatch at each bisection iteration.

    Parameters
    ----------
    x_half : ndarray
        Half-domain grid on which the outward shooting solve is performed.
    V_half : ndarray
        Potential sampled on the same half-domain grid.
    parity : str
        State parity whose mismatch function is being bisected, either
        ``"even"`` or ``"odd"``.
    bracket : tuple[float, float]
        Energy interval ``(lo, hi)`` that must already contain a sign change
        in the outward-shooting boundary mismatch.
    tol : float, default=1e-12
        Stopping tolerance applied to both the midpoint mismatch magnitude and
        the remaining bracket width.
    max_iter : int, default=80
        Maximum number of bisection iterations to record before stopping.
    diagnostic_scale : bool, default=False
        If ``True``, record ``mismatch_mid`` after dividing by the half-domain
        ``L2`` norm for plotting only. The raw mismatch sign is still used to
        update the bisection bracket.

    Returns
    -------
    list[dict]
        Per-iteration diagnostics containing the current bracket endpoints,
        midpoint, midpoint mismatch, and bracket width.
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

        if diagnostic_scale:
            mismatch_mid_plot = diagnostic_mismatch_outward_shooting(
                x_half, V_half, mid, parity
            )
        else:
            mismatch_mid_plot = fmid

        history.append(
            {
                "iteration": iteration,
                "lo": lo,
                "hi": hi,
                "mid": mid,
                "mismatch_mid": mismatch_mid_plot,
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


# ===========================================================================
# FUNCTION: bisect_energy_outward_shooting
# ===========================================================================
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

    # Repeatedly bisect the current sign-changing bracket. Each step tests the
    # midpoint energy, then keeps only the half-interval that still contains
    # the root of the raw mismatch M(E).
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = boundary_mismatch_outward_shooting(
            x_half, V_half, mid, parity, mode="value"
        )

        if not np.isfinite(fmid):
            raise ValueError("Non-finite mismatch during bisection.")

        # Stop immediately if the midpoint already satisfies the mismatch
        # tolerance, because this energy is accurate enough for the solver.
        if abs(fmid) < tol:
            return mid, fmid

        # If the bracket itself is already smaller than tol, update it one last
        # time so the remaining interval still brackets the root, then hand the
        # tiny interval to the safeguarded secant cleanup below.
        if abs(hi - lo) < tol:
            if np.signbit(flo) != np.signbit(fmid):
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid

            break

        # Keep whichever half-interval still shows a sign change, because that
        # is the half that must still contain the eigenvalue.
        if np.signbit(flo) != np.signbit(fmid):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    # The outward-shooting solver uses this post-processing stage: once
    # bisection has safely trapped the eigenvalue inside a tiny sign-changing
    # bracket, try a few secant-style updates within that same interval to
    # reduce the final raw mismatch more aggressively without giving up the
    # robustness of the bracketing solve.
    for _ in range(8):
        denom = fhi - flo
        if denom == 0.0 or not np.isfinite(denom):
            break

        # Use the line through the current bracket endpoints to propose a
        # better root estimate. Reject the proposal immediately if it leaves
        # the last sign-changing interval, because at that point we would lose
        # the safety guarantee inherited from bisection.
        trial = lo - flo * (hi - lo) / denom
        if not (min(lo, hi) <= trial <= max(lo, hi)):
            break

        ftrial = boundary_mismatch_outward_shooting(
            x_half, V_half, trial, parity, mode="value"
        )
        if not np.isfinite(ftrial):
            break

        # Stop early if the secant step already drives the boundary mismatch
        # below the requested tolerance.
        if abs(ftrial) < tol:
            return trial, ftrial

        # Otherwise keep the half-interval that still brackets the root and
        # continue polishing inside that smaller sign-changing interval.
        if np.signbit(flo) != np.signbit(ftrial):
            hi, fhi = trial, ftrial
        else:
            lo, flo = trial, ftrial

    if abs(flo) <= abs(fhi):
        return lo, flo

    return hi, fhi


# ===========================================================================
# FUNCTION: build_full_wavefunction
# ===========================================================================
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


# ===========================================================================
# FUNCTION: solve_state_from_bracket_outward_shooting
# ===========================================================================
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

    # Refine the sign-changing energy bracket to a single eigenvalue by
    # bisection, while also returning the final raw mismatch for diagnostics.
    energy, _raw_mismatch = bisect_energy_outward_shooting(
        x_half, V_half, parity, bracket, tol=tol
    )

    # Recompute the half-domain wavefunction at the converged eigenvalue so the
    # returned state uses the final solved energy rather than a trial iterate.
    psi_half = half_domain_wavefunction_outward_shooting(x_half, V_half, energy, parity)

    # Reflect the solved half-domain state across x = 0 with the requested
    # parity sign so the final state lives on the full interval [-x_max, x_max].
    x_full, psi_full = build_full_wavefunction(x_half, psi_half, parity)

    # Scale the reconstructed full-domain state to unit norm so different
    # eigenstates are returned with the standard probability normalization.
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


# ===========================================================================
# FUNCTION: initial_conditions_inward_shooting
# ===========================================================================
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

    Parameters
    ----------
    x_desc : ndarray
        Descending half-domain grid, ordered from ``x_max`` down to ``0``, on
        which the inward Numerov integration is carried out.
    V_desc : ndarray
        Potential sampled on the same descending grid. Its first entries are
        used to estimate the forbidden-region decay rate near ``x_max``.
    energy : float
        Trial energy used to build the asymptotic tail scale
        ``sqrt(2[V(x)-E])`` at the outer boundary.

    Returns
    -------
    tuple[float, float]
        Starting values ``(psi0, psi1)`` at the first two descending-grid
        points for inward Numerov shooting.
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


# ===========================================================================
# FUNCTION: half_domain_wavefunction_inward_shooting
# ===========================================================================
def half_domain_wavefunction_inward_shooting(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    energy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the decaying tail inward from x_max to x = 0.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain. Inward shooting starts here
        in the forbidden-region tail and integrates toward the origin.
    n_grid : int
        Number of grid points on the descending half-domain mesh from
        ``x_max`` to ``0``.
    potential_fn : callable
        Potential function evaluated on the descending grid to generate the
        sampled potential profile for this trial solve.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when building the
        descending-grid potential values.
    energy : float
        Trial energy used to form the Numerov coefficient and the asymptotic
        inward-shooting startup values.

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

    # Convert this trial energy into the Numerov coefficient q(x), then choose
    # the first two wavefunction values psi0 = psi(x_0) and psi1 = psi(x_1).
    # These startup values approximate the decaying tail at x_max and let the
    # inward Numerov recurrence start on the descending half-domain grid.
    q_desc = q_from_energy(V_desc, energy)
    psi0, psi1 = initial_conditions_inward_shooting(x_desc, V_desc, energy)

    # March the trial solution from x = x_max down to x = 0 using the
    # descending grid, the Numerov coefficient q(x), and the two startup
    # values fixed above.
    psi_desc = numerov_march(x_desc, q_desc, psi0=psi0, psi1=psi1)

    return x_desc, psi_desc


# ===========================================================================
# FUNCTION: boundary_mismatch_inward_shooting
# ===========================================================================
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

    This is the inward-shooting counterpart of `boundary_mismatch_outward_shooting()`.
    Instead of starting at x=0 and checking the value at x_max, the solver starts near
    x_max from the expected decaying tail and integrates inward to the origin.
    The origin condition is then used as the scalar root-finding function M(E).

    Even states should satisfy psi'(0) = 0, so M(E)=psi'_E(0).
    Odd states should satisfy psi(0) = 0, so M(E)=psi_E(0).

    For the harmonic oscillator, this origin mismatch is preferable to checking
    the far wall after outward integration because the unphysical growing tail
    contaminates outward shooting on large domains.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function used to evaluate the confining potential on that
        descending grid.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when constructing the
        sampled potential values.
    energy : float
        Trial energy at which the inward-shooting mismatch function is
        evaluated.
    parity : str
        Symmetry sector of the trial state, either ``"even"`` or ``"odd"``,
        which determines whether the origin mismatch is taken from ``psi'(0)``
        or ``psi(0)``.

    Returns
    -------
    float
        Scalar origin mismatch used for root finding: ``psi'_E(0)`` for even
        states or ``psi_E(0)`` for odd states.
    """

    # First build the inward-shot trial wavefunction for this energy; the
    # origin mismatch is then read off from its value or derivative at x = 0.
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


# ===========================================================================
# FUNCTION: diagnostic_mismatch_inward_shooting
# ===========================================================================
def diagnostic_mismatch_inward_shooting(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    energy: float,
    parity: str,
) -> float:
    """
    Evaluate a scale-invariant inward-shooting mismatch for plotting.

    The inward solver starts from an arbitrary tail amplitude at ``x_max``.
    That amplitude does not affect the root locations, but it can make the raw
    mismatch span many orders of magnitude and visually dominate diagnostic
    plots. For plotting only, divide the origin mismatch by the half-domain
    ``L2`` norm of the trial wavefunction so the zeros are unchanged while the
    curve becomes easier to interpret.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function used to evaluate the confining potential on that
        descending grid.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when constructing the
        sampled potential values.
    energy : float
        Trial energy at which the inward-shooting diagnostic mismatch is
        evaluated.
    parity : str
        Symmetry sector of the trial state, either ``"even"`` or ``"odd"``.

    Returns
    -------
    float
        Origin mismatch divided by the half-domain ``L2`` norm.
    """

    x_desc, psi_desc = half_domain_wavefunction_inward_shooting(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=potential_fn,
        potential_kwargs=potential_kwargs,
        energy=energy,
    )
    x_asc = x_desc[::-1]
    psi_asc = psi_desc[::-1]
    scale = max(
        float(np.sqrt(np.trapezoid(np.abs(psi_asc) ** 2, x_asc))),
        1.0e-300,
    )

    if parity == "even":
        return float(derivative_at_right_edge(x_desc, psi_desc) / scale)

    if parity == "odd":
        return float(psi_desc[-1] / scale)

    raise ValueError("parity must be 'even' or 'odd'")


# ===========================================================================
# FUNCTION: sample_mismatch_inward_shooting
# ===========================================================================
def sample_mismatch_inward_shooting(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    e_min: float,
    e_max: float,
    n_scan: int = 1000,
    diagnostic_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the inward-shooting parity mismatch over an energy interval.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function evaluated on the descending grid for each trial
        energy.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when building the
        sampled potential values.
    parity : str
        State parity whose inward origin mismatch is being sampled, either
        ``"even"`` or ``"odd"``.
    e_min : float
        Lower end of the trial-energy interval to scan.
    e_max : float
        Upper end of the trial-energy interval to scan.
    n_scan : int, default=1000
        Number of equally spaced trial energies used to sample the mismatch
        curve between ``e_min`` and ``e_max``.
    diagnostic_scale : bool, default=False
        If ``True``, return a scale-invariant mismatch intended for plotting.
        This does not change the root locations, only the vertical scaling.

    Returns
    -------
    tuple[ndarray, ndarray]
        Pair ``(energies, mismatches)`` containing the sampled trial energies
        and the corresponding inward-shooting origin mismatch values.
    """

    # Build a uniform trial-energy scan on [e_min, e_max], then evaluate the
    # inward-shooting mismatch M(E) at each sample so neighboring sign changes
    # can be turned into root brackets.
    energies = np.linspace(e_min, e_max, n_scan)

    # Sample either the diagnostics-only scaled mismatch or the raw inward
    # solver mismatch at the same trial energies, depending on what the caller
    # wants to visualize.
    mismatches = np.array(
        [
            (
                boundary_mismatch_inward_shooting(
                    x_max,
                    n_grid,
                    potential_fn,
                    potential_kwargs,
                    energy,
                    parity,
                )
                if not diagnostic_scale
                else diagnostic_mismatch_inward_shooting(
                    x_max,
                    n_grid,
                    potential_fn,
                    potential_kwargs,
                    energy,
                    parity,
                )
            )
            for energy in energies
        ],
        dtype=float,
    )

    return energies, mismatches


# ===========================================================================
# FUNCTION: find_brackets_inward_shooting
# ===========================================================================
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

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function evaluated on the descending grid for each trial
        energy in the scan.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when constructing the
        sampled potential values.
    parity : str
        State parity whose inward-shooting mismatch is scanned, either
        ``"even"`` or ``"odd"``.
    e_min : float
        Lower end of the trial-energy interval to scan for sign changes.
    e_max : float
        Upper end of the trial-energy interval to scan for sign changes.
    n_scan : int, default=2000
        Number of sampled trial energies used to detect sign-changing mismatch
        intervals.

    Returns
    -------
    list[tuple[float, float]]
        Energy brackets that contain inward-shooting mismatch zeros and can be
        passed to a root-refinement routine.
    """

    energies, vals = sample_mismatch_inward_shooting(
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

    # Walk through neighboring scan points and turn either an exact zero hit or
    # a sign change in M(E) into a bracket for the later root-refinement step.
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


# ===========================================================================
# FUNCTION: bisect_energy_inward_shooting
# ===========================================================================
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

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function evaluated on the descending grid for each trial
        energy tested during bisection.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when constructing the
        sampled potential values.
    parity : str
        State parity whose inward mismatch root is being refined, either
        ``"even"`` or ``"odd"``.
    bracket : tuple[float, float]
        Energy interval ``(lo, hi)`` that must already contain a sign change in
        the inward-shooting mismatch.
    tol : float, default=1e-12
        Stopping tolerance applied to both the mismatch magnitude and the
        remaining bracket width.
    max_iter : int, default=200
        Maximum number of bisection iterations before the routine returns the
        midpoint of the last valid bracket.

    Returns
    -------
    tuple[float, float]
        Refined energy estimate and its final inward-shooting mismatch value.
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

    # Repeatedly bisect the current sign-changing bracket. Each step tests the
    # midpoint energy, then keeps only the half-interval that still contains
    # the root of the raw mismatch M(E).
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

        # Stop immediately if the midpoint already satisfies the mismatch
        # tolerance, because this energy is accurate enough for the solver.
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid, fmid

        # Keep whichever half-interval still shows a sign change, because that
        # is the half that must still contain the eigenvalue.
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


# ===========================================================================
# FUNCTION: bisection_history_inward_shooting
# ===========================================================================
def bisection_history_inward_shooting(
    x_max: float,
    n_grid: int,
    potential_fn,
    potential_kwargs: dict,
    parity: str,
    bracket: tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 80,
    diagnostic_scale: bool = False,
) -> list[dict]:
    """
    Record the inward-shooting bisection process for diagnostic plots.

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function evaluated on the descending grid for each trial
        energy visited during the recorded bisection process.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when constructing the
        sampled potential values.
    parity : str
        State parity whose inward mismatch root is being traced, either
        ``"even"`` or ``"odd"``.
    bracket : tuple[float, float]
        Energy interval ``(lo, hi)`` that must already bracket a sign change in
        the inward-shooting mismatch.
    tol : float, default=1e-12
        Stopping tolerance applied to both the midpoint mismatch magnitude and
        the remaining bracket width.
    max_iter : int, default=80
        Maximum number of bisection iterations to record before stopping.
    diagnostic_scale : bool, default=False
        If ``True``, record a scale-invariant mismatch for plotting while still
        using the raw mismatch signs to drive the bisection logic.

    Returns
    -------
    list[dict]
        Per-iteration diagnostics containing the current bracket endpoints,
        midpoint, and midpoint mismatch.
    """

    lo, hi = bracket
    flo = boundary_mismatch_inward_shooting(
        x_max, n_grid, potential_fn, potential_kwargs, lo, parity
    )

    history: list[dict] = []

    # Record the same midpoint sequence that the bisection solver would follow,
    # while optionally storing a scaled mismatch for plotting only.
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
                "mismatch_mid": (
                    diagnostic_mismatch_inward_shooting(
                        x_max,
                        n_grid,
                        potential_fn,
                        potential_kwargs,
                        mid,
                        parity,
                    )
                    if diagnostic_scale
                    else fmid
                ),
            }
        )

        if abs(fmid) < tol or abs(hi - lo) < tol:
            break

        if np.signbit(flo) != np.signbit(fmid):
            # Keep the half-interval that still brackets the zero.
            hi = mid
        else:
            lo = mid
            flo = fmid

    return history


# ===========================================================================
# FUNCTION: solve_state_from_bracket_inward_shooting
# ===========================================================================
def solve_state_from_bracket_inward_shooting(
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

    Parameters
    ----------
    x_max : float
        Outer truncation point of the half-domain used for inward shooting.
    n_grid : int
        Number of grid points on the descending mesh from ``x_max`` to ``0``.
    potential_fn : callable
        Potential function used to evaluate the confining potential on the
        inward-shooting grid.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn`` when constructing the
        sampled potential values.
    parity : str
        Symmetry sector of the desired state, either ``"even"`` or ``"odd"``.
    bracket : tuple[float, float]
        Energy interval that brackets the target eigenvalue in the selected
        parity sector.
    tol : float, default=1e-12
        Root-finding tolerance passed to the inward-shooting bisection solver.

    Returns
    -------
    StateSolution
        Normalized full-domain bound-state solution together with its energy,
        parity label, and final inward-shooting mismatch.
    """

    # Refine the sign-changing energy bracket to a single eigenvalue by
    # bisection, while also returning the final raw mismatch for diagnostics.
    energy, mismatch = bisect_energy_inward_shooting(
        x_max,
        n_grid,
        potential_fn,
        potential_kwargs,
        parity,
        bracket,
        tol=tol,
    )

    # Recompute the half-domain wavefunction at the converged eigenvalue so the
    # returned state uses the final solved energy rather than a trial iterate.
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

    # Reflect the solved half-domain state across x = 0 with the requested
    # parity sign so the final state lives on the full interval [-x_max, x_max].
    x_full, psi_full = build_full_wavefunction(x_half, psi_half, parity)

    # Scale the reconstructed full-domain state to unit norm so different
    # eigenstates are returned with the standard probability normalization.
    psi_full = normalize_wavefunction(x_full, psi_full)

    return StateSolution(
        energy=energy,
        parity=parity,
        x_full=x_full,
        psi_full=psi_full,
        mismatch=mismatch,
    )


# ===========================================================================
# FUNCTION: solve_symmetric_potential_inward_shooting
# ===========================================================================
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

    Parameters
    ----------
    x_max : float
        Outer truncation point of the symmetric half-domain.
    n_grid : int
        Number of grid points on the half-domain mesh used for inward shooting.
    potential_fn : callable
        Symmetric confining potential function to evaluate on the spatial grid.
    potential_kwargs : dict | None, default=None
        Optional keyword arguments forwarded to ``potential_fn``. If omitted,
        an empty dictionary is used.
    n_even : int, default=3
        Number of even-parity bound states to compute.
    n_odd : int, default=3
        Number of odd-parity bound states to compute.
    e_min : float | None, default=None
        Lower end of the trial-energy scan. If omitted, the solver uses the
        minimum sampled potential value on ``[0, x_max]``.
    e_max : float | None, default=None
        Upper end of the trial-energy scan. If omitted, the solver uses the
        maximum sampled potential value on ``[0, x_max]`` and enlarges it if
        needed to ensure a reasonably wide search window.
    scan_points : int, default=1200
        Number of trial energies used when scanning each parity sector for
        sign-changing mismatch brackets.
    tol : float, default=1e-12
        Root-finding tolerance passed to the inward-shooting bisection solver
        for each bracketed state.

    Returns
    -------
    list[StateSolution]
        Solved even and odd bound states, merged and sorted by increasing
        energy.
    """

    # If there are no user-specified potential kwargs, use an empty dict to avoid
    # passing None to the potential function.
    if potential_kwargs is None:
        potential_kwargs = {}

    # Probe the potential once on [0, x_max] so default energy bounds can be
    # inferred from the actual confining profile used by the inward solver.
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
        # Symmetry lets each parity sector be searched independently. Scan the
        # inward mismatch M(E) over the chosen energy window and keep only the
        # sign-changing brackets that isolate candidate roots for this sector.
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

        # Refine the first n_needed brackets into actual bound states for this
        # parity sector, then merge both parity families by energy afterward.
        for bracket in brackets[:n_needed]:
            # Each bracket is assumed to isolate one state in this parity
            # sector, so it can be solved independently and appended.
            solutions.append(
                solve_state_from_bracket_inward_shooting(
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


# ===========================================================================
# FUNCTION: solve_symmetric_potential_outward_shooting
# ===========================================================================
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

    # Solve on the symmetry-reduced half-domain [0, x_max], then evaluate the
    # chosen potential once on that grid so all later shooting scans reuse the
    # same sampled geometry.
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

    # Work one parity sector at a time. First scan the mismatch M(E) over
    # the requested energy window and collect sign-changing brackets for
    # that parity, then refine the first n_needed brackets into actual
    # bound states. The even/odd solutions are accumulated separately here
    # and sorted together by energy after the loop.
    for parity, n_needed in [("even", n_even), ("odd", n_odd)]:
        # Sample the outward-shooting mismatch over [e_min, e_max] for this
        # parity sector and keep only the energy intervals where the mismatch
        # changes sign; those intervals are the initial root brackets.
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

        # Turn the first n_needed brackets for this parity sector into actual
        # eigenstates: each bracket is refined to one eigenvalue, then the
        # corresponding wavefunction is reconstructed and stored.
        for bracket in brackets[:n_needed]:
            solutions.append(
                solve_state_from_bracket_outward_shooting(
                    x_half, V_half, parity, bracket, tol=tol
                )
            )

    solutions.sort(key=lambda s: s.energy)

    return solutions
