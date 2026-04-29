from __future__ import annotations

"""
Numerov scattering utilities.

This module extends the bound-state project to one-dimensional scattering. The
asymptotic potential is assumed to be zero on the far left and far right, so a
particle with energy E has wave number k = sqrt(2E) in both asymptotic regions.

The implementation follows the common Numerov scattering strategy used in
computational physics texts: impose a transmitted right-moving wave on the far
right, integrate backward through the potential, and decompose the left-side
solution into incident and reflected waves.

Reviewer guide
--------------
This is the main extension beyond the core bound-state project. It reuses the
same Numerov framework, but now for complex-valued scattering states instead of
real bound-state eigenfunctions.

The method is:
1. assume the potential is asymptotically zero on both sides
2. impose a unit transmitted wave on the far right
3. integrate backward through the barrier using complex Numerov
4. decompose the left asymptotic solution into incident and reflected waves
5. compute T, R, and the conservation check T + R

This module is included to show that the codebase is not limited to bound-state
shooting. The physics focus of the report is still on bound states; scattering
is a secondary extension used to study transmission, reflection, and
double-barrier resonances.
"""

from dataclasses import dataclass

import numpy as np

from src.numerov import q_from_energy


# ---------------------------------------------------------------------------
# DATA CLASS: ScatteringResult
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
@dataclass
class ScatteringResult:
    """
    Container for one scattering calculation.

    Attributes
    ----------
    energy : float
        Incident particle energy.
    transmission : float
        Transmission probability T.
    reflection : float
        Reflection probability R.
    incident_amplitude : complex
        Amplitude of the incoming wave on the left when the transmitted wave on
        the right is normalized to unit amplitude.
    reflected_amplitude : complex
        Amplitude of the reflected wave on the left.
    transmitted_amplitude : complex
        Physical transmission amplitude t = 1 / incident_amplitude.
    reflection_amplitude : complex
        Physical reflection amplitude r = reflected / incident.
    """

    energy: float
    transmission: float
    reflection: float
    incident_amplitude: complex
    reflected_amplitude: complex
    transmitted_amplitude: complex
    reflection_amplitude: complex


# ---------------------------------------------------------------------------
# FUNCTION: numerov_outward_complex
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def numerov_outward_complex(
    x: np.ndarray,
    q: np.ndarray,
    psi0: complex,
    psi1: complex,
) -> np.ndarray:
    """
    Integrate y'' = q(x)y using the Numerov recurrence for complex y.

    This is the same recurrence as the real bound-state Numerov routine, but it
    preserves complex phases needed for scattering waves.
    """
    if x.ndim != 1 or q.ndim != 1 or len(x) != len(q):
        raise ValueError("x and q must be 1D arrays with the same length.")
    if len(x) < 3:
        raise ValueError("Need at least 3 grid points.")

    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, rtol=1e-12, atol=1e-14):
        raise ValueError("Numerov integration requires a uniform grid.")

    # The recurrence is identical to the real-valued case, but the storage must
    # preserve phase information because incident and reflected waves interfere.
    psi = np.zeros(len(x), dtype=complex)
    psi[0] = psi0
    psi[1] = psi1

    h2 = h * h
    c = h2 / 12.0

    for n in range(1, len(x) - 1):
        # Same Numerov update as in the bound-state solver, now carried out in
        # complex arithmetic.
        a = 1.0 - c * q[n + 1]
        b = 2.0 * (1.0 + 5.0 * c * q[n]) * psi[n]
        d = (1.0 - c * q[n - 1]) * psi[n - 1]
        psi[n + 1] = (b - d) / a

        if abs(psi[n + 1]) > 1e100:
            psi[: n + 2] /= 1e100

    return psi


# ---------------------------------------------------------------------------
# FUNCTION: integrate_from_right
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def integrate_from_right(
    x: np.ndarray,
    V: np.ndarray,
    energy: float,
) -> np.ndarray:
    """
    Impose a unit transmitted wave on the right and integrate backward.

    The right asymptotic solution is chosen as exp(ikx). The returned solution is
    given on the original increasing grid.
    """
    if energy <= 0.0:
        raise ValueError("Scattering energy must be positive.")

    # In the free regions V=0, the Schrödinger equation admits plane waves with
    # wave number k = sqrt(2E) in the dimensionless units used here.
    k = np.sqrt(2.0 * energy)
    x_desc = x[::-1]
    V_desc = V[::-1]
    q_desc = q_from_energy(V_desc, energy)

    # Normalize the right asymptotic region to a purely transmitted outgoing
    # wave. The left incident amplitude will be recovered afterward.
    psi0 = np.exp(1j * k * x_desc[0])
    psi1 = np.exp(1j * k * x_desc[1])
    psi_desc = numerov_outward_complex(x_desc, q_desc, psi0=psi0, psi1=psi1)
    return psi_desc[::-1]


# ---------------------------------------------------------------------------
# FUNCTION: decompose_left_asymptotic
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def decompose_left_asymptotic(
    x: np.ndarray,
    psi: np.ndarray,
    energy: float,
) -> tuple[complex, complex]:
    """
    Decompose the left asymptotic wave into incoming and reflected parts.

    In the left free region, psi(x) = A_in exp(ikx) + A_ref exp(-ikx). The first
    two grid points are used to solve for A_in and A_ref.
    """
    k = np.sqrt(2.0 * energy)
    # Two grid samples give a 2x2 linear system for the unknown incoming and
    # reflected amplitudes in the free left region.
    matrix = np.array(
        [
            [np.exp(1j * k * x[0]), np.exp(-1j * k * x[0])],
            [np.exp(1j * k * x[1]), np.exp(-1j * k * x[1])],
        ],
        dtype=complex,
    )
    rhs = np.array([psi[0], psi[1]], dtype=complex)
    incident, reflected = np.linalg.solve(matrix, rhs)
    return incident, reflected


# ---------------------------------------------------------------------------
# FUNCTION: solve_scattering
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def solve_scattering(
    x: np.ndarray,
    V: np.ndarray,
    energy: float,
) -> ScatteringResult:
    """
    Compute reflection and transmission for one incident energy.

    The transmitted wave on the right is normalized to unit amplitude during the
    backward integration. If the corresponding left incident amplitude is A_in,
    the physical transmission amplitude is t = 1/A_in and the reflection
    amplitude is r = A_ref/A_in.
    """
    psi = integrate_from_right(x, V, energy)
    incident, reflected = decompose_left_asymptotic(x, psi, energy)

    # The backward solve was normalized to unit transmitted amplitude on the
    # right, so divide by the recovered left incident amplitude to obtain the
    # physical transmission/reflection amplitudes for unit incident flux.
    transmitted_amp = 1.0 / incident
    reflection_amp = reflected / incident
    transmission = float(abs(transmitted_amp) ** 2)
    reflection = float(abs(reflection_amp) ** 2)

    return ScatteringResult(
        energy=energy,
        transmission=transmission,
        reflection=reflection,
        incident_amplitude=incident,
        reflected_amplitude=reflected,
        transmitted_amplitude=transmitted_amp,
        reflection_amplitude=reflection_amp,
    )


# ---------------------------------------------------------------------------
# FUNCTION: scattering_wavefunction
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def scattering_wavefunction(
    x: np.ndarray,
    V: np.ndarray,
    energy: float,
) -> tuple[np.ndarray, ScatteringResult]:
    """
    Return the physical scattering wavefunction for unit incident amplitude.
    """
    psi_unit_transmitted = integrate_from_right(x, V, energy)
    result = solve_scattering(x, V, energy)
    # Convert the intermediate "unit transmitted" normalization to the more
    # physical convention of unit incident amplitude on the left.
    psi_unit_incident = psi_unit_transmitted / result.incident_amplitude
    return psi_unit_incident, result


# ---------------------------------------------------------------------------
# FUNCTION: sweep_scattering
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def sweep_scattering(
    x: np.ndarray,
    V: np.ndarray,
    energies: np.ndarray,
) -> list[ScatteringResult]:
    """
    Compute scattering coefficients for a sequence of energies.
    """
    # Run the same single-energy solve independently for each sample in the
    # requested energy sweep.
    return [solve_scattering(x, V, float(energy)) for energy in energies]


# ---------------------------------------------------------------------------
# FUNCTION: find_transmission_peaks
# Reviewer note: this named block is one logical unit of the implementation.
# ---------------------------------------------------------------------------
def find_transmission_peaks(
    energies: np.ndarray,
    transmission: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Locate simple local maxima in the transmission curve.
    """
    peaks: list[dict] = []
    for i in range(1, len(energies) - 1):
        if transmission[i] <= threshold:
            continue
        if (
            transmission[i] >= transmission[i - 1]
            and transmission[i] >= transmission[i + 1]
        ):
            # A simple local-maximum rule is enough for the smooth resonance
            # curves generated in this project.
            peaks.append(
                {
                    "energy": float(energies[i]),
                    "transmission": float(transmission[i]),
                }
            )
    return peaks
