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
"""

from dataclasses import dataclass

import numpy as np

from src.numerov import q_from_energy


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

    psi = np.zeros(len(x), dtype=complex)
    psi[0] = psi0
    psi[1] = psi1

    h2 = h * h
    c = h2 / 12.0

    for n in range(1, len(x) - 1):
        a = 1.0 - c * q[n + 1]
        b = 2.0 * (1.0 + 5.0 * c * q[n]) * psi[n]
        d = (1.0 - c * q[n - 1]) * psi[n - 1]
        psi[n + 1] = (b - d) / a

        if abs(psi[n + 1]) > 1e100:
            psi[: n + 2] /= 1e100

    return psi


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

    k = np.sqrt(2.0 * energy)
    x_desc = x[::-1]
    V_desc = V[::-1]
    q_desc = q_from_energy(V_desc, energy)

    psi0 = np.exp(1j * k * x_desc[0])
    psi1 = np.exp(1j * k * x_desc[1])
    psi_desc = numerov_outward_complex(x_desc, q_desc, psi0=psi0, psi1=psi1)
    return psi_desc[::-1]


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
    psi_unit_incident = psi_unit_transmitted / result.incident_amplitude
    return psi_unit_incident, result


def sweep_scattering(
    x: np.ndarray,
    V: np.ndarray,
    energies: np.ndarray,
) -> list[ScatteringResult]:
    """
    Compute scattering coefficients for a sequence of energies.
    """
    return [solve_scattering(x, V, float(energy)) for energy in energies]


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
        if transmission[i] >= transmission[i - 1] and transmission[i] >= transmission[i + 1]:
            peaks.append(
                {
                    "energy": float(energies[i]),
                    "transmission": float(transmission[i]),
                }
            )
    return peaks
