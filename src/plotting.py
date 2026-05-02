from __future__ import annotations

"""
Plotting helpers for report-ready figures.

The functions in this module keep all Matplotlib logic in one place so the rest
of the project code can focus on the numerical calculations.

This file is deliberately separate from the numerical solvers. It contains only
presentation logic: turning validated data into figures used in the report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# FUNCTION: _ensure_parent
# ---------------------------------------------------------------------------
def _ensure_parent(path: str | Path) -> None:
    """
    Create the parent directory of an output path if it does not exist.

    Parameters
    ----------
    path : str or Path
        Target file path.

    Returns
    -------
    None
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FUNCTION: plot_potential_and_states
# ---------------------------------------------------------------------------
def plot_potential_and_states(
    x: np.ndarray,
    V: np.ndarray,
    states,
    path: str | Path,
    title: str,
    n_show: int = 4,
    scale: float = 0.8,
    potential_label: str = "V(x)",
) -> None:
    """
    Plot the potential together with several shifted eigenstates.

    Each displayed wavefunction is first normalized to a comparable visual
    height, then shifted vertically by its eigenenergy before plotting. This
    produces the standard bound-state visualization in which each state is
    drawn on top of its corresponding energy level, rather than around
    y = 0 as an unshifted wavefunction would be.

    Parameters
    ----------
    x : ndarray
        Full spatial grid.
    V : ndarray
        Potential on the full grid.
    states : sequence
        Sequence of StateSolution objects.
    path : str or Path
        Output image path.
    title : str
        Figure title.
    n_show : int, optional
        Number of states to overlay.
    scale : float, optional
        Relative vertical scaling of the wavefunctions.
    potential_label : str, optional
        Legend label for the potential curve.

    Returns
    -------
    None

    Notes
    -----
    Extremely large wall values are clipped for visualization so that the
    wavefunctions remain visible in square-well plots.
    """

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    # Very tall artificial walls dominate the y-axis visually, so clip them to
    # keep the bound states legible without changing the underlying data files.
    V_plot = np.clip(V, None, 25.0)

    plt.plot(x, V_plot, label=potential_label)

    for i, state in enumerate(states[:n_show]):
        # Rescale each eigenfunction to a comparable plotting height, then
        # shift it vertically by its eigenvalue so the state sits on its level.
        psi_scaled = scale * state.psi_full / np.max(np.abs(state.psi_full))

        plt.plot(
            state.x_full,
            state.energy + psi_scaled,
            label=f"n={i}, {state.parity}, E={state.energy:.6f}",
        )

    plt.xlabel("$x$")
    plt.ylabel("Energy / shifted wavefunction")
    plt.title(title)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_probability_densities
# ---------------------------------------------------------------------------
def plot_probability_densities(
    states,
    path: str | Path,
    title: str,
    n_show: int = 4,
) -> None:
    """
    Plot |psi(x)|^2 for several states.

    Parameters
    ----------
    states : sequence
        Sequence of StateSolution objects.
    path : str or Path
        Output image path.
    title : str
        Figure title.
    n_show : int, optional
        Number of states to plot.

    Returns
    -------
    None
    """

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    for i, state in enumerate(states[:n_show]):
        plt.plot(
            state.x_full,
            np.abs(state.psi_full) ** 2,
            label=f"$n={i}$, $E={state.energy:.6f}$",
        )

    plt.xlabel("$x$")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.title(title)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_energy_comparison
# ---------------------------------------------------------------------------
def plot_energy_comparison(
    numerical: np.ndarray,
    exact: np.ndarray,
    path: str | Path,
    title: str,
    exact_label: str = "exact",
    numerical_label: str = "numerical",
) -> None:
    """
    Compare numerical and exact energy levels on the same figure.

    Parameters
    ----------
    numerical : ndarray
        Computed energy levels.
    exact : ndarray
        Exact reference values.
    path : str or Path
        Output image path.
    title : str
        Figure title.
    exact_label : str, optional
        Legend label for the exact reference curve.
    numerical_label : str, optional
        Legend label for the numerical curve.

    Returns
    -------
    None
    """

    n = np.arange(len(exact))

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    plt.plot(n, exact, marker="o", markersize=9, label=exact_label)
    plt.plot(n, numerical, marker="s", label=numerical_label)
    plt.xlabel("State index $n$")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_error_curve
# ---------------------------------------------------------------------------
def plot_error_curve(
    xvals: np.ndarray,
    errors: np.ndarray,
    xlabel: str,
    path: str | Path,
    title: str,
    slopes: list[dict] | None = None,
) -> None:
    """
    Plot absolute energy errors on log-log axes.

    Parameters
    ----------
    xvals : ndarray
        Horizontal coordinate, typically h or x_max.
    errors : ndarray
        Error array with one column per state.
    xlabel : str
        Label for the horizontal axis.
    path : str or Path
        Output image path.
    title : str
        Figure title.
    slopes : list[dict], optional
        Optional convergence fit rows. If provided, each legend entry includes
        the fitted exponent p from error approximately proportional to h^p.

    Returns
    -------
    None
    """

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    for i in range(errors.shape[1]):
        label = f"State {i}"

        if slopes is not None and i < len(slopes):
            slope = slopes[i].get("convergence_exponent_p", np.nan)

            if np.isfinite(slope):
                label = f"State ${i}$, $p = {slope:.2f}$"

        # Each column of `errors` corresponds to one state across all sampled
        # grid spacings or box sizes.
        plt.loglog(xvals, errors[:, i], marker="o", label=label)

    plt.xlabel(xlabel)
    plt.ylabel("Energy error = |numerical - exact|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_splitting_curve
# ---------------------------------------------------------------------------
def plot_splitting_curve(
    xvals: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    splitting: np.ndarray,
    xlabel: str,
    path: str | Path,
    title: str,
) -> None:
    """
    Plot the lowest two energies and their splitting versus a parameter.

    Parameters
    ----------
    xvals : ndarray
        Swept parameter values.
    e0, e1 : ndarray
        Lowest even/odd energy levels.
    splitting : ndarray
        Difference E1 - E0.
    xlabel : str
        Label for the horizontal axis.
    path : str or Path
        Output image path.
    title : str
        Figure title.

    Returns
    -------
    None
    """

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    plt.plot(xvals, e0, marker="o", label="E0")
    plt.plot(xvals, e1, marker="s", label="E1")
    plt.plot(xvals, splitting, marker="^", label="E1 - E0")
    plt.xlabel(xlabel)
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_root_finding_diagnostic
# ---------------------------------------------------------------------------
def plot_root_finding_diagnostic(
    energies: np.ndarray,
    mismatches: np.ndarray,
    histories: list[list[dict]],
    path: str | Path,
    title: str,
    history_labels: list[str] | None = None,
    mismatch_label: str = r"boundary mismatch $M(E)$",
) -> None:
    """
    Plot the shooting mismatch and bisection midpoints for selected states.

    The zero line marks the desired boundary condition. Each sequence of
    bisection points shows how the root finder narrows in on an eigenvalue.

    Parameters
    ----------
    energies : ndarray
        Trial energies used to sample the shooting mismatch.
    mismatches : ndarray
        Boundary mismatch values for the trial energies.
    histories : list[list[dict]]
        Bisection history records, one list per state/root.
    path : str or Path
        Output image path.
    title : str
        Figure title.
    history_labels : list[str], optional
        Labels for the bisection histories. If omitted, generic state labels
        are used.
    mismatch_label : str, optional
        Legend label for the mismatch curve.
    """

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    finite_curve = np.asarray(mismatches[np.isfinite(mismatches)], dtype=float)

    if finite_curve.size == 0:
        finite_curve = np.array([1.0], dtype=float)

    nonzero_abs = np.abs(finite_curve[np.abs(finite_curve) > 0.0])

    if nonzero_abs.size == 0:
        linthresh = 1.0
    else:
        # Use a small data-driven linear region around zero so the plot shows
        # both the sign of the mismatch and the wide dynamic range away from it.
        linthresh = max(1.0e-12, float(np.quantile(nonzero_abs, 0.05)))

    plt.plot(energies, mismatches, label=mismatch_label)
    plt.axhline(0.0, linestyle=":", label="Target mismatch = 0")

    if history_labels is None:
        history_labels = [f"State {i} bisection" for i in range(len(histories))]

    for i, history in enumerate(histories):
        # Each history is the sequence of bisection midpoints for one root.
        mids = np.array([row["mid"] for row in history], dtype=float)
        vals = np.array([row["mismatch_mid"] for row in history], dtype=float)
        label = history_labels[i] if i < len(history_labels) else f"State {i} bisection"

        plt.scatter(mids, vals, s=18, label=label)

    plt.yscale("symlog", linthresh=linthresh)
    plt.xlabel("Trial energy $E$")
    plt.ylabel("Boundary mismatch")
    plt.title(title)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_scattering_coefficients
# ---------------------------------------------------------------------------
def plot_scattering_coefficients(
    energies: np.ndarray,
    transmission: np.ndarray,
    reflection: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    """
    Plot transmission and reflection probabilities versus incident energy.

    Parameters
    ----------
    energies : ndarray
        Incident energies.
    transmission : ndarray
        Transmission probabilities T(E).
    reflection : ndarray
        Reflection probabilities R(E).
    path : str or Path
        Output image path.
    title : str
        Figure title.

    Returns
    -------
    None
    """

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    plt.plot(energies, transmission, label="transmission $T(E)$")
    plt.plot(energies, reflection, linestyle=":", label="reflection $R(E)$")
    plt.plot(energies, transmission + reflection, linestyle="--", label="$T(E)+R(E)$")
    plt.xlabel("Incident energy $E$")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(-0.05, 1.08)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_scattering_potential_and_probability
# ---------------------------------------------------------------------------
def plot_scattering_potential_and_probability(
    x: np.ndarray,
    V: np.ndarray,
    psi: np.ndarray,
    energy: float,
    path: str | Path,
    title: str,
) -> None:
    """
    Plot a scattering probability density together with the barrier potential.

    The potential is rescaled for visualization so that the wave intensity and
    barrier shape can be shown on one axis.

    Parameters
    ----------
    x : ndarray
        Spatial grid on which the scattering state and potential are sampled.
    V : ndarray
        Potential values evaluated on the same grid.
    psi : ndarray
        Complex scattering wavefunction whose probability density
        ``|psi(x)|^2`` is plotted.
    energy : float
        Incident scattering energy shown in the plot label.
    path : str | Path
        Output file path where the figure is written.
    title : str
        Figure title displayed above the plot.
    """

    density = np.abs(psi) ** 2
    density_scale = np.max(density)

    if density_scale == 0.0 or not np.isfinite(density_scale):
        density_scale = 1.0

    V_scale = np.max(np.abs(V))

    # Rescale the potential onto the same vertical range as the probability
    # density so both shapes can be inspected on one axis.
    V_plot = V / V_scale * density_scale if V_scale > 0.0 else V

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    plt.plot(x, density, label=rf"$|\psi(x)|^2$, $E={energy:.3f}$")
    plt.plot(x, V_plot, linestyle="--", label="rescaled $V(x)$")
    plt.xlabel("$x$")
    plt.ylabel("Relative scale")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# FUNCTION: plot_numerov_vs_rk4_errors
# ---------------------------------------------------------------------------
def plot_numerov_vs_rk4_errors(
    h_numerov: np.ndarray,
    numerov_errors: np.ndarray,
    h_rk4: np.ndarray,
    rk4_errors: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    """
    Compare Numerov and RK4 harmonic-oscillator energy errors.

    The plotted value is the maximum absolute energy error among the displayed
    low-lying states for each grid spacing. This keeps the method-comparison
    figure readable while the CSV table retains state-by-state errors.

    Parameters
    ----------
    h_numerov : ndarray
        Grid spacings used for the Numerov convergence study.
    numerov_errors : ndarray
        Numerov absolute energy-error table, with one row per grid spacing and
        one column per displayed state.
    h_rk4 : ndarray
        Grid spacings used for the RK4 convergence study.
    rk4_errors : ndarray
        RK4 absolute energy-error table, with one row per grid spacing and one
        column per displayed state.
    path : str | Path
        Output file path where the comparison figure is written.
    title : str
        Figure title displayed above the plot.
    """

    # Collapse the per-state error tables to one conservative curve per method
    # by plotting the worst low-state error at each spacing.
    numerov_max = np.max(numerov_errors, axis=1)
    rk4_max = np.max(rk4_errors, axis=1)

    # Make sure the output directory exists before plotting
    _ensure_parent(path)

    # Create a new Matplotlib figure
    plt.figure(figsize=(8, 6))

    plt.loglog(h_numerov, numerov_max, marker="o", label="Numerov, max state error")
    plt.loglog(h_rk4, rk4_max, marker="s", label="RK4, max state error")
    plt.xlabel("Grid spacing $h$")
    plt.ylabel("Max absolute energy error")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
