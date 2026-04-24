from __future__ import annotations

"""
Plotting helpers for report-ready figures.

The functions in this module keep all Matplotlib logic in one place so the rest
of the project code can focus on the numerical calculations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def plot_potential_and_states(
    x: np.ndarray,
    V: np.ndarray,
    states,
    path: str | Path,
    title: str,
    n_show: int = 4,
    scale: float = 0.8,
) -> None:
    """
    Plot the potential together with several shifted eigenstates.

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

    Returns
    -------
    None

    Notes
    -----
    Extremely large wall values are clipped for visualization so that the
    wavefunctions remain visible in square-well plots.
    """
    _ensure_parent(path)
    plt.figure(figsize=(8, 5))

    V_plot = np.clip(V, None, 50.0)
    plt.plot(x, V_plot, label="V(x)")

    for i, state in enumerate(states[:n_show]):
        psi_scaled = scale * state.psi_full / np.max(np.abs(state.psi_full))
        plt.plot(
            state.x_full,
            state.energy + psi_scaled,
            label=f"n={i}, {state.parity}, E={state.energy:.4f}",
        )

    plt.xlabel("x")
    plt.ylabel("Energy / shifted wavefunction")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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
    _ensure_parent(path)
    plt.figure(figsize=(8, 5))

    for i, state in enumerate(states[:n_show]):
        plt.plot(
            state.x_full,
            np.abs(state.psi_full) ** 2,
            label=f"n={i}, E={state.energy:.4f}",
        )

    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_energy_comparison(
    numerical: np.ndarray,
    exact: np.ndarray,
    path: str | Path,
    title: str,
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

    Returns
    -------
    None
    """
    _ensure_parent(path)
    n = np.arange(len(exact))

    plt.figure(figsize=(7, 4.5))
    plt.plot(n, exact, marker="o", label="exact")
    plt.plot(n, numerical, marker="s", label="numerical")
    plt.xlabel("state index n")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_error_curve(
    xvals: np.ndarray,
    errors: np.ndarray,
    xlabel: str,
    path: str | Path,
    title: str,
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

    Returns
    -------
    None
    """
    _ensure_parent(path)
    plt.figure(figsize=(7, 4.5))

    for i in range(errors.shape[1]):
        plt.loglog(xvals, errors[:, i], marker="o", label=f"state {i}")

    plt.xlabel(xlabel)
    plt.ylabel("Absolute energy error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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
    _ensure_parent(path)
    plt.figure(figsize=(7, 4.5))
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


def plot_root_finding_diagnostic(
    energies: np.ndarray,
    mismatches: np.ndarray,
    histories: list[list[dict]],
    path: str | Path,
    title: str,
    history_labels: list[str] | None = None,
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
    """
    _ensure_parent(path)
    plt.figure(figsize=(8, 5))

    scale = np.nanmax(np.abs(mismatches))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    mismatch_plot = np.clip(mismatches / scale, -1.0, 1.0)

    plt.plot(energies, mismatch_plot, label="scaled mismatch")
    plt.axhline(0.0, linestyle=":", label="target mismatch = 0")

    if history_labels is None:
        history_labels = [f"state {i} bisection" for i in range(len(histories))]

    for i, history in enumerate(histories):
        mids = np.array([row["mid"] for row in history], dtype=float)
        vals = np.array([row["mismatch_mid"] for row in history], dtype=float)
        vals = np.clip(vals / scale, -1.0, 1.0)
        label = history_labels[i] if i < len(history_labels) else f"state {i} bisection"
        plt.scatter(mids, vals, s=18, label=label)

    plt.xlabel("trial energy E")
    plt.ylabel("scaled boundary mismatch")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
