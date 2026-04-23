from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> None:
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
    _ensure_parent(path)
    plt.figure(figsize=(8, 5))

    V_plot = np.clip(V, None, 50)  # cap huge walls
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
    _ensure_parent(path)
    plt.figure(figsize=(7, 4.5))
    
    for i in range(errors.shape[1]):
        plt.loglog(xvals, errors[:, i], marker="o", label=f"state {i}")
    
    plt.xlabel(xlabel)
    plt.ylabel("absolute energy error")
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
