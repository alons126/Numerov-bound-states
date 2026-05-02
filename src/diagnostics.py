from __future__ import annotations

"""
Root-finding diagnostic figure builders for bound-state experiments.

This module centralizes the plotting orchestration for shooting diagnostics so
the experiment runners only need to call one helper per physical case.
"""

from pathlib import Path

import numpy as np

from src.plotting import plot_root_finding_diagnostic, plot_root_finding_zoom
from src.potentials import (
    finite_square_well,
    harmonic_oscillator,
    infinite_square_well_numeric,
    quartic_double_well,
)
from src.rk4_compare import RK4_bisection_history, RK4_find_brackets, RK4_sample_mismatch
from src.shooting import (
    bisection_history_inward_shooting,
    bisection_history_outward_shooting,
    find_brackets_inward_shooting,
    find_brackets_outward_shooting,
    sample_boundary_mismatch_outward_shooting,
    sample_mismatch_inward_shooting,
)


def _diagnostic_label_slug(label: str) -> str:
    """
    Convert a human-readable diagnostic label into a filename-safe suffix.

    Parameters
    ----------
    label : str
        Plot label such as ``"State 0, even"``.

    Returns
    -------
    str
        Lowercase, underscore-separated slug suitable for filenames.
    """

    return label.lower().replace(",", "").replace(" ", "_")


def _plot_outward_root_diagnostics(
    results_dir: Path,
    prefix: str,
    potential_title: str,
    x_half: np.ndarray,
    V_half: np.ndarray,
    diagnostic_specs: list[dict[str, object]],
) -> None:
    """
    Build outward-shooting diagnostic figures and per-root zoom views.

    Parameters
    ----------
    results_dir : Path
        Output directory for the generated figures.
    prefix : str
        Filename prefix for the experiment, such as
        ``"1_infinite_square_well_Numerov"``.
    potential_title : str
        Human-readable potential name used in figure titles.
    x_half : ndarray
        Half-domain grid for outward shooting.
    V_half : ndarray
        Potential sampled on the same half-domain grid.
    diagnostic_specs : list[dict[str, object]]
        Per-parity plotting specifications.
    """

    for spec in diagnostic_specs:
        parity = str(spec["parity"])
        state_labels = list(spec["state_labels"])
        e_min = float(spec["e_min"])
        e_max = float(spec["e_max"])
        mismatch_label = str(spec["mismatch_label"])

        energies_scan, mismatches_scan = sample_boundary_mismatch_outward_shooting(
            x_half,
            V_half,
            parity=parity,
            e_min=e_min,
            e_max=e_max,
            n_scan=1600,
        )
        brackets = find_brackets_outward_shooting(
            x_half,
            V_half,
            parity=parity,
            e_min=e_min,
            e_max=e_max,
            n_scan=1600,
        )
        histories = [
            bisection_history_outward_shooting(
                x_half,
                V_half,
                parity,
                bracket,
                max_iter=30,
            )
            for bracket in brackets[: len(state_labels)]
        ]

        plot_root_finding_diagnostic(
            energies_scan,
            mismatches_scan,
            histories,
            Path(spec["path"]),
            str(spec["title"]),
            history_labels=state_labels,
            mismatch_label=mismatch_label,
        )

        for label, history in zip(state_labels, histories):
            zoom_energies, zoom_mismatches = sample_boundary_mismatch_outward_shooting(
                x_half,
                V_half,
                parity=parity,
                e_min=history[0]["lo"],
                e_max=history[0]["hi"],
                n_scan=600,
            )
            plot_root_finding_zoom(
                zoom_energies,
                zoom_mismatches,
                history,
                results_dir
                / f"{prefix}_root_finding_{_diagnostic_label_slug(label)}_zoom.png",
                f"{potential_title} - root zoom - {label.lower()}",
                history_label=label,
                mismatch_label=mismatch_label,
            )


def _plot_inward_root_diagnostics(
    results_dir: Path,
    prefix: str,
    potential_title: str,
    potential_fn,
    potential_kwargs: dict,
    x_max: float,
    diagnostic_specs: list[dict[str, object]],
) -> None:
    """
    Build inward-shooting Numerov diagnostic figures and per-root zoom views.

    Parameters
    ----------
    results_dir : Path
        Output directory for the generated figures.
    prefix : str
        Filename prefix for the experiment.
    potential_title : str
        Human-readable potential name used in figure titles.
    potential_fn : callable
        Potential function used by the inward-shooting solver.
    potential_kwargs : dict
        Keyword arguments forwarded to ``potential_fn``.
    x_max : float
        Half-width of the truncated domain.
    diagnostic_specs : list[dict[str, object]]
        Per-parity plotting specifications.
    """

    for spec in diagnostic_specs:
        parity = str(spec["parity"])
        state_labels = list(spec["state_labels"])
        e_min = float(spec["e_min"])
        e_max = float(spec["e_max"])
        mismatch_label = str(spec["mismatch_label"])

        energies_scan, mismatches_scan = sample_mismatch_inward_shooting(
            x_max=x_max,
            n_grid=500,
            potential_fn=potential_fn,
            potential_kwargs=potential_kwargs,
            parity=parity,
            e_min=e_min,
            e_max=e_max,
            n_scan=400,
        )
        brackets = find_brackets_inward_shooting(
            x_max=x_max,
            n_grid=500,
            potential_fn=potential_fn,
            potential_kwargs=potential_kwargs,
            parity=parity,
            e_min=e_min,
            e_max=e_max,
            n_scan=400,
        )
        histories = [
            bisection_history_inward_shooting(
                x_max=x_max,
                n_grid=1600,
                potential_fn=potential_fn,
                potential_kwargs=potential_kwargs,
                parity=parity,
                bracket=bracket,
                max_iter=30,
            )
            for bracket in brackets[: len(state_labels)]
        ]

        plot_root_finding_diagnostic(
            energies_scan,
            mismatches_scan,
            histories,
            Path(spec["path"]),
            str(spec["title"]),
            history_labels=state_labels,
            mismatch_label=mismatch_label,
        )

        for label, history in zip(state_labels, histories):
            zoom_energies, zoom_mismatches = sample_mismatch_inward_shooting(
                x_max=x_max,
                n_grid=1600,
                potential_fn=potential_fn,
                potential_kwargs=potential_kwargs,
                parity=parity,
                e_min=history[0]["lo"],
                e_max=history[0]["hi"],
                n_scan=600,
            )
            plot_root_finding_zoom(
                zoom_energies,
                zoom_mismatches,
                history,
                results_dir
                / f"{prefix}_root_finding_{_diagnostic_label_slug(label)}_zoom.png",
                f"{potential_title} - root zoom - {label.lower()}",
                history_label=label,
                mismatch_label=mismatch_label,
            )


def plot_infinite_well_root_diagnostics(results_dir: Path, a: float = 1.0) -> None:
    """
    Plot shooting/root-finding diagnostics for the first four infinite-well states.

    Parameters
    ----------
    results_dir : Path
        Directory where the diagnostic figures for this experiment are written.
    a : float, optional
        Half-width of the square well, so the numerical domain on the half-line
        is ``[0, a]``.
    """

    x_half = np.linspace(0.0, a, 900)
    V_half = infinite_square_well_numeric(x_half, a=a, wall_height=1e6)

    _plot_outward_root_diagnostics(
        results_dir=results_dir,
        prefix="1_infinite_square_well_Numerov",
        potential_title="Infinite square well",
        x_half=x_half,
        V_half=V_half,
        diagnostic_specs=[
            {
                "parity": "even",
                "e_min": 0.1,
                "e_max": 15.0,
                "state_labels": ["State 0, even", "State 2, even"],
                "path": results_dir
                / "1_infinite_square_well_Numerov_root_finding_even.png",
                "title": "Infinite square well - shooting roots - even states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(a)$",
            },
            {
                "parity": "odd",
                "e_min": 2.0,
                "e_max": 25.0,
                "state_labels": ["State 1, odd", "State 3, odd"],
                "path": results_dir
                / "1_infinite_square_well_Numerov_root_finding_odd.png",
                "title": "Infinite square well - shooting roots - odd states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(a)$",
            },
        ],
    )


def plot_harmonic_oscillator_root_diagnostics(
    results_dir: Path,
    omega: float = 1.0,
    x_max: float = 8.0,
) -> None:
    """
    Plot shooting/root-finding diagnostics for the first four Numerov
    harmonic-oscillator states.

    Parameters
    ----------
    results_dir : Path
        Directory where the harmonic-oscillator diagnostic figures are saved.
    omega : float, optional
        Harmonic-oscillator frequency in
        ``V(x)=\\frac{1}{2}\\omega^2 x^2``.
    x_max : float, optional
        Half-width of the truncated numerical domain used by the inward solver.
    """

    _plot_inward_root_diagnostics(
        results_dir=results_dir,
        prefix="2a_harmonic_oscillator_Numerov",
        potential_title="Harmonic oscillator (Numerov)",
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max=x_max,
        diagnostic_specs=[
            {
                "parity": "even",
                "e_min": 0.1,
                "e_max": 3.2,
                "state_labels": ["State 0, even", "State 2, even"],
                "path": results_dir
                / "2a_harmonic_oscillator_Numerov_root_finding_even.png",
                "title": "Harmonic oscillator (Numerov) - shooting roots - even states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi'_E(0)$",
            },
            {
                "parity": "odd",
                "e_min": 0.7,
                "e_max": 4.3,
                "state_labels": ["State 1, odd", "State 3, odd"],
                "path": results_dir
                / "2a_harmonic_oscillator_Numerov_root_finding_odd.png",
                "title": "Harmonic oscillator (Numerov) - shooting roots - odd states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(0)$",
            },
        ],
    )


def plot_harmonic_oscillator_RK4_root_diagnostics(
    results_dir: Path,
    omega: float = 1.0,
    x_max: float = 8.0,
) -> None:
    """
    Plot shooting/root-finding diagnostics for the first four RK4
    harmonic-oscillator states.

    Parameters
    ----------
    results_dir : Path
        Directory where the RK4 diagnostic figures are saved.
    omega : float, optional
        Harmonic-oscillator frequency.
    x_max : float, optional
        Half-width of the truncated numerical domain used by the inward solver.
    """

    diagnostic_specs = [
        {
            "parity": "even",
            "e_min": 0.1,
            "e_max": 3.2,
            "state_labels": ["State 0, even", "State 2, even"],
            "path": results_dir / "2b_harmonic_oscillator_RK4_root_finding_even.png",
            "title": "Harmonic oscillator (RK4) - shooting roots - even states",
            "mismatch_label": r"Raw mismatch: $M(E)=\psi'_E(0)$",
        },
        {
            "parity": "odd",
            "e_min": 0.7,
            "e_max": 4.3,
            "state_labels": ["State 1, odd", "State 3, odd"],
            "path": results_dir / "2b_harmonic_oscillator_RK4_root_finding_odd.png",
            "title": "Harmonic oscillator (RK4) - shooting roots - odd states",
            "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(0)$",
        },
    ]

    for spec in diagnostic_specs:
        parity = str(spec["parity"])
        state_labels = list(spec["state_labels"])
        e_min = float(spec["e_min"])
        e_max = float(spec["e_max"])
        mismatch_label = str(spec["mismatch_label"])

        energies_scan, mismatches_scan = RK4_sample_mismatch(
            parity=parity,
            x_max=x_max,
            n_grid=500,
            e_min=e_min,
            e_max=e_max,
            omega=omega,
            n_scan=400,
        )
        brackets = RK4_find_brackets(
            parity=parity,
            x_max=x_max,
            n_grid=500,
            e_min=e_min,
            e_max=e_max,
            omega=omega,
            n_scan=400,
        )
        histories = [
            RK4_bisection_history(
                parity=parity,
                bracket=bracket,
                x_max=x_max,
                n_grid=1600,
                omega=omega,
                max_iter=30,
            )
            for bracket in brackets[: len(state_labels)]
        ]

        plot_root_finding_diagnostic(
            energies_scan,
            mismatches_scan,
            histories,
            Path(spec["path"]),
            str(spec["title"]),
            history_labels=state_labels,
            mismatch_label=mismatch_label,
        )

        for label, history in zip(state_labels, histories):
            zoom_energies, zoom_mismatches = RK4_sample_mismatch(
                parity=parity,
                x_max=x_max,
                n_grid=1600,
                e_min=history[0]["lo"],
                e_max=history[0]["hi"],
                omega=omega,
                n_scan=600,
            )
            plot_root_finding_zoom(
                zoom_energies,
                zoom_mismatches,
                history,
                results_dir
                / f"2b_harmonic_oscillator_RK4_root_finding_{_diagnostic_label_slug(label)}_zoom.png",
                f"Harmonic oscillator (RK4) - root zoom - {label.lower()}",
                history_label=label,
                mismatch_label=mismatch_label,
            )


def plot_double_well_root_diagnostics(
    results_dir: Path,
    potential_kwargs: dict[str, float | bool],
    x_max: float = 3.0,
) -> None:
    """
    Plot shooting/root-finding diagnostics for the first four quartic
    double-well states.

    Parameters
    ----------
    results_dir : Path
        Directory where the quartic double-well diagnostic figures are saved.
    potential_kwargs : dict[str, float | bool]
        Keyword arguments forwarded to ``quartic_double_well()``.
    x_max : float, optional
        Half-width of the half-line domain ``[0, x_max]`` used for outward
        shooting in the diagnostic plots.
    """

    x_half = np.linspace(0.0, x_max, 1200)
    V_half = quartic_double_well(x_half, **potential_kwargs)

    _plot_outward_root_diagnostics(
        results_dir=results_dir,
        prefix="3_double_well_Numerov",
        potential_title="Quartic double well",
        x_half=x_half,
        V_half=V_half,
        diagnostic_specs=[
            {
                "parity": "even",
                "e_min": 1.0,
                "e_max": 8.5,
                "state_labels": ["State 0, even", "State 2, even"],
                "path": results_dir / "3_double_well_Numerov_root_finding_even.png",
                "title": "Quartic double well - shooting roots - even states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(x_{\max})$",
            },
            {
                "parity": "odd",
                "e_min": 1.0,
                "e_max": 8.5,
                "state_labels": ["State 1, odd", "State 3, odd"],
                "path": results_dir / "3_double_well_Numerov_root_finding_odd.png",
                "title": "Quartic double well - shooting roots - odd states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(x_{\max})$",
            },
        ],
    )


def plot_finite_square_well_root_diagnostics(
    results_dir: Path,
    x_max: float = 4.0,
    V0: float = 12.0,
    a: float = 1.0,
) -> None:
    """
    Plot shooting/root-finding diagnostics for the first four finite-well states.

    Parameters
    ----------
    results_dir : Path
        Directory where the finite-square-well diagnostic figures are saved.
    x_max : float, optional
        Half-width of the outward-shooting numerical domain.
    V0 : float, optional
        Barrier height outside the well.
    a : float, optional
        Half-width of the well.
    """

    x_half = np.linspace(0.0, x_max, 1200)
    V_half = finite_square_well(x_half, V0=V0, a=a)

    _plot_outward_root_diagnostics(
        results_dir=results_dir,
        prefix="4_finite_square_well_Numerov",
        potential_title="Finite square well",
        x_half=x_half,
        V_half=V_half,
        diagnostic_specs=[
            {
                "parity": "even",
                "e_min": 0.0,
                "e_max": 11.9,
                "state_labels": ["State 0, even", "State 2, even"],
                "path": results_dir / "4_finite_square_well_Numerov_root_finding_even.png",
                "title": "Finite square well - shooting roots - even states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(x_{\max})$",
            },
            {
                "parity": "odd",
                "e_min": 0.0,
                "e_max": 11.9,
                "state_labels": ["State 1, odd", "State 3, odd"],
                "path": results_dir / "4_finite_square_well_Numerov_root_finding_odd.png",
                "title": "Finite square well - shooting roots - odd states",
                "mismatch_label": r"Raw mismatch: $M(E)=\psi_E(x_{\max})$",
            },
        ],
    )
