from __future__ import annotations

"""
High-level numerical experiments used in the project report.

Each function in this module runs one physical case study, writes tabulated data
to CSV, and saves the associated figures into the results directory.
"""

from pathlib import Path

import numpy as np

from src.analysis import (
    convergence_vs_box_size,
    convergence_vs_grid,
    estimate_convergence_slopes,
    exact_harmonic_oscillator_energies,
    exact_square_well_energies,
    save_csv_rows,
    splitting_vs_parameter,
)
from src.plotting import (
    plot_energy_comparison,
    plot_error_curve,
    plot_potential_and_states,
    plot_probability_densities,
    plot_root_finding_diagnostic,
    plot_splitting_curve,
)
from src.potentials import (
    finite_square_well,
    harmonic_oscillator,
    infinite_square_well_numeric,
    quartic_double_well,
    quartic_oscillator,
)
from src.shooting import (
    bisection_history,
    bisection_history_inward_decay,
    find_brackets,
    find_inward_decay_brackets,
    sample_boundary_mismatch,
    sample_inward_decay_mismatch,
    solve_symmetric_potential,
    solve_symmetric_potential_inward_decay,
)


def plot_infinite_well_root_diagnostics(results_dir: Path, a: float = 1.0) -> None:
    """
    Plot shooting/root-finding diagnostics for all four infinite-well states.

    Even and odd states use different parity boundary conditions at x = 0, so
    the cleanest visualization is two separate mismatch plots:
    - even sector: global states n=0 and n=2
    - odd sector: global states n=1 and n=3
    """
    x_half = np.linspace(0.0, a, 900)
    V_half = infinite_square_well_numeric(x_half, a=a, wall_height=1e6)

    diagnostic_specs = [
        {
            "parity": "even",
            "e_min": 0.1,
            "e_max": 15.0,
            "state_labels": ["state 0, even", "state 2, even"],
            "path": results_dir / "1_infinite_square_well_root_finding_even.png",
            "title": "Infinite well shooting roots, even states",
            "mismatch_label": r"scaled mismatch: $M(E)/\max |M|$, $M(E)=\psi_E(a)$",
        },
        {
            "parity": "odd",
            "e_min": 2.0,
            "e_max": 25.0,
            "state_labels": ["state 1, odd", "state 3, odd"],
            "path": results_dir / "1_infinite_square_well_root_finding_odd.png",
            "title": "Infinite well shooting roots, odd states",
            "mismatch_label": r"scaled mismatch: $M(E)/\max |M|$, $M(E)=\psi_E(a)$",
        },
    ]

    for spec in diagnostic_specs:
        energies_scan, mismatches_scan = sample_boundary_mismatch(
            x_half,
            V_half,
            parity=spec["parity"],
            e_min=spec["e_min"],
            e_max=spec["e_max"],
            n_scan=1600,
        )
        brackets = find_brackets(
            x_half,
            V_half,
            parity=spec["parity"],
            e_min=spec["e_min"],
            e_max=spec["e_max"],
            n_scan=1600,
        )
        histories = [
            bisection_history(x_half, V_half, spec["parity"], bracket, max_iter=30)
            for bracket in brackets[: len(spec["state_labels"])]
        ]

        plot_root_finding_diagnostic(
            energies_scan,
            mismatches_scan,
            histories,
            spec["path"],
            spec["title"],
            history_labels=spec["state_labels"],
            mismatch_label=spec["mismatch_label"],
        )



def plot_harmonic_oscillator_root_diagnostics(
    results_dir: Path,
    omega: float = 1.0,
    x_max: float = 8.0,
) -> None:
    """
    Plot shooting/root-finding diagnostics for the first four harmonic-oscillator states.

    Even and odd harmonic-oscillator states use different parity boundary conditions:
    - even sector: global states n=0 and n=2
    - odd sector: global states n=1 and n=3

    The mismatch is now sampled with the stable inward-shooting formulation.
    The roots enforce parity at the origin: M(E)=psi'_E(0) for even states and
    M(E)=psi_E(0) for odd states.
    """

    diagnostic_specs = [
        {
            "parity": "even",
            "e_min": 0.1,
            "e_max": 3.2,
            "state_labels": ["state 0, even", "state 2, even"],
            "path": results_dir / "2_harmonic_oscillator_root_finding_even.png",
            "title": "Harmonic oscillator shooting roots, even states",
            "mismatch_label": (
                r"scaled mismatch: $M(E)/\max |M|$, "
                r"$M(E)=\psi'_E(0)$"
            ),
        },
        {
            "parity": "odd",
            "e_min": 0.7,
            "e_max": 4.3,
            "state_labels": ["state 1, odd", "state 3, odd"],
            "path": results_dir / "2_harmonic_oscillator_root_finding_odd.png",
            "title": "Harmonic oscillator shooting roots, odd states",
            "mismatch_label": (
                r"scaled mismatch: $M(E)/\max |M|$, "
                r"$M(E)=\psi_E(0)$"
            ),
        },
    ]

    for spec in diagnostic_specs:
        energies_scan, mismatches_scan = sample_inward_decay_mismatch(
            x_max=x_max,
            n_grid=500,
            potential_fn=harmonic_oscillator,
            potential_kwargs={"omega": omega},
            parity=spec["parity"],
            e_min=spec["e_min"],
            e_max=spec["e_max"],
            n_scan=400,
        )
        brackets = find_inward_decay_brackets(
            x_max=x_max,
            n_grid=500,
            potential_fn=harmonic_oscillator,
            potential_kwargs={"omega": omega},
            parity=spec["parity"],
            e_min=spec["e_min"],
            e_max=spec["e_max"],
            n_scan=400,
        )
        histories = [
            bisection_history_inward_decay(
                x_max=x_max,
                n_grid=1600,
                potential_fn=harmonic_oscillator,
                potential_kwargs={"omega": omega},
                parity=spec["parity"],
                bracket=bracket,
                max_iter=30,
            )
            for bracket in brackets[: len(spec["state_labels"])]
        ]

        plot_root_finding_diagnostic(
            energies_scan,
            mismatches_scan,
            histories,
            spec["path"],
            spec["title"],
            history_labels=spec["state_labels"],
            mismatch_label=spec["mismatch_label"],
        )

def run_square_well(results_dir: Path) -> None:
    """
    Run the infinite square well benchmark case.

    The actual infinite-well calculation is done on [0, a], not [0, 1.2a].
    This is important because the exact boundary condition is psi(a)=0.
    If x_max is larger than a, the convergence plot mixes grid error with the
    error caused by replacing an infinite wall with a finite numerical barrier.
    """
    a = 1.0
    x_max = a
    n_grid = 2500

    states = solve_symmetric_potential(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1e6},
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=80.0,
    )

    numerical = np.array([s.energy for s in states[:4]])
    exact = exact_square_well_energies(np.arange(1, 5), a=a)

    rows = []
    for i, (en, ex) in enumerate(zip(numerical, exact)):
        rows.append(
            {
                "state_index": i,
                "parity": states[i].parity,
                "numerical_energy": en,
                "exact_energy": ex,
                "relative_error": abs((en - ex) / ex),
            }
        )
    save_csv_rows(results_dir / "1_infinite_square_well_energies.csv", rows)

    # For visualization only, draw the artificial walls slightly outside the
    # solved interval. The states themselves are still computed on [-a, a].
    x_plot = np.linspace(-1.2 * a, 1.2 * a, 1200)
    V_plot = infinite_square_well_numeric(x_plot, a=a, wall_height=1e6)

    plot_potential_and_states(
        x_plot,
        V_plot,
        states,
        results_dir / "1_infinite_square_well_states.png",
        "Infinite well states",
        potential_label=r"$V(x)=0$ for $|x|\leq a$, $V_0$ for $|x|>a$ (numerically: $V_0 \gg 1$)",
    )
    plot_probability_densities(
        states,
        results_dir / "1_infinite_square_well_densities.png",
        "Infinite well densities",
    )
    plot_energy_comparison(
        numerical,
        exact,
        results_dir / "1_infinite_square_well_energy_comparison.png",
        "Infinite well energies",
        exact_label=r"exact: $E_n=\frac{(n+1)^2\pi^2}{8a^2}$",
        numerical_label="numerical",
    )

    conv = convergence_vs_grid(
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1e6},
        x_max=a,
        grid_sizes=[50, 80, 120, 180, 260, 400],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=60.0,
        reference_energies=exact[:3],
    )
    conv_slopes = estimate_convergence_slopes(conv["h"], conv["energy_errors"])
    save_csv_rows(results_dir / "1_infinite_square_well_convergence_slopes.csv", conv_slopes)

    plot_error_curve(
        conv["h"],
        conv["energy_errors"],
        "grid spacing h",
        results_dir / "1_infinite_square_well_convergence_vs_h.png",
        "Infinite well convergence",
        slopes=conv_slopes,
    )

    plot_infinite_well_root_diagnostics(results_dir, a=a)


def run_harmonic_oscillator(results_dir: Path) -> None:
    """
    Run the harmonic-oscillator benchmark case.

    This experiment validates the solver against the exact ladder spectrum and
    produces both grid-refinement and box-size convergence studies.
    """
    omega = 1.0
    x_max = 8.0
    n_grid = 2500

    states = solve_symmetric_potential_inward_decay(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=6.5,
    )

    numerical = np.array([s.energy for s in states[:4]])
    exact = exact_harmonic_oscillator_energies(np.arange(4), omega=omega)

    rows = []
    for i, (en, ex) in enumerate(zip(numerical, exact)):
        rows.append(
            {
                "state_index": i,
                "parity": states[i].parity,
                "numerical_energy": en,
                "exact_energy": ex,
                "relative_error": abs((en - ex) / ex),
            }
        )
    save_csv_rows(results_dir / "2_harmonic_oscillator_energies.csv", rows)

    x = states[0].x_full
    V = harmonic_oscillator(x, omega=omega)
    plot_potential_and_states(
        x,
        V,
        states,
        results_dir / "2_harmonic_oscillator_states.png",
        "Harmonic oscillator states",
        potential_label=r"$V(x)=\frac{1}{2}\omega^2x^2$",
    )
    plot_probability_densities(
        states,
        results_dir / "2_harmonic_oscillator_densities.png",
        "Harmonic oscillator densities",
    )
    plot_energy_comparison(
        numerical,
        exact,
        results_dir / "2_harmonic_oscillator_energy_comparison.png",
        "Harmonic oscillator energies",
        exact_label=r"exact: $E_n=\omega\left(n+\frac{1}{2}\right)$",
    )
    plot_harmonic_oscillator_root_diagnostics(
        results_dir,
        omega=omega,
        x_max=x_max,
    )

    conv_h = convergence_vs_grid(
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max=x_max,
        grid_sizes=[500, 800, 1200, 1800, 2500],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=5.0,
        reference_energies=exact[:3],
    )
    conv_h_slopes = estimate_convergence_slopes(conv_h["h"], conv_h["energy_errors"])
    save_csv_rows(results_dir / "2_harmonic_convergence_slopes.csv", conv_h_slopes)

    plot_error_curve(
        conv_h["h"],
        conv_h["energy_errors"],
        "grid spacing h",
        results_dir / "2_harmonic_convergence_vs_h.png",
        "Harmonic oscillator convergence vs h",
        slopes=conv_h_slopes,
    )

    conv_box = convergence_vs_box_size(
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max_values=[4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        n_grid=2500,
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=5.0,
        reference_energies=exact[:3],
    )
    plot_error_curve(
        conv_box["x_max"],
        conv_box["energy_errors"],
        "box size x_max",
        results_dir / "2_harmonic_convergence_vs_x_max.png",
        "Harmonic oscillator convergence vs box size",
    )


def run_double_well(results_dir: Path) -> None:
    """
    Run the quartic double-well study.

    In addition to plotting the low-lying states, this experiment sweeps the
    double-well parameter b and records the tunneling splitting E1 - E0.
    """
    base_kwargs = {"a": 1.0, "b": 6.0, "shift_min_to_zero": True}
    x_max = 3.0
    n_grid = 3000

    states = solve_symmetric_potential(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=quartic_double_well,
        potential_kwargs=base_kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )

    rows = []
    for i, s in enumerate(states[:4]):
        rows.append(
            {
                "state_index": i,
                "parity": s.parity,
                "energy": s.energy,
                "mismatch": s.mismatch,
            }
        )
    save_csv_rows(results_dir / "3_double_well_energies.csv", rows)

    x = states[0].x_full
    V = quartic_double_well(x, **base_kwargs)
    plot_potential_and_states(
        x,
        V,
        states,
        results_dir / "3_double_well_states.png",
        "Quartic double well",
    )
    plot_probability_densities(
        states,
        results_dir / "3_double_well_densities.png",
        "Quartic double well densities",
    )

    sweep_rows = splitting_vs_parameter(
        potential_fn=quartic_double_well,
        base_kwargs={"a": 1.0, "shift_min_to_zero": True},
        varied_param="b",
        varied_values=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        x_max=x_max,
        n_grid=n_grid,
        e_min=0.0,
        e_max=25.0,
    )
    save_csv_rows(results_dir / "3_double_well_splitting_vs_b.csv", sweep_rows)

    b_vals = np.array([r["b"] for r in sweep_rows], dtype=float)
    e0 = np.array([r["E0"] for r in sweep_rows], dtype=float)
    e1 = np.array([r["E1"] for r in sweep_rows], dtype=float)
    splitting = np.array([r["splitting"] for r in sweep_rows], dtype=float)
    plot_splitting_curve(
        b_vals,
        e0,
        e1,
        splitting,
        "double-well parameter b",
        results_dir / "3_double_well_splitting.png",
        "Quartic double well splitting",
    )


def run_finite_square_well(results_dir: Path) -> None:
    """
    Run the finite square well as an additional nontrivial potential.

    This case demonstrates that the same solver handles a finite number of bound
    states when the confining walls have finite height.
    """
    x_max = 4.0
    n_grid = 3000
    kwargs = {"V0": 12.0, "a": 1.0}

    states = solve_symmetric_potential(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=finite_square_well,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=11.9,
    )

    rows = []
    for i, s in enumerate(states[:4]):
        rows.append(
            {
                "state_index": i,
                "parity": s.parity,
                "energy": s.energy,
            }
        )
    save_csv_rows(results_dir / "4_finite_square_well_energies.csv", rows)

    x = states[0].x_full
    V = finite_square_well(x, **kwargs)
    plot_potential_and_states(
        x,
        V,
        states,
        results_dir / "4_finite_square_well_states.png",
        "Finite square well states",
    )
    plot_probability_densities(
        states,
        results_dir / "4_finite_square_well_densities.png",
        "Finite square well densities",
    )


def run_quartic_oscillator_demo(results_dir: Path) -> None:
    """
    Run an optional anharmonic quartic-oscillator demonstration.

    This figure is not required for the core project, but it is useful if you want
    one extra example showing that the solver is reusable beyond the main cases.
    """
    x_max = 6.0
    n_grid = 2400
    kwargs = {"lam": 0.1}

    states = solve_symmetric_potential(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=quartic_oscillator,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=10.0,
    )

    x = states[0].x_full
    V = quartic_oscillator(x, **kwargs)
    plot_potential_and_states(
        x,
        V,
        states,
        results_dir / "5_quartic_oscillator_states.png",
        "Quartic oscillator",
    )
