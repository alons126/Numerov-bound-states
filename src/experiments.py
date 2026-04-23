from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import (
    convergence_vs_box_size,
    convergence_vs_grid,
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
    plot_splitting_curve,
)
from src.potentials import (
    finite_square_well,
    harmonic_oscillator,
    infinite_square_well_numeric,
    quartic_double_well,
    quartic_oscillator,
)
from src.shooting import solve_symmetric_potential


def run_square_well(results_dir: Path) -> None:
    a = 1.0
    x_max = 1.2
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

    x = states[0].x_full
    V = infinite_square_well_numeric(x, a=a, wall_height=1e6)
    plot_potential_and_states(
        x, V, states, results_dir / "1_infinite_square_well_states.png", "Infinite square well"
    )
    plot_probability_densities(
        states,
        results_dir / "1_infinite_square_well_densities.png",
        "Infinite square well densities",
    )
    plot_energy_comparison(
        numerical,
        exact,
        results_dir / "1_infinite_square_well_energy_comparison.png",
        "Square well energies",
    )

    conv = convergence_vs_grid(
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1e6},
        x_max=x_max,
        grid_sizes=[500, 800, 1200, 1800, 2500],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=60.0,
        reference_energies=exact[:3],
    )
    
    plot_error_curve(
        conv["h"],
        conv["energy_errors"],
        "grid spacing h",
        results_dir / "1_infinite_square_well_convergence_vs_h.png",
        "Square well convergence",
    )


def run_harmonic_oscillator(results_dir: Path) -> None:
    omega = 1.0
    x_max = 8.0
    n_grid = 2500

    states = solve_symmetric_potential(
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
        "Harmonic oscillator",
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
    plot_error_curve(
        conv_h["h"],
        conv_h["energy_errors"],
        "grid spacing h",
        results_dir / "2_harmonic_convergence_vs_h.png",
        "HO convergence vs h",
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
        results_dir / "2_harmonic_convergence_vs_xmax.png",
        "HO convergence vs box size",
    )


def run_double_well(results_dir: Path) -> None:
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
        x, V, states, results_dir / "3_double_well_states.png", "Quartic double well"
    )
    plot_probability_densities(
        states, results_dir / "3_double_well_densities.png", "Double well densities"
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
        "Double-well splitting",
    )


def run_finite_square_well(results_dir: Path) -> None:
    # Finite square well: a good extra potential with easy physical interpretation.
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
        "Finite square well",
    )
    plot_probability_densities(
        states,
        results_dir / "4_finite_square_well_densities.png",
        "Finite square well densities",
    )


def run_quartic_oscillator_demo(results_dir: Path) -> None:
    # Optional second extra potential if you want one more clean figure.
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
