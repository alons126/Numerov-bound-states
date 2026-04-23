from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import (
    convergence_vs_box_size,
    convergence_vs_grid,
    exact_harmonic_oscillator_energies,
    exact_square_well_energies,
    save_energy_table,
)
from src.plotting import (
    plot_energy_comparison,
    plot_error_curve,
    plot_potential_and_states,
    plot_probability_densities,
)
from src.potentials import (
    harmonic_oscillator,
    infinite_square_well_numeric,
    shifted_double_well_quartic,
)
from src.shooting import solve_symmetric_potential
from tests.test_solver import run_all_tests


RESULTS = Path("results")


def run_square_well() -> None:
    a = 1.0
    x_max = 1.2
    n_grid = 2500

    states = solve_symmetric_potential(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1.0e6},
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
    save_energy_table(RESULTS / "square_well_energies.csv", rows)

    x = states[0].x_full
    V = infinite_square_well_numeric(x, a=a, wall_height=1.0e6)

    plot_potential_and_states(
        x=x,
        V=V,
        states=states,
        path=RESULTS / "square_well_states.png",
        title="Infinite square well: potential and eigenstates",
    )
    plot_probability_densities(
        states=states,
        path=RESULTS / "square_well_densities.png",
        title="Infinite square well: probability densities",
    )
    plot_energy_comparison(
        numerical=numerical,
        exact=exact,
        path=RESULTS / "square_well_energy_comparison.png",
        title="Infinite square well: exact vs numerical energies",
    )

    conv = convergence_vs_grid(
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1.0e6},
        x_max=x_max,
        grid_sizes=[500, 800, 1200, 1800, 2500],
        exact_energies=exact[:3],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=60.0,
    )
    plot_error_curve(
        xvals=conv["h"],
        errors=conv["energy_errors"],
        xlabel="grid spacing h",
        path=RESULTS / "square_well_convergence_vs_h.png",
        title="Infinite square well: convergence with grid spacing",
    )


def run_harmonic_oscillator() -> None:
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
    save_energy_table(RESULTS / "harmonic_oscillator_energies.csv", rows)

    x = states[0].x_full
    V = harmonic_oscillator(x, omega=omega)

    plot_potential_and_states(
        x=x,
        V=V,
        states=states,
        path=RESULTS / "harmonic_oscillator_states.png",
        title="Harmonic oscillator: potential and eigenstates",
    )
    plot_probability_densities(
        states=states,
        path=RESULTS / "harmonic_oscillator_densities.png",
        title="Harmonic oscillator: probability densities",
    )
    plot_energy_comparison(
        numerical=numerical,
        exact=exact,
        path=RESULTS / "harmonic_oscillator_energy_comparison.png",
        title="Harmonic oscillator: exact vs numerical energies",
    )

    conv_h = convergence_vs_grid(
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max=x_max,
        grid_sizes=[500, 800, 1200, 1800, 2500],
        exact_energies=exact[:3],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=5.0,
    )
    plot_error_curve(
        xvals=conv_h["h"],
        errors=conv_h["energy_errors"],
        xlabel="grid spacing h",
        path=RESULTS / "harmonic_oscillator_convergence_vs_h.png",
        title="Harmonic oscillator: convergence with grid spacing",
    )

    conv_box = convergence_vs_box_size(
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max_values=[4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        n_grid=2500,
        exact_energies=exact[:3],
        n_even=2,
        n_odd=1,
        e_min=0.1,
        e_max=5.0,
    )
    plot_error_curve(
        xvals=conv_box["x_max"],
        errors=conv_box["energy_errors"],
        xlabel="box size x_max",
        path=RESULTS / "harmonic_oscillator_convergence_vs_xmax.png",
        title="Harmonic oscillator: convergence with box size",
    )


def run_double_well() -> None:
    x_max = 3.0
    n_grid = 3000

    states = solve_symmetric_potential(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=shifted_double_well_quartic,
        potential_kwargs={"a": 1.0, "b": 6.0},
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )

    rows = []
    for i, state in enumerate(states[:4]):
        rows.append(
            {
                "state_index": i,
                "parity": state.parity,
                "energy": state.energy,
                "mismatch": state.mismatch,
            }
        )
    save_energy_table(RESULTS / "double_well_energies.csv", rows)

    x = states[0].x_full
    V = shifted_double_well_quartic(x, a=1.0, b=6.0)

    plot_potential_and_states(
        x=x,
        V=V,
        states=states,
        path=RESULTS / "double_well_states.png",
        title="Double well: potential and eigenstates",
    )
    plot_probability_densities(
        states=states,
        path=RESULTS / "double_well_densities.png",
        title="Double well: probability densities",
    )


def print_summary() -> None:
    sq = np.genfromtxt(RESULTS / "square_well_energies.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    ho = np.genfromtxt(RESULTS / "harmonic_oscillator_energies.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    dw = np.genfromtxt(RESULTS / "double_well_energies.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")

    print("\nSquare well relative errors:")
    for row in sq:
        print(f"  n={row['state_index']}: rel. error = {row['relative_error']:.3e}")

    print("\nHarmonic oscillator relative errors:")
    for row in ho:
        print(f"  n={row['state_index']}: rel. error = {row['relative_error']:.3e}")

    print("\nDouble well lowest energies:")
    for row in dw:
        print(f"  n={row['state_index']}, parity={row['parity']}, E = {row['energy']:.6f}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    run_square_well()
    run_harmonic_oscillator()
    run_double_well()
    run_all_tests()
    print_summary()
    print(f"\nDone. Results written to: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()