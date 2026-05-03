from __future__ import annotations

"""
High-level numerical experiments used in the project report.

Each function in this module runs one physical case study, writes tabulated data
to CSV, and saves the associated figures into a dedicated subdirectory under
the results directory.

This file is the experiment plan in executable form. The lower-level modules
know how to solve or analyze a problem; this module chooses which problems to
run, which outputs to save, and which comparisons support the report.

The major cases are:
- Square well: analytic validation with clean outward-shooting diagnostics
- Harmonic oscillator: analytic validation with inward shooting plus Numerov
  versus RK4
- Quartic double well: outward-shooting tunneling splitting and the most
  careful convergence study in the project
- Finite square well: additional finite-depth outward-shooting bound-state
  example
- Scattering: single- and double-barrier transmission study
"""

from pathlib import Path

import numpy as np

# Import analysis helpers used to build convergence studies, exact benchmarks,
# and CSV outputs for the experiments
from src.analysis import (
    convergence_vs_box_size_fixed_spacing,
    convergence_vs_grid,
    convergence_vs_grid_successive,
    estimate_convergence_slopes,
    exact_harmonic_oscillator_energies,
    exact_square_well_energies,
    save_csv_rows,
    splitting_vs_parameter,
)

# Import root-finding diagnostic figure builders
from src.diagnostics import (
    plot_double_well_root_diagnostics,
    plot_finite_square_well_root_diagnostics,
    plot_harmonic_oscillator_RK4_root_diagnostics,
    plot_harmonic_oscillator_root_diagnostics,
    plot_infinite_well_root_diagnostics,
)

# Import plotting helpers used to generate the report figures
from src.plotting import (
    plot_energy_comparison,
    plot_error_curve,
    plot_numerov_vs_RK4_errors,
    plot_potential_and_states,
    plot_probability_densities,
    plot_scattering_coefficients,
    plot_scattering_potential_and_probability,
    plot_splitting_curve,
)

# Import potential definitions that specify the physical systems studied by the
# experiments
from src.potentials import (
    double_square_barrier,
    finite_square_well,
    harmonic_oscillator,
    infinite_square_well_numeric,
    quartic_double_well,
    square_barrier,
)

# Import bound-state shooting diagnostics and solvers
from src.shooting import (
    solve_symmetric_potential_outward_shooting,
    solve_symmetric_potential_inward_shooting,
)

# Import scattering utilities for transmission, reflection, and resonant-state
# calculations
from src.scattering import (
    find_transmission_peaks,
    scattering_wavefunction,
    sweep_scattering,
)

# Import RK4-based harmonic-oscillator comparison routines and diagnostics
from src.rk4_compare import (
    RK4_harmonic_convergence_vs_grid,
    RK4_harmonic_convergence_vs_box_size_fixed_spacing,
    RK4_solve_harmonic_oscillator_energies,
)


# ===========================================================================
# ===========================================================================
# FUNCTION: _experiment_results_dir
# ===========================================================================
# ===========================================================================
def _experiment_results_dir(results_root: Path, name: str) -> Path:
    """
    Return the output directory for one experiment and create it if needed.

    Parameters
    ----------
    results_root : Path
        Shared results root, usually ``results/``.
    name : str
        Per-experiment directory name.

    Returns
    -------
    Path
        Created output directory for the experiment.
    """

    path = results_root / name
    path.mkdir(parents=True, exist_ok=True)

    return path


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Infinite square well
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ===========================================================================
# ===========================================================================
# FUNCTION: run_square_well
# ===========================================================================
# ===========================================================================
def run_square_well(results_dir: Path) -> None:
    """
    Run the infinite square well benchmark case.

    The actual infinite-well calculation is done on [0, a], not [0, 1.2a].
    This is important because the exact boundary condition is psi(a)=0.
    If x_max is larger than a, the convergence plot mixes grid error with the
    error caused by replacing an infinite wall with a finite numerical barrier.

    Parameters
    ----------
    results_dir : Path
        Root results directory under which this experiment creates its own
        output subdirectory.
    """

    # -----------------------------------------------------
    # Set up the experiment
    # -----------------------------------------------------

    # Create an output folder under results_dir for this experiment
    experiment_dir = _experiment_results_dir(
        results_dir, "1_infinite_square_well_Numerov"
    )

    # Define the physical and numerical parameters for the experiment
    a = 1.0  # Well half-width, so the full well is [-a, a]
    x_max = a  # The solver's spatial domain will be [-x_max, x_max]
    n_grid = 2500  # Number of grid points for the Numerov solver, including boundaries

    # -----------------------------------------------------
    # Run the experiment
    # -----------------------------------------------------

    print("Running infinite square well experiment...")

    # Run the solver to find the first four bound states of the infinite square well
    states = solve_symmetric_potential_outward_shooting(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1e6},
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=80.0,
    )

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing infinite square well results...")

    # Plot the root-finding diagnostics for all four states, with separate panels
    # for the even and odd sectors
    plot_infinite_well_root_diagnostics(experiment_dir, a=a)

    # Tabulate the numerical energies and compare them to the exact formula
    # E_n = ((n+1)^2 * pi^2) / (8 * a^2)
    numerical = np.array([s.energy for s in states[:4]])
    exact = exact_square_well_energies(np.arange(1, 5), a=a)

    # Save a CSV comparing the numerical and exact energies, along with the relative
    # error
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
    save_csv_rows(experiment_dir / "1_infinite_square_well_Numerov_energies.csv", rows)

    # For visualization only, draw the artificial walls slightly outside the
    # solved interval. The states themselves are still computed on [-a, a].
    x_plot = np.linspace(-1.2 * a, 1.2 * a, 1200)
    V_plot = infinite_square_well_numeric(x_plot, a=a, wall_height=1e6)

    plot_potential_and_states(
        x_plot,
        V_plot,
        states,
        experiment_dir / "1_infinite_square_well_Numerov_states.png",
        "Infinite square well - states",
        potential_label=r"$V(x)=0$ for $|x|\leq a$, $V_0$ for $|x|>a$ (numerically: $V_0 \gg 1$)",
    )

    plot_probability_densities(
        states,
        experiment_dir / "1_infinite_square_well_Numerov_state_densities.png",
        "Infinite square well - state densities",
    )

    plot_energy_comparison(
        numerical,
        exact,
        experiment_dir / "1_infinite_square_well_Numerov_state_energy_comparison.png",
        "Infinite square well - state energy comparison",
        exact_label=r"Exact: $E_n=\frac{(n+1)^2\pi^2}{8a^2}$",
        numerical_label="Numerical",
    )

    # Convergence study: vary the grid spacing h at fixed x_max=a, so the error is
    # purely from the discretization and not from the finite-domain truncation.
    conv = convergence_vs_grid(
        potential_fn=infinite_square_well_numeric,
        potential_kwargs={"a": a, "wall_height": 1e6},
        x_max=a,
        # Stop before the lowest-state error reaches the root-finding / floating
        # point floor; otherwise the fitted slope understates the true Numerov
        # asymptotic order.
        grid_sizes=[50, 80, 120, 180],
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=80.0,
        reference_energies=exact,
    )
    conv_slopes = estimate_convergence_slopes(conv["h"], conv["energy_errors"])
    save_csv_rows(
        experiment_dir
        / "1_infinite_square_well_Numerov_energy_convergence_vs_h_slopes.csv",
        conv_slopes,
    )

    plot_error_curve(
        conv["h"],
        conv["energy_errors"],
        "Grid spacing $h$",
        experiment_dir / "1_infinite_square_well_Numerov_energy_convergence_vs_h.png",
        "Infinite square well - energy convergence vs grid spacing $h$",
        slopes=conv_slopes,
    )


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Harmonic oscillator
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ===========================================================================
# FUNCTION: run_harmonic_oscillator_RK4_comparison
# ===========================================================================
def run_harmonic_oscillator_RK4_comparison(
    rk4_results_dir: Path,
    comparison_results_dir: Path,
    numerov_convergence: dict[str, np.ndarray],
    omega: float = 1.0,
    x_max: float = 8.0,
    target_h: float | None = None,
) -> None:
    """
    Compare the specialized Numerov integrator with general RK4 shooting.

    The harmonic oscillator is used because it has an exact spectrum and a
    smooth potential. Both methods use inward shooting, so the comparison mainly
    isolates the difference between the Numerov and RK4 integration formulas.

    Parameters
    ----------
    rk4_results_dir : Path
        Output directory for RK4-only tables and figures.
    comparison_results_dir : Path
        Output directory for direct Numerov-versus-RK4 comparison data and plots.
    numerov_convergence : dict[str, np.ndarray]
        Precomputed Numerov grid-convergence data, expected to include matching
        ``"h"`` and ``"energy_errors"`` arrays for the first four states.
    omega : float, optional
        Harmonic-oscillator frequency in the potential
        ``V(x)=\\frac{1}{2}\\omega^2 x^2``.
    x_max : float, optional
        Half-width of the truncated spatial domain used for both solvers.
    target_h : float | None, optional
        Nominal grid spacing for the optional RK4 box-size study. If ``None``,
        that box-size comparison is skipped.
    """

    # -----------------------------------------------------
    # Set up the comparison
    # -----------------------------------------------------

    n_states = 4
    numerov_h = np.asarray(numerov_convergence["h"], dtype=float)
    numerov_errors = np.asarray(numerov_convergence["energy_errors"], dtype=float)

    # Convert the Numerov spacing samples back into integer grid sizes so the
    # RK4 comparison uses matching meshes.
    grid_sizes = [int(round(x_max / h)) + 1 for h in numerov_h]

    # -----------------------------------------------------
    # Run the RK4 calculations
    # -----------------------------------------------------

    print("Running harmonic oscillator RK4 comparison...")

    # First generate one representative RK4 spectrum table and figure.
    rk4_reference_rows = RK4_solve_harmonic_oscillator_energies(
        x_max=x_max,
        n_grid=1200,
        n_states=n_states,
        omega=omega,
    )
    save_csv_rows(
        rk4_results_dir / "2b_harmonic_oscillator_RK4_energies.csv",
        [
            {
                "state_index": row.state_index,
                "parity": row.parity,
                "rk4_energy": row.energy,
                "exact_energy": row.exact_energy,
                "absolute_error": row.absolute_error,
                "relative_error": row.relative_error,
            }
            for row in rk4_reference_rows
        ],
    )

    plot_energy_comparison(
        np.array([row.energy for row in rk4_reference_rows], dtype=float),
        np.array([row.exact_energy for row in rk4_reference_rows], dtype=float),
        rk4_results_dir / "2b_harmonic_oscillator_RK4_state_energy_comparison.png",
        "Harmonic oscillator (RK4) - state energy comparison",
        exact_label=r"exact: $E_n=\omega\left(n+\frac{1}{2}\right)$",
        numerical_label="RK4",
    )

    # Then compute the RK4 convergence curve on the same family of spacings
    # used by the Numerov study.
    rk4_convergence = RK4_harmonic_convergence_vs_grid(
        x_max=x_max,
        grid_sizes=grid_sizes,
        n_states=n_states,
        omega=omega,
    )

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing harmonic oscillator RK4 comparison results...")

    rk4_slopes = estimate_convergence_slopes(
        rk4_convergence["h"], rk4_convergence["energy_errors"]
    )
    save_csv_rows(
        rk4_results_dir
        / "2b_harmonic_oscillator_RK4_energy_convergence_vs_h_slopes.csv",
        rk4_slopes,
    )

    plot_error_curve(
        rk4_convergence["h"],
        rk4_convergence["energy_errors"],
        "Grid spacing $h$",
        rk4_results_dir / "2b_harmonic_oscillator_RK4_energy_convergence_vs_h.png",
        "Harmonic oscillator (RK4) - energy convergence vs grid spacing $h$",
        slopes=rk4_slopes,
    )

    if not np.allclose(numerov_h, rk4_convergence["h"]):
        raise ValueError("Numerov and RK4 comparisons must use identical h values.")

    comparison_rows = []

    for method, h_values, errors in [
        ("Numerov", numerov_h, numerov_errors),
        ("RK4", rk4_convergence["h"], rk4_convergence["energy_errors"]),
    ]:
        # Flatten both error tables into one CSV so the report can compare the
        # methods row by row without special-case parsing.
        for grid_index, h_value in enumerate(h_values):
            row = {"method": method, "h": h_value}
            for state_index in range(errors.shape[1]):
                row[f"state_{state_index}_abs_error"] = errors[grid_index, state_index]
            row["max_abs_error"] = float(np.max(errors[grid_index, :]))
            comparison_rows.append(row)
    save_csv_rows(
        comparison_results_dir
        / "2c_harmonic_oscillator_Numerov_VS_RK4_state_error_comparison.csv",
        comparison_rows,
    )

    plot_numerov_vs_RK4_errors(
        numerov_h,
        numerov_errors,
        rk4_convergence["h"],
        rk4_convergence["energy_errors"],
        comparison_results_dir
        / "2c_harmonic_oscillator_Numerov_VS_RK4_state_error_comparison.png",
        "Harmonic oscillator - Numerov vs RK4 state error comparison",
    )

    plot_harmonic_oscillator_RK4_root_diagnostics(
        rk4_results_dir,
        omega=omega,
        x_max=x_max,
    )

    if target_h is not None:
        # -----------------------------------------------------
        # Run the fixed-spacing box-size study
        # -----------------------------------------------------

        print("Running harmonic oscillator RK4 box-size study...")

        rk4_box = RK4_harmonic_convergence_vs_box_size_fixed_spacing(
            x_max_values=[4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            target_h=target_h,
            n_states=n_states,
            omega=omega,
            e_min=0.1,
            e_max=6.0,
        )

        # -----------------------------------------------------
        # Analyze and save the box-size results
        # -----------------------------------------------------

        print("Analyzing harmonic oscillator RK4 box-size results...")

        save_csv_rows(
            rk4_results_dir
            / "2b_harmonic_oscillator_RK4_energy_convergence_vs_x_max.csv",
            [
                {
                    "x_max": x_val,
                    "n_grid": int(n_val),
                    "h": h_val,
                    **{
                        f"state_{state_index}_abs_error": rk4_box["energy_errors"][
                            row_index, state_index
                        ]
                        for state_index in range(rk4_box["energy_errors"].shape[1])
                    },
                }
                for row_index, (x_val, n_val, h_val) in enumerate(
                    zip(rk4_box["x_max"], rk4_box["n_grid"], rk4_box["h"])
                )
            ],
        )

        plot_error_curve(
            rk4_box["x_max"],
            rk4_box["energy_errors"],
            "Box size $x_{\\max}$",
            rk4_results_dir
            / "2b_harmonic_oscillator_RK4_energy_convergence_vs_x_max.png",
            "Harmonic oscillator (RK4) - energy convergence vs box size $x_{\\max}$",
        )


# ===========================================================================
# FUNCTION: run_harmonic_oscillator
# ===========================================================================
def run_harmonic_oscillator(results_dir: Path) -> None:
    """
    Run the harmonic-oscillator benchmark case.

    This experiment validates the solver against the exact ladder spectrum and
    produces both grid-refinement and box-size convergence studies.

    Parameters
    ----------
    results_dir : Path
        Root results directory under which the Numerov, RK4, and comparison
        subdirectories are created.
    """

    # -----------------------------------------------------
    # Set up the experiment
    # -----------------------------------------------------

    # Create output folders for the Numerov and RK4 calculations, plus a third
    # folder for direct comparison data and plots.
    numerov_results_dir = _experiment_results_dir(
        results_dir, "2a_harmonic_oscillator_Numerov"
    )
    rk4_results_dir = _experiment_results_dir(results_dir, "2b_harmonic_oscillator_RK4")
    comparison_results_dir = _experiment_results_dir(
        results_dir, "2c_harmonic_oscillator_Numerov_VS_RK4_comparison"
    )

    # Define the physical and numerical parameters for the experiment
    omega = 1.0  # Harmonic-oscillator frequency in V(x) = 0.5 * omega^2 * x^2
    x_max = 8.0  # The solver's spatial domain will be [-x_max, x_max]
    n_grid = 2500  # Number of grid points for the Numerov solver, including boundaries

    # -----------------------------------------------------
    # Run the experiment
    # -----------------------------------------------------

    print("Running harmonic oscillator experiment (Numerov)...")

    # Run the solver to find the first four bound states. Use inward shooting for
    # the harmonic oscillator. On a truncated infinite domain, outward shooting
    # can pick up the exponentially growing forbidden-region solution instead of
    # the physical decaying tail
    states = solve_symmetric_potential_inward_shooting(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=6.5,
    )

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing harmonic oscillator results (Numerov)...")

    # Plot the root-finding diagnostics for all four states, with separate panels
    # for the even and odd sectors
    plot_harmonic_oscillator_root_diagnostics(
        numerov_results_dir,
        omega=omega,
        x_max=x_max,
    )

    # Tabulate the numerical energies and compare them to the exact formula
    # E_n = omega * (n + 0.5)
    numerical = np.array([s.energy for s in states[:4]])
    exact = exact_harmonic_oscillator_energies(np.arange(4), omega=omega)

    # Save a CSV comparing the numerical and exact energies, along with the relative
    # error
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
    save_csv_rows(
        numerov_results_dir / "2a_harmonic_oscillator_Numerov_energies.csv",
        rows,
    )

    # Plot the states, probability densities, and energy comparison for the first four
    # states
    x = states[0].x_full
    V = harmonic_oscillator(x, omega=omega)

    plot_potential_and_states(
        x,
        V,
        states,
        numerov_results_dir / "2a_harmonic_oscillator_Numerov_states.png",
        "Harmonic oscillator - states",
        potential_label=r"$V(x)=\frac{1}{2}\omega^2x^2$",
    )

    plot_probability_densities(
        states,
        numerov_results_dir / "2a_harmonic_oscillator_Numerov_state_densities.png",
        "Harmonic oscillator - state densities",
    )

    plot_energy_comparison(
        numerical,
        exact,
        numerov_results_dir
        / "2a_harmonic_oscillator_Numerov_state_energy_comparison.png",
        "Harmonic oscillator - state energy comparison",
        exact_label=r"Exact: $E_n=\omega\left(n+\frac{1}{2}\right)$",
        numerical_label="Numerical",
    )

    # Grid-refinement convergence for all four displayed states. Use the
    # inward-decay solver here as well, so the convergence study matches the
    # stable harmonic-oscillator calculation used for the final figures.
    # Grid convergence changes h at fixed domain size, isolating the
    # discretization error from the finite-domain truncation error.
    conv_h = convergence_vs_grid(
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max=x_max,
        grid_sizes=[500, 800, 1200, 1800, 2500],
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=6.0,
        reference_energies=exact[:4],
        solver_fn=solve_symmetric_potential_inward_shooting,
    )
    conv_h_slopes = estimate_convergence_slopes(conv_h["h"], conv_h["energy_errors"])
    save_csv_rows(
        numerov_results_dir
        / "2a_harmonic_oscillator_Numerov_energy_convergence_vs_h_slopes.csv",
        conv_h_slopes,
    )

    plot_error_curve(
        conv_h["h"],
        conv_h["energy_errors"],
        "Grid spacing $h$",
        numerov_results_dir
        / "2a_harmonic_oscillator_Numerov_energy_convergence_vs_h.png",
        "Harmonic oscillator (Numerov) - energy convergence vs grid spacing $h$",
        slopes=conv_h_slopes,
    )

    # Keep the same nominal spacing for the RK4 and Numerov box-size studies.
    target_h = x_max / (n_grid - 1)

    # Box-size convergence should isolate the finite-domain truncation error.
    # Therefore h is kept approximately fixed while x_max changes. Holding
    # n_grid fixed would also change h and mix two different error sources.
    # Convergence claims are only meaningful if discretization and
    # domain-truncation errors are not folded together.
    conv_box = convergence_vs_box_size_fixed_spacing(
        potential_fn=harmonic_oscillator,
        potential_kwargs={"omega": omega},
        x_max_values=[4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        target_h=target_h,
        n_even=2,
        n_odd=2,
        e_min=0.1,
        e_max=6.0,
        reference_energies=exact[:4],
        solver_fn=solve_symmetric_potential_inward_shooting,
    )
    save_csv_rows(
        numerov_results_dir
        / "2a_harmonic_oscillator_Numerov_energy_convergence_vs_x_max.csv",
        [
            {
                "x_max": x_val,
                "n_grid": int(n_val),
                "h": h_val,
                **{
                    f"state_{state_index}_abs_error": conv_box["energy_errors"][
                        row_index, state_index
                    ]
                    for state_index in range(conv_box["energy_errors"].shape[1])
                },
            }
            for row_index, (x_val, n_val, h_val) in enumerate(
                zip(conv_box["x_max"], conv_box["n_grid"], conv_box["h"])
            )
        ],
    )

    plot_error_curve(
        conv_box["x_max"],
        conv_box["energy_errors"],
        "Box size $x_{\\max}$",
        numerov_results_dir
        / "2a_harmonic_oscillator_Numerov_energy_convergence_vs_x_max.png",
        "Harmonic oscillator (Numerov) - energy convergence vs box size $x_{\\max}$",
    )

    # -----------------------------------------------------
    # Run the RK4 comparison and box-size convergence studies
    # -----------------------------------------------------

    print(
        "Running harmonic oscillator RK4 comparison and box-size convergence studies..."
    )

    # The RK4 comparison reuses the same grid spacings as the Numerov
    # grid-convergence study, so both methods are compared on matched meshes at
    # the same fixed domain size.
    run_harmonic_oscillator_RK4_comparison(
        rk4_results_dir=rk4_results_dir,
        comparison_results_dir=comparison_results_dir,
        numerov_convergence=conv_h,
        omega=omega,
        x_max=x_max,
        target_h=target_h,
    )


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Quartic double well
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ===========================================================================
# ===========================================================================
# FUNCTION: run_quartic_double_well
# ===========================================================================
# ===========================================================================
def run_quartic_double_well(results_dir: Path) -> None:
    """
    Run the quartic double-well study.

    In addition to plotting the low-lying states, this experiment sweeps the
    double-well parameter b and records the tunneling splitting E1 - E0.

    Parameters
    ----------
    results_dir : Path
        Root results directory under which this experiment creates its output
        subdirectory.
    """

    # -----------------------------------------------------
    # Set up the experiment
    # -----------------------------------------------------

    experiment_dir = _experiment_results_dir(results_dir, "3_double_well_Numerov")

    base_kwargs = {"a": 1.0, "b": 6.0, "shift_min_to_zero": True}
    # The quartic wells centered near x = sqrt(b/2) develop long forbidden-region
    # tails, so x_max = 3.0 leaves visible box error in the published energies.
    # Use a larger default box and keep h near 1e-3.
    x_max = 4.0
    n_grid = 4000

    # -----------------------------------------------------
    # Run the experiment
    # -----------------------------------------------------

    print("Running quartic double well experiment...")

    # Run the solver to find the first four bound states of the quartic double
    # well. This experiment uses the project's standard outward half-domain
    # solver, treating the finite box at x_max as part of the numerical
    # approximation being studied.
    states = solve_symmetric_potential_outward_shooting(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=quartic_double_well,
        potential_kwargs=base_kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing quartic double well results...")

    # Plot the root-finding diagnostics for all four states, with separate panels
    # for the even and odd sectors
    plot_double_well_root_diagnostics(
        experiment_dir,
        potential_kwargs=base_kwargs,
        x_max=x_max,
    )

    # Tabulate the numerical energies and mismatches for the first four states. No
    # exact reference is available for this non-analytic potential
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
    save_csv_rows(experiment_dir / "3_double_well_Numerov_energies.csv", rows)

    # Plot the states and probability densities for the first four states, along with
    # the potential curve. The states are more localized than in the harmonic case,
    # so the same x_max and n_grid give good visual resolution of the states while still
    # showing the box boundaries.
    x = states[0].x_full
    V = quartic_double_well(x, **base_kwargs)

    plot_potential_and_states(
        x,
        V,
        states,
        experiment_dir / "3_double_well_Numerov_states.png",
        "Quartic double well - states",
        potential_label=(
            r"$V(x)=a x^4 - b x^2 + \frac{b^2}{4a}$"
            if base_kwargs["shift_min_to_zero"]
            else r"$V(x)=a x^4 - b x^2$"
        ),
    )

    plot_probability_densities(
        states,
        experiment_dir / "3_double_well_Numerov_state_densities.png",
        "Quartic double well - state densities",
    )

    # Use successive refinements instead of one finite reference. For this
    # non-analytic spectrum, the quantity plotted versus h is
    # |E(h_i) - E(h_{i+1})| rather than |E(h_i) - E_ref|. That avoids fitting
    # against a single finite-reference floor once the grids become very fine
    # and makes the convergence study a self-consistency check instead of a
    # comparison to one arbitrarily chosen "best" grid.
    conv_h = convergence_vs_grid_successive(
        potential_fn=quartic_double_well,
        potential_kwargs=base_kwargs,
        x_max=x_max,
        grid_sizes=[600, 1000, 1600, 2200, 3000, 4000],
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    conv_h_slopes = estimate_convergence_slopes(conv_h["h"], conv_h["energy_errors"])
    save_csv_rows(
        experiment_dir / "3_double_well_Numerov_energy_convergence_vs_h_slopes.csv",
        conv_h_slopes,
    )

    # With no closed-form spectrum available, the plotted y-values are not
    # |E_numerical - E_exact|. They are the successive-refinement surrogate
    # |E(h_i) - E(h_{i+1})| returned by convergence_vs_grid_successive().
    plot_error_curve(
        conv_h["h"],
        conv_h["energy_errors"],
        "Grid spacing $h$",
        experiment_dir / "3_double_well_Numerov_energy_convergence_vs_h.png",
        "Quartic double well - energy convergence vs grid spacing $h$",
        slopes=conv_h_slopes,
        ylabel="Successive energy difference $|E(h_i) - E(h_{i+1})|$",
    )

    # For box-size convergence, compare against a larger box while keeping h
    # approximately fixed so the curve mostly reflects boundary truncation.
    # The final x_max = 5.0 row is the larger-box reference itself, so its
    # plotted error is exactly zero and is omitted automatically on the log
    # axis by plot_error_curve().
    reference_states_box = solve_symmetric_potential_outward_shooting(
        x_max=5.0,
        n_grid=5000,
        potential_fn=quartic_double_well,
        potential_kwargs=base_kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
    )
    reference_energies_box = np.array(
        [s.energy for s in reference_states_box[:4]], dtype=float
    )

    target_h = x_max / (n_grid - 1)

    # Re-solve the same four low-lying states on a sequence of larger boxes,
    # adjusting n_grid each time so the spacing stays near target_h. The
    # returned errors are absolute differences from the larger-box reference
    # spectrum, so this study isolates boundary truncation more cleanly than a
    # fixed-n_grid sweep would.
    conv_box = convergence_vs_box_size_fixed_spacing(
        potential_fn=quartic_double_well,
        potential_kwargs=base_kwargs,
        x_max_values=[2.5, 3.0, 3.5, 4.0, 4.5],
        target_h=target_h,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=20.0,
        reference_energies=reference_energies_box,
    )
    save_csv_rows(
        experiment_dir / "3_double_well_Numerov_energy_convergence_vs_x_max.csv",
        [
            {
                "x_max": x_val,
                "n_grid": int(n_val),
                "h": h_val,
                **{
                    f"state_{state_index}_abs_error": conv_box["energy_errors"][
                        row_index, state_index
                    ]
                    for state_index in range(conv_box["energy_errors"].shape[1])
                },
            }
            for row_index, (x_val, n_val, h_val) in enumerate(
                zip(conv_box["x_max"], conv_box["n_grid"], conv_box["h"])
            )
        ],
    )

    plot_error_curve(
        conv_box["x_max"],
        conv_box["energy_errors"],
        "Box size $x_{\\max}$",
        experiment_dir / "3_double_well_Numerov_energy_convergence_vs_x_max.png",
        "Quartic double well - energy convergence vs box size $x_{\\max}$",
        ylabel="Energy error relative to larger-box reference",
    )

    # Sweep the double-well parameter b while keeping a fixed, and solve the
    # lowest even/odd pair at each value. This isolates how strengthening the
    # barrier changes E0, E1, and the tunneling splitting E1 - E0 across the
    # family of quartic double wells.
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
    save_csv_rows(
        experiment_dir / "3_double_well_Numerov_energy_splitting_vs_b.csv",
        sweep_rows,
    )

    b_vals = np.array([r["b"] for r in sweep_rows], dtype=float)
    e0 = np.array([r["E0"] for r in sweep_rows], dtype=float)
    e1 = np.array([r["E1"] for r in sweep_rows], dtype=float)
    splitting = np.array([r["splitting"] for r in sweep_rows], dtype=float)
    plot_splitting_curve(
        b_vals,
        e0,
        e1,
        splitting,
        "Parameter $b$",
        experiment_dir / "3_double_well_Numerov_energy_splitting_vs_b.png",
        "Quartic double well - energy splitting vs parameter $b$",
        ylabel="Energy / splitting",
    )


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Finite square well
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ===========================================================================
# FUNCTION: run_finite_square_well
# ===========================================================================
def run_finite_square_well(results_dir: Path) -> None:
    """
    Run the finite square well as an additional nontrivial potential.

    This case demonstrates that the same solver handles a finite number of bound
    states when the confining walls have finite height.

    Parameters
    ----------
    results_dir : Path
        Root results directory under which this experiment creates its output
        subdirectory.
    """

    # -----------------------------------------------------
    # Set up the experiment
    # -----------------------------------------------------

    experiment_dir = _experiment_results_dir(
        results_dir, "4_finite_square_well_Numerov"
    )

    # Define the physical and numerical parameters for the experiment
    x_max = 4.0
    n_grid = 3000
    kwargs = {"V0": 12.0, "a": 1.0}

    # -----------------------------------------------------
    # Run the experiment
    # -----------------------------------------------------

    print("Running finite square well experiment (Numerov)...")

    # Run the solver to find the first four bound states of the finite square
    # well. As in the infinite-well and quartic boxed studies, this uses the
    # outward half-domain solver and checks the boundary leakage at x_max.
    states = solve_symmetric_potential_outward_shooting(
        x_max=x_max,
        n_grid=n_grid,
        potential_fn=finite_square_well,
        potential_kwargs=kwargs,
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=11.9,
    )

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing finite square well results...")

    # Save the numerical bound-state energies for the first four states.
    rows = []
    for i, s in enumerate(states[:4]):
        rows.append(
            {
                "state_index": i,
                "parity": s.parity,
                "energy": s.energy,
            }
        )
    save_csv_rows(experiment_dir / "4_finite_square_well_Numerov_energies.csv", rows)

    # Plot the states and probability densities for the first four states, along with
    # the potential curve
    x = states[0].x_full
    V = finite_square_well(x, **kwargs)

    plot_potential_and_states(
        x,
        V,
        states,
        experiment_dir / "4_finite_square_well_Numerov_states.png",
        "Finite square well - states",
    )

    plot_probability_densities(
        states,
        experiment_dir / "4_finite_square_well_Numerov_state_densities.png",
        "Finite square well - state densities",
    )

    # Measure grid convergence for the first four finite-well bound states by
    # re-solving the problem on several coarser meshes and comparing each
    # spectrum to the current finest-grid result. With no analytic reference
    # available for this boxed finite well, the finest saved run acts as the
    # practical stand-in reference in |E_N - E_ref|.
    reference_energies = np.array([s.energy for s in states[:4]], dtype=float)
    conv_h = convergence_vs_grid(
        potential_fn=finite_square_well,
        potential_kwargs=kwargs,
        x_max=x_max,
        grid_sizes=[400, 700, 1000, 1500, 2200],
        n_even=2,
        n_odd=2,
        e_min=0.0,
        e_max=11.9,
        reference_energies=reference_energies,
    )
    conv_h_slopes = estimate_convergence_slopes(conv_h["h"], conv_h["energy_errors"])
    save_csv_rows(
        experiment_dir
        / "4_finite_square_well_Numerov_energy_convergence_vs_h_slopes.csv",
        conv_h_slopes,
    )

    plot_error_curve(
        conv_h["h"],
        conv_h["energy_errors"],
        "Grid spacing $h$",
        experiment_dir / "4_finite_square_well_Numerov_energy_convergence_vs_h.png",
        "Finite square well - energy convergence vs grid spacing $h$",
        slopes=conv_h_slopes,
        ylabel="Energy error relative to finest-grid reference",
    )

    plot_finite_square_well_root_diagnostics(
        experiment_dir,
        x_max=x_max,
        V0=float(kwargs["V0"]),
        a=float(kwargs["a"]),
    )


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Scattering experiments
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ===========================================================================
# FUNCTION: run_scattering
# ===========================================================================
def run_scattering(results_dir: Path) -> None:
    """
    Run the scattering extension.

    This experiment computes transmission and reflection probabilities for a
    single finite barrier and a double-barrier resonant tunneling structure. The
    double-barrier case is the important extension: resonant transmission peaks
    appear when the energy matches a quasi-bound state in the central well.

    Parameters
    ----------
    results_dir : Path
        Root results directory under which the single- and double-barrier
        scattering output subdirectories are created.
    """

    # -----------------------------------------------------
    # Set up the experiment
    # -----------------------------------------------------

    # Create output folders for the single- and double-barrier scattering experiments
    single_results_dir = _experiment_results_dir(
        results_dir, "5_scattering_single_barrier_Numerov"
    )
    double_results_dir = _experiment_results_dir(
        results_dir, "6_scattering_double_barrier_Numerov"
    )

    # Define the physical and numerical parameters for the experiment
    x_min = -8.0
    x_max = 8.0
    n_grid = 4000
    x = np.linspace(x_min, x_max, n_grid)

    # -----------------------------------------------------
    # Run the experiment - single barrier (basic tunneling/over-barrier validation)
    # -----------------------------------------------------

    print("Running single barrier scattering experiment (Numerov)...")

    # First validate scattering on a single finite barrier, then move to the
    # double-barrier system where resonant tunneling appears.
    single_kwargs = {"V0": 5.0, "width": 1.2, "center": 0.0}
    V_single = square_barrier(x, **single_kwargs)
    energies_single = np.linspace(0.2, 10.0, 240)

    # Sweep over the list of incident energies, solving the same single-barrier
    # scattering problem once per energy and collecting the resulting
    # transmission/reflection data into one results list.
    single_results = sweep_scattering(x, V_single, energies_single)

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing single-barrier scattering results...")

    # Tabulate the transmission and reflection coefficients for the single barrier,
    # along with the sum T + R to check unitarity. The CSV is saved before plotting
    # so the numerical data is available even if the plotting code has issues
    single_rows = [
        {
            "energy": result.energy,
            "transmission": result.transmission,
            "reflection": result.reflection,
            "T_plus_R": result.transmission + result.reflection,
        }
        for result in single_results
    ]
    save_csv_rows(
        single_results_dir
        / "5_scattering_single_barrier_Numerov_transmission_reflection.csv",
        single_rows,
    )

    # Extract the transmission and reflection coefficients into separate arrays for
    # plotting. The CSV saved above contains the same data in a more accessible
    # format for analysis and comparison
    T_single = np.array([result.transmission for result in single_results], dtype=float)
    R_single = np.array([result.reflection for result in single_results], dtype=float)

    plot_scattering_coefficients(
        energies_single,
        T_single,
        R_single,
        single_results_dir
        / "5_scattering_single_barrier_Numerov_transmission_reflection.png",
        "Single barrier scattering - transmission and reflection",
    )

    # -----------------------------------------------------
    # Run the experiment - double barrier (resonant tunneling)
    # -----------------------------------------------------

    print("Running double-barrier scattering experiment...")

    double_kwargs = {"V0": 5.0, "barrier_width": 0.6, "well_width": 1.4, "center": 0.0}
    V_double = double_square_barrier(x, **double_kwargs)
    energies_double = np.linspace(0.2, 5.0, 420)

    # The double-barrier sweep is finer and focused on the energy range where
    # resonant peaks are expected.
    double_results = sweep_scattering(x, V_double, energies_double)

    # Extract arrays for plotting and peak finding after the sweep is complete.
    T_double = np.array([result.transmission for result in double_results], dtype=float)
    R_double = np.array([result.reflection for result in double_results], dtype=float)
    peaks = find_transmission_peaks(energies_double, T_double, threshold=0.65)

    # -----------------------------------------------------
    # Analyze and save the results
    # -----------------------------------------------------

    print("Analyzing double-barrier scattering results...")

    # Save transmission, reflection, and the conservation check T + R for the
    # double-barrier sweep. Resonance peaks are saved separately below.
    double_rows = [
        {
            "energy": result.energy,
            "transmission": result.transmission,
            "reflection": result.reflection,
            "T_plus_R": result.transmission + result.reflection,
        }
        for result in double_results
    ]
    save_csv_rows(
        double_results_dir
        / "6_scattering_double_barrier_Numerov_transmission_reflection.csv",
        double_rows,
    )

    plot_scattering_coefficients(
        energies_double,
        T_double,
        R_double,
        double_results_dir
        / "6_scattering_double_barrier_Numerov_transmission_reflection.png",
        "Double-barrier scattering - transmission and reflection",
    )

    if peaks:
        resonance_energy = peaks[0]["energy"]
        resonance_rows = peaks
    else:
        resonance_energy = float(energies_double[np.argmax(T_double)])
        resonance_rows = [
            {
                "energy": resonance_energy,
                "transmission": float(np.max(T_double)),
            }
        ]
    save_csv_rows(
        double_results_dir
        / "6_scattering_double_barrier_Numerov_transmission_peaks.csv",
        resonance_rows,
    )

    psi_res, resonance_result = scattering_wavefunction(x, V_double, resonance_energy)
    plot_scattering_potential_and_probability(
        x,
        V_double,
        psi_res,
        resonance_result.energy,
        double_results_dir
        / "6_scattering_double_barrier_Numerov_state_probability_density.png",
        "Double-barrier scattering - resonant state probability density",
    )
