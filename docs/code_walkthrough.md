# Code walkthrough for the Numerov bound-state project
This document is intended for a project reviewer. It explains the role of each file and the purpose of each top-level code block. The source files themselves also contain docstrings, section banners, and inline comments.
## Big picture
The code separates the project into layers: low-level numerical methods, physical potentials, analysis helpers, plotting helpers, high-level experiments, and tests. The central equation is the dimensionless one-dimensional time-independent Schrodinger equation
```text
psi''(x) = 2 [V(x) - E] psi(x)
```
The bound-state part uses Numerov integration plus shooting/root finding. The harmonic oscillator also uses inward shooting for stability. The scattering extension uses complex Numerov integration to compute transmission and reflection.

## Project workflow
### `scripts/run_solver.py`
Entry point. Creates/clears the results directory, runs every experiment in order, and then runs the test suite.
| Lines | Block | Purpose |
|---:|---|---|
| 39-89 | `function main` | Run the full project pipeline and regenerate the results directory. |

### `tests/test_solver.py`
Regression and sanity tests. These tests protect the validated numerical behavior from silent regressions.
| Lines | Block | Purpose |
|---:|---|---|
| 48-56 | `function test_normalization` | Check that the wavefunction normalization helper produces unit norm. |
| 63-70 | `function test_derivative_at_right_edge_polynomial` | Check the high-order right-edge derivative stencil on a smooth polynomial. |
| 77-102 | `function test_find_rk4_brackets_accepts_exact_zero_hit` | Check that RK4 bracketing keeps roots that land exactly on a scan point. |
| 109-126 | `function test_square_well_ground_state` | Verify that the square-well ground-state energy matches the exact result. |
| 133-153 | `function test_square_well_convergence_order` | Check that the square-well energies recover near-fourth-order grid convergence. |
| 160-177 | `function test_harmonic_oscillator_first_levels` | Verify that the first few harmonic-oscillator energies are accurate. |
| 184-201 | `function test_harmonic_oscillator_inward_decay_first_levels` | Verify that the inward-decay harmonic solver resolves the first levels accurately. |
| 208-222 | `function test_double_well_splitting_positive` | Check that the first odd state lies above the first even state in the double well. |
| 229-239 | `function test_quartic_double_well_exact_shifted_minimum` | Check that the shifted double well uses the analytic minimum rather than a grid-dependent sampled minimum. |
| 246-290 | `function test_double_well_larger_box_improves_low_lying_energies` | Check that enlarging the quartic-double-well box reduces truncation error. |
| 297-315 | `function test_double_well_same_box_grid_errors_decrease` | Check that successive double-well grid refinements reduce the coarse-grid error estimates. |
| 322-339 | `function test_double_well_successive_convergence_order` | Check that the lowest double-well states recover near-fourth-order successive-refinement convergence. |
| 347-364 | `function test_double_well_low_state_mismatches_are_polished` | Check that the low double-well roots are polished to small final boundary mismatches. |
| 370-387 | `function test_double_well_even_odd_mismatch_scans_differ` | Check that even and odd double-well mismatch scans remain parity-specific. |
| 390-402 | `function test_scattering_probability_conservation` | Check that scattering approximately conserves probability current. |
| 406-418 | `function test_run_solver_import_without_experiment_dependencies` | Check that the entry script can be imported without importing plotting code. |
| 422-442 | `function run_all_tests` | Execute the lightweight regression suite and print a short summary. |


## Core numerical methods
### `src/numerov.py`
Low-level Numerov recurrence, wavefunction normalization, and boundary derivative estimation.
| Lines | Block | Purpose |
|---:|---|---|
| 21-39 | `function q_from_energy` | Build the Numerov coefficient q(x) for a trial energy. |
| 46-111 | `function numerov_outward` | Integrate a second-order ODE outward with the Numerov recurrence. |
| 118-146 | `function normalize_wavefunction` | Normalize a wavefunction safely on a discrete grid. |
| 153-188 | `function derivative_at_right_edge` | Estimate the derivative at the last grid point with a backward stencil. |

### `src/shooting.py`
Bound-state shooting solvers, including parity-based outward shooting and stable inward shooting for confining potentials.
| Lines | Block | Purpose |
|---:|---|---|
| 31-53 | `class StateSolution` | Container for a single bound-state solution. |
| 60-92 | `function initial_conditions` | Construct parity-consistent starting values near x = 0. |
| 99-126 | `function half_domain_wavefunction` | Compute the half-domain trial wavefunction for a given energy. |
| 133-172 | `function boundary_mismatch` | Evaluate the mismatch used for eigenvalue shooting. |
| 179-226 | `function find_brackets` | Scan an energy interval and locate sign-changing brackets. |
| 233-252 | `function sample_boundary_mismatch` | Sample the shooting mismatch over an energy interval. |
| 259-304 | `function bisection_history` | Record the bisection process for one eigenvalue bracket. |
| 311-368 | `function bisect_energy` | Refine a sign-changing eigenvalue bracket with bisection. |
| 375-408 | `function build_full_wavefunction` | Reconstruct the full wavefunction from its half-domain representation. |
| 415-452 | `function solve_state_from_bracket` | Compute one bound state starting from a valid energy bracket. |
| 459-487 | `function inward_decay_initial_conditions` | Construct stable starting values at x_max for inward shooting. |
| 494-515 | `function inward_decay_half_domain_wavefunction` | Integrate the decaying tail inward from x_max to x = 0. |
| 522-549 | `function inward_decay_boundary_mismatch` | Evaluate the parity mismatch at the origin for inward shooting. |
| 556-584 | `function sample_inward_decay_mismatch` | Sample the inward-shooting parity mismatch over an energy interval. |
| 591-626 | `function find_inward_decay_brackets` | Locate sign-changing brackets for inward-shooting eigenvalue searches. |
| 633-692 | `function bisect_energy_inward_decay` | Refine an inward-shooting eigenvalue bracket with bisection. |
| 699-747 | `function bisection_history_inward_decay` | Record the inward-shooting bisection process for diagnostic plots. |
| 754-795 | `function solve_state_from_inward_decay_bracket` | Compute one bound state using inward shooting from the decaying tail. |
| 802-870 | `function solve_symmetric_potential_inward_decay` | Solve symmetric confining potentials by shooting inward from x_max. |
| 877-952 | `function solve_symmetric_potential` | Solve for multiple bound states of a symmetric potential. |

### `src/rk4_compare.py`
Fourth-order Runge-Kutta implementation used only for the harmonic-oscillator method comparison.
| Lines | Block | Purpose |
|---:|---|---|
| 24-32 | `class RK4EnergyResult` | One RK4 eigenvalue result for the harmonic oscillator. |
| 39-50 | `function _harmonic_rhs` | Right-hand side for the first-order Schrodinger system. |
| 57-64 | `function rk4_step` | Advance the first-order Schrodinger system by one RK4 step. |
| 71-104 | `function rk4_inward_mismatch` | Compute the parity mismatch using inward RK4 shooting. |
| 111-139 | `function find_rk4_brackets` | Locate sign-changing energy brackets for RK4 inward shooting. |
| 146-161 | `function sample_rk4_mismatch` | Sample the RK4 inward-shooting mismatch over an energy interval. |
| 168-194 | `function bisect_rk4_energy` | Refine one RK4 shooting bracket with bisection. |
| 201-237 | `function bisection_history_rk4` | Record the RK4 inward-shooting bisection process for one bracket. |
| 244-295 | `function solve_harmonic_oscillator_rk4_energies` | Compute the lowest harmonic-oscillator energies using RK4 shooting. |
| 302-328 | `function rk4_harmonic_convergence_vs_grid` | Return RK4 harmonic-oscillator energies and errors for several grids. |
| 335-379 | `function rk4_harmonic_convergence_vs_box_size_fixed_spacing` | Return RK4 harmonic-oscillator errors versus box size at fixed spacing. |

### `src/scattering.py`
Complex Numerov scattering solver for transmission, reflection, and double-barrier resonances.
| Lines | Block | Purpose |
|---:|---|---|
| 28-57 | `class ScatteringResult` | Container for one scattering calculation. |
| 64-101 | `function numerov_outward_complex` | Integrate y'' = q(x)y using the Numerov recurrence for complex y. |
| 108-130 | `function integrate_from_right` | Impose a unit transmitted wave on the right and integrate backward. |
| 137-158 | `function decompose_left_asymptotic` | Decompose the left asymptotic wave into incoming and reflected parts. |
| 165-194 | `function solve_scattering` | Compute reflection and transmission for one incident energy. |
| 201-212 | `function scattering_wavefunction` | Return the physical scattering wavefunction for unit incident amplitude. |
| 219-227 | `function sweep_scattering` | Compute scattering coefficients for a sequence of energies. |
| 234-253 | `function find_transmission_peaks` | Locate simple local maxima in the transmission curve. |


## Physics and analysis layer
### `src/potentials.py`
Definitions of every potential used in the report.
| Lines | Block | Purpose |
|---:|---|---|
| 18-34 | `function harmonic_oscillator` | Harmonic-oscillator potential V(x) = 1/2 * omega^2 * x^2. |
| 41-66 | `function infinite_square_well_numeric` | Numerical approximation to an infinite square well. |
| 73-95 | `function finite_square_well` | Finite square well with barrier height V0 outside |x| <= a. |
| 102-136 | `function quartic_double_well` | Quartic double-well potential V(x) = a x^4 - b x^2, with an option to shift the analytic minima to zero for cleaner convergence studies. |
| 143-169 | `function square_barrier` | Rectangular scattering barrier. |
| 176-213 | `function double_square_barrier` | Symmetric double-barrier scattering potential. |

### `src/analysis.py`
Exact benchmark spectra, convergence helpers, CSV export, and double-well sweeps.
| Lines | Block | Purpose |
|---:|---|---|
| 23-40 | `function exact_square_well_energies` | Exact energies for an infinite square well on [-a, a]. |
| 47-67 | `function exact_harmonic_oscillator_energies` | Exact harmonic-oscillator spectrum E_n = omega (n + 1/2). |
| 74-92 | `function relative_error` | Compute componentwise relative error. |
| 99-146 | `function estimate_convergence_slopes` | Estimate convergence exponents from a log-log error curve. |
| 153-177 | `function save_csv_rows` | Save a list of dictionaries as a CSV table. |
| 184-205 | `function energies_from_states` | Extract the first n state energies from a list of StateSolution objects. |
| 212-268 | `function convergence_vs_grid` | Study eigenvalue convergence as the grid is refined. |
| 282-345 | `function convergence_vs_grid_successive` | Study grid convergence by comparing each grid to the next finer grid when no exact reference spectrum is available. |
| 352-408 | `function convergence_vs_box_size` | Study eigenvalue convergence as the computational box size changes. |
| 415-493 | `function convergence_vs_box_size_fixed_spacing` | Study box-size convergence while keeping grid spacing approximately fixed. |
| 500-562 | `function splitting_vs_parameter` | Measure double-well ground-state splitting while varying one parameter. |

### `src/experiments.py`
High-level routines that connect solvers, potentials, CSV outputs, and plots.
| Lines | Block | Purpose |
|---:|---|---|
| 72-268 | `function run_harmonic_rk4_comparison` | Compare the specialized Numerov integrator with general RK4 shooting. |
| 273-337 | `function plot_infinite_well_root_diagnostics` | Plot shooting/root-finding diagnostics for all four infinite-well states. |
| 343-428 | `function plot_harmonic_oscillator_root_diagnostics` | Plot inward-shooting root diagnostics for the first four harmonic-oscillator states. |
| 434-500 | `function plot_double_well_root_diagnostics` | Plot parity-separated root diagnostics for the first four quartic-double-well states. |
| 507-607 | `function run_square_well` | Run the infinite square well benchmark case and its convergence studies. |
| 609-757 | `function run_harmonic_oscillator` | Run the harmonic-oscillator benchmark, including inward-shooting diagnostics and the RK4 comparison. |
| 763-929 | `function run_double_well` | Run the quartic double-well study, including convergence and tunneling-splitting sweeps. |
| 935-982 | `function run_finite_square_well` | Run the finite square well as an additional nontrivial bound-state example. |
| 988-1089 | `function run_scattering` | Run the scattering extension for single- and double-barrier tunneling, including resonant-peak extraction. |

### `src/plotting.py`
Report-ready Matplotlib figures.
| Lines | Block | Purpose |
|---:|---|---|
| 20-33 | `function _ensure_parent` | Create the parent directory of an output path if it does not exist. |
| 40-102 | `function plot_potential_and_states` | Plot the potential together with several shifted eigenstates. |
| 109-149 | `function plot_probability_densities` | Plot |psi(x)|^2 for several states. |
| 156-198 | `function plot_energy_comparison` | Compare numerical and exact energy levels on the same figure. |
| 205-253 | `function plot_error_curve` | Plot absolute energy errors on log-log axes. |
| 260-302 | `function plot_splitting_curve` | Plot the lowest two energies and their splitting versus a parameter. |
| 309-369 | `function plot_root_finding_diagnostic` | Plot the shooting mismatch and bisection midpoints for selected states. |
| 376-415 | `function plot_scattering_coefficients` | Plot transmission and reflection probabilities versus incident energy. |
| 422-454 | `function plot_scattering_potential_and_probability` | Plot a scattering probability density together with the barrier potential. |
| 461-489 | `function plot_numerov_vs_rk4_errors` | Compare Numerov and RK4 harmonic-oscillator energy errors. |

## Important numerical checks
- The Numerov derivative stencil is fourth order when enough points are available. This matters because even inward-shooting states use the condition `psi'(0)=0`.
- The parity-based startup in `initial_conditions` includes higher-order Taylor terms, so the first Numerov step does not spoil the observed fourth-order convergence.
- Harmonic-oscillator inward shooting starts from the decaying forbidden-region tail and integrates toward the origin. This avoids contamination by the growing exponential mode.
- The quartic double well can shift its analytic minima to zero. That keeps the physical potential fixed as the grid changes and prevents fake convergence effects from a grid-sampled minimum.
- RK4 comparison solves the same harmonic oscillator with the same grid sizes so the comparison focuses on the integration formula rather than a different physical setup.
- The bound-state bisection routine finishes with a few safeguarded secant-style polishing steps inside the final bracket, which noticeably reduces the reported boundary mismatch for steep roots.
- Scattering outputs include a conservation check through `T + R`, which should remain close to one.
- The tests include analytic benchmarks, convergence-order checks, parity-specific double-well checks, normalization checks, derivative-stencil checks, and scattering sanity checks.
