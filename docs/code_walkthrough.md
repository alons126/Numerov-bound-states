# Code walkthrough for the Numerov bound-state project
This document is intended for a project reviewer. It explains the role of each file and the purpose of each top-level code block. The source files themselves also contain docstrings, section banners, and inline comments.

## Big picture
The code separates the project into layers: low-level numerical methods, physical potentials, analysis helpers, plotting helpers, high-level experiments, and tests. The central equation is the dimensionless one-dimensional time-independent Schrodinger equation
```text
psi''(x) = 2 [V(x) - E] psi(x)
```
The bound-state part uses Numerov integration plus shooting/root finding. That structure is not just an implementation preference: the project is solving a boundary-value eigenproblem, so only special energies satisfy the physical boundary conditions. The harmonic oscillator also uses inward shooting for stability. The scattering extension uses complex Numerov integration to compute transmission and reflection. A dedicated diagnostics layer now sits between the solvers and the report figures, so root-finding plots are assembled consistently across all bound-state experiments.

## Project workflow
### `scripts/run_solver.py`
Entry point. Creates/clears the canonical project `results/` directory, runs every experiment in order, and then runs the test suite.
| Lines | Block | Purpose |
|---:|---|---|
| 47-120 | `function main` | Run the full project pipeline and regenerate the results directory. |

### `tests/test_solver.py`
Regression and sanity tests. These tests protect the validated numerical behavior from silent regressions.
| Lines | Block | Purpose |
|---:|---|---|
| 77-93 | `function test_normalization` | Check that the wavefunction normalization helper produces unit norm. |
| 94-107 | `function test_derivative_at_right_edge_polynomial` | Check the high-order right-edge derivative stencil on a smooth polynomial. |
| 108-141 | `function test_find_rk4_brackets_accepts_exact_zero_hit` | Check that RK4 bracketing keeps roots that land exactly on a scan point. |
| 142-166 | `function test_square_well_ground_state` | Verify that the square-well ground-state energy matches the exact result. |
| 167-194 | `function test_square_well_convergence_order` | Check that the square-well energies recover near-fourth-order grid convergence. |
| 220-244 | `function test_harmonic_oscillator_first_levels` | Verify that the first few harmonic-oscillator energies are accurate. |
| 245-269 | `function test_harmonic_oscillator_inward_shooting_first_levels` | Verify that the inward-shooting harmonic solver resolves the first levels accurately. |
| 270-291 | `function test_double_well_splitting_positive` | Check that the first odd state lies above the first even state in the double well. |
| 292-309 | `function test_quartic_double_well_exact_shifted_minimum` | Check that the shifted double well uses the analytic minimum rather than a grid-dependent sampled minimum. |
| 310-360 | `function test_double_well_larger_box_improves_low_lying_energies` | Check that enlarging the quartic-double-well box reduces truncation error. |
| 361-385 | `function test_double_well_same_box_grid_errors_decrease` | Check that successive double-well grid refinements reduce the coarse-grid error estimates. |
| 386-411 | `function test_double_well_successive_convergence_order` | Check that the lowest double-well states recover near-fourth-order successive-refinement convergence. |
| 412-435 | `function test_double_well_low_state_mismatches_are_polished` | Check that the low double-well roots are polished to small final boundary mismatches. |
| 436-456 | `function test_double_well_even_odd_mismatch_scans_differ` | Check that even and odd double-well mismatch scans remain parity-specific. |
| 457-473 | `function test_scattering_probability_conservation` | Check that scattering approximately conserves probability current. |
| 474-492 | `function test_run_solver_import_without_experiment_dependencies` | Check that the entry script can be imported without importing plotting code. |
| 493-512 | `function run_all_tests` | Execute the lightweight regression suite and print a short summary. |

## Core numerical methods
### `src/numerov.py`
Low-level Numerov recurrence, wavefunction normalization, and boundary derivative estimation. This file carries the numerical details needed to keep the overall shooting method as accurate as the formal Numerov recurrence.
| Lines | Block | Purpose |
|---:|---|---|
| 40-64 | `function q_from_energy` | Build the Numerov coefficient q(x) for a trial energy. |
| 65-157 | `function numerov_march` | Integrate a second-order ODE by marching along the supplied grid order, so the same helper supports both ascending-grid outward shooting and descending-grid inward shooting. |
| 158-202 | `function normalize_wavefunction` | Normalize a wavefunction safely on a discrete grid. |
| 203-246 | `function derivative_at_right_edge` | Estimate the derivative at the last grid point with a backward stencil. |

### `src/shooting.py`
Bound-state shooting solvers, including parity-based outward shooting and stable inward shooting for confining potentials. The outward startup is the more general one because it comes from the exact parity conditions at x = 0 for any symmetric potential. The inward startup is more specialized: it assumes x_max already lies in a decaying forbidden tail, so in this project it is used mainly for the harmonic oscillator and the matching RK4 comparison. The refinement stage also differs slightly: outward shooting adds a short safeguarded secant cleanup after bisection, while inward shooting stays with plain bisection on the origin-parity mismatch. This is the code realization of the course-level shooting idea: guess an energy, integrate, evaluate the mismatch, then refine the guess with root finding.
| Lines | Block | Purpose |
|---:|---|---|
| 70-99 | `class StateSolution` | Container for a single bound-state solution. |
| 100-154 | `function initial_conditions_outward_shooting` | Construct parity-consistent starting values near x = 0. |
| 155-201 | `function half_domain_wavefunction_outward_shooting` | Compute the half-domain trial wavefunction for a given energy. |
| 202-265 | `function boundary_mismatch_outward_shooting` | Evaluate the mismatch used for eigenvalue shooting. |
| 290-357 | `function find_brackets_outward_shooting` | Scan an energy interval and locate sign-changing brackets. |
| 358-437 | `function sample_mismatch_outward_shooting` | Sample the outward-shooting mismatch over an energy interval. |
| 438-532 | `function bisection_history_outward_shooting` | Record the bisection process for one eigenvalue bracket. |
| 533-658 | `function bisect_energy_outward_shooting` | Refine a sign-changing eigenvalue bracket with bisection. |
| 659-701 | `function build_full_wavefunction` | Reconstruct the full wavefunction from its half-domain representation. |
| 702-764 | `function solve_state_from_bracket_outward_shooting` | Compute one bound state starting from a valid energy bracket. |
| 765-827 | `function initial_conditions_inward_shooting` | Construct tail-based starting values at x_max for inward shooting in decaying forbidden regions. |
| 828-888 | `function half_domain_wavefunction_inward_shooting` | Integrate the decaying tail inward from x_max to x = 0. |
| 889-964 | `function boundary_mismatch_inward_shooting` | Evaluate the parity mismatch at the origin for inward shooting. |
| 1033-1119 | `function sample_mismatch_inward_shooting` | Sample the inward-shooting parity mismatch over an energy interval. |
| 1120-1199 | `function find_brackets_inward_shooting` | Locate sign-changing brackets for inward-shooting eigenvalue searches. |
| 1200-1309 | `function bisect_energy_inward_shooting` | Refine an inward-shooting eigenvalue bracket with bisection. |
| 1310-1414 | `function bisection_history_inward_shooting` | Record the inward-shooting bisection process for diagnostic plots. |
| 1415-1500 | `function solve_state_from_bracket_inward_shooting` | Compute one bound state using inward shooting from the decaying tail. |
| 1501-1640 | `function solve_symmetric_potential_inward_shooting` | Solve symmetric confining infinite-domain problems by shooting inward from x_max when a decaying tail model is appropriate. |
| 1641-1748 | `function solve_symmetric_potential_outward_shooting` | Solve for multiple bound states of a symmetric potential. |

### `src/RK4_harmonic_oscillator.py`
Fourth-order Runge-Kutta implementation used only for the harmonic-oscillator method comparison.
| Lines | Block | Purpose |
|---:|---|---|
| 41-54 | `class RK4EnergyResult` | One RK4 eigenvalue result for the harmonic oscillator. |
| 55-90 | `function _harmonic_rhs` | Right-hand side for the first-order Schrodinger system. |
| 91-129 | `function RK4_step` | Advance the first-order Schrodinger system by one RK4 step. |
| 130-196 | `function RK4_inward_mismatch` | Compute the parity mismatch using inward RK4 shooting. |
| 247-323 | `function RK4_find_brackets` | Locate sign-changing energy brackets for RK4 inward shooting. |
| 324-383 | `function RK4_sample_mismatch` | Sample the RK4 inward-shooting mismatch over an energy interval. |
| 384-451 | `function RK4_bisect_energy` | Refine one RK4 shooting bracket with bisection. |
| 452-530 | `function RK4_bisection_history` | Record the RK4 inward-shooting bisection process for one bracket. |
| 531-623 | `function RK4_solve_harmonic_oscillator_energies` | Compute the lowest harmonic-oscillator energies using RK4 shooting. |
| 624-677 | `function RK4_harmonic_convergence_vs_grid` | Return RK4 harmonic-oscillator energies and errors for several grids. |
| 678-748 | `function RK4_harmonic_convergence_vs_box_size_fixed_spacing` | Return RK4 harmonic-oscillator errors versus box size at fixed spacing. |

## Diagnostics and presentation layer
### `src/diagnostics.py`
Bound-state root-finding diagnostic orchestration. This file keeps the per-potential mismatch scans, final-root markers, and zoomed bisection views out of `experiments.py`, so the experiment runners stay focused on numerical setup and output bookkeeping.
| Lines | Block | Purpose |
|---:|---|---|
| 51-71 | `function _diagnostic_label_slug` | Convert a state label into a filename-safe suffix for zoom plots. |
| 72-184 | `function _plot_outward_root_diagnostics` | Build global and zoomed raw-mismatch diagnostics for outward-shooting problems. |
| 185-307 | `function _plot_inward_root_diagnostics` | Build global and zoomed raw-mismatch diagnostics for inward Numerov problems. |
| 308-360 | `function plot_infinite_well_root_diagnostics` | Generate the infinite-square-well diagnostic plots. |
| 361-418 | `function plot_harmonic_oscillator_root_diagnostics` | Generate the Numerov harmonic-oscillator diagnostic plots. |
| 419-549 | `function plot_harmonic_oscillator_RK4_root_diagnostics` | Generate the RK4 harmonic-oscillator diagnostic plots. |
| 550-610 | `function plot_double_well_root_diagnostics` | Generate the quartic-double-well diagnostic plots. |
| 611-669 | `function plot_finite_square_well_root_diagnostics` | Generate the finite-square-well diagnostic plots, including near-threshold root zooms. |

### `src/scattering.py`
Complex Numerov scattering solver for transmission, reflection, and double-barrier resonances.
| Lines | Block | Purpose |
|---:|---|---|
| 81-147 | `function numerov_outward_complex` | Integrate y'' = q(x)y using the Numerov recurrence for complex y. |
| 148-209 | `function integrate_from_right` | Impose a unit transmitted wave on the right and integrate backward, choosing the right boundary because it contains only the transmitted outgoing component while the left boundary must later be decomposed into incident and reflected parts. |
| 210-260 | `function decompose_left_asymptotic` | Decompose the left asymptotic wave into incoming and reflected parts. |
| 261-315 | `function solve_scattering` | Compute reflection and transmission for one incident energy. |
| 316-353 | `function scattering_wavefunction` | Return the physical scattering wavefunction for unit incident amplitude. |
| 354-386 | `function sweep_scattering` | Compute scattering coefficients for a sequence of energies. |
| 387-430 | `function find_transmission_peaks` | Locate simple local maxima in the transmission curve. |

## Physics and analysis layer
### `src/potentials.py`
Definitions of every potential used in the report. These are chosen to separate analytic validation problems from the more physical exploration problems.
| Lines | Block | Purpose |
|---:|---|---|
| 38-61 | `function harmonic_oscillator` | Harmonic-oscillator potential V(x) = 1/2 * omega^2 * x^2. |
| 62-95 | `function infinite_square_well_numeric` | Numerical approximation to an infinite square well. |
| 96-126 | `function finite_square_well` | Finite square well with barrier height V0 outside |x| <= a. |
| 127-188 | `function quartic_double_well` | Quartic double-well potential V(x) = a x^4 - b x^2, with an option to shift the analytic minima to zero for cleaner convergence studies. |
| 189-224 | `function square_barrier` | Rectangular scattering barrier. |
| 225-264 | `function double_square_barrier` | Symmetric double-barrier scattering potential. |

### `src/analysis.py`
Exact benchmark spectra, convergence helpers, CSV export, and double-well sweeps. This is where the project checks the numerical claims instead of only producing eigenvalues.
| Lines | Block | Purpose |
|---:|---|---|
| 33-57 | `function exact_square_well_energies` | Exact energies for an infinite square well on [-a, a]. |
| 58-85 | `function exact_harmonic_oscillator_energies` | Exact harmonic-oscillator spectrum E_n = omega (n + 1/2). |
| 86-154 | `function estimate_convergence_slopes` | Estimate convergence exponents from a log-log error curve. |
| 155-185 | `function save_csv_rows` | Save a list of dictionaries as a CSV table. |
| 186-222 | `function energies_from_states` | Extract the first n state energies from a list of StateSolution objects. |
| 223-287 | `function convergence_vs_grid` | Study eigenvalue convergence as the grid is refined. |
| 288-371 | `function convergence_vs_grid_successive` | Study grid convergence by comparing each grid to the next finer grid when no exact reference spectrum is available. |
| 372-459 | `function convergence_vs_box_size_fixed_spacing` | Study box-size convergence while keeping grid spacing approximately fixed. |
| 460-529 | `function splitting_vs_parameter` | Measure double-well ground-state splitting while varying one parameter. |

### `src/experiments.py`
High-level routines that connect solvers, potentials, CSV outputs, and plots. Each experiment writes into its own subdirectory under `results/`, and the harmonic-oscillator study is split into Numerov-only, RK4-only, and comparison outputs. After the diagnostics refactor, this file now delegates root-finding figures to `src/diagnostics.py` instead of building them inline.
| Lines | Block | Purpose |
|---:|---|---|
| 99-129 | `function _experiment_results_dir` | Create and return one experiment-specific output directory under the shared results root. |
| 130-277 | `function run_square_well` | Run the infinite square well benchmark case and its convergence study. |
| 278-474 | `function run_harmonic_oscillator_Numerov` | Run the Numerov harmonic-oscillator benchmark, including inward-shooting diagnostics and Numerov-only convergence studies. |
| 475-634 | `function run_harmonic_oscillator_RK4` | Run the RK4 harmonic-oscillator study on the same convergence-grid family used by the Numerov benchmark. |
| 635-696 | `function run_harmonic_oscillator_Numerov_VS_RK4` | Build the direct Numerov-versus-RK4 error comparison outputs. |
| 697-769 | `function run_harmonic_oscillator` | Wrap the Numerov, RK4, and comparison sub-studies for the harmonic oscillator. |
| 770-1015 | `function run_quartic_double_well` | Run the quartic double-well study, including convergence and tunneling-splitting sweeps. |
| 1016-1149 | `function run_finite_square_well` | Run the finite square well, including outward-shooting root diagnostics and a finest-grid-reference convergence study for the first four bound states; because the potential jumps at the well edge, this case shows reduced observed order compared with the smooth benchmark potentials. |
| 1150-1317 | `function run_scattering` | Run the scattering extension for single- and double-barrier tunneling, including resonant-peak extraction. |

### `src/plotting.py`
Report-ready Matplotlib figures.
| Lines | Block | Purpose |
|---:|---|---|
| 22-41 | `function _ensure_parent` | Create the parent directory of an output path if it does not exist. |
| 42-72 | `function _symlog_linthresh` | Choose a stable linear threshold for symlog diagnostic axes. |
| 73-155 | `function plot_potential_and_states` | Plot the potential together with several shifted eigenstates. |
| 156-205 | `function plot_probability_densities` | Plot |psi(x)|^2 for several states. |
| 206-258 | `function plot_energy_comparison` | Compare numerical and exact energy levels on the same figure. |
| 259-345 | `function plot_error_curve` | Plot per-state convergence curves on log-log axes, usually exact-reference energy errors but also successive-refinement surrogates such as `|E(h_i)-E(h_{i+1})|` for the double well. |
| 346-402 | `function plot_splitting_curve` | Plot the lowest two energies and their splitting versus a parameter. |
| 403-474 | `function plot_root_finding_diagnostic` | Plot the global mismatch scan and mark only the final root estimate for each state. |
| 475-548 | `function plot_root_finding_zoom` | Plot a zoomed local bracket with the full bisection history overlaid. |
| 549-598 | `function plot_scattering_coefficients` | Plot transmission and reflection probabilities versus incident energy. |
| 599-661 | `function plot_scattering_potential_and_probability` | Plot a scattering probability density together with the barrier potential. |
| 662-714 | `function plot_numerov_vs_RK4_errors` | Compare Numerov and RK4 harmonic-oscillator energy errors. |

## Important numerical checks
- The Numerov derivative stencil is fourth order when enough points are available. This matters because even inward-shooting states use the condition `psi'(0)=0`, so a low-order boundary derivative would spoil the accuracy of the full eigenvalue solve.
- The parity-based startup in `initial_conditions_outward_shooting` includes higher-order Taylor terms, so the first Numerov step does not spoil the observed fourth-order convergence.
- Harmonic-oscillator inward shooting starts from the decaying forbidden-region tail and integrates toward the origin. This avoids contamination by the growing exponential mode that can dominate outward integration on a truncated infinite domain.
- The quartic double well can shift its analytic minima to zero. That keeps the physical potential fixed as the grid changes and prevents fake convergence effects from a grid-sampled minimum.
- Convergence studies are split intentionally: grid-refinement studies vary `h` at fixed domain size, while box-size studies keep `h` approximately fixed and vary `x_max`. That separation lets the project distinguish discretization error from domain-truncation error.
- RK4 comparison solves the same harmonic oscillator on the same convergence-grid family used by the Numerov study, so the comparison focuses on the integration formula rather than a different physical setup.
- The bound-state bisection routine finishes with a few safeguarded secant-style polishing steps inside the final bracket, which noticeably reduces the reported boundary mismatch for steep roots.
- Normalization is not cosmetic. The code uses numerical quadrature because physically meaningful wavefunctions must satisfy `∫|psi|^2 dx = 1`.
- Scattering outputs include a conservation check through `T + R`, which should remain close to one.
- The diagnostics layer now separates global raw-mismatch scans from zoomed single-root views. That keeps the report figures readable while still plotting the actual solver mismatch instead of an alternative transformed quantity.
- The experiments layer now writes outputs into experiment-specific subdirectories, so the results tree mirrors the project sections instead of mixing all CSV and PNG files in one folder.
- The tests include analytic benchmarks, convergence-order checks, parity-specific double-well checks, normalization checks, derivative-stencil checks, and scattering sanity checks.
