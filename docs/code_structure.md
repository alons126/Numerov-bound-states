# Code Structure

The project is organized into modular components so the numerical method, the physics definitions, and the reporting pipeline remain separate.

## High-level workflow

The project runs in a top-down pipeline:

1. `scripts/run_solver.py` clears the generated `results/` directory.
2. It runs each experiment in the same order used in the writeup.
3. Each experiment solves a physical problem, saves figures and CSV outputs, and delegates plotting to the shared plotting and diagnostics helpers.
4. The workflow finishes by calling `run_all_tests()` from `tests/test_solver.py`, which runs a lightweight regression subset as a numerical sanity check on the current codebase.

## Core solver

- **[`numerov.py`](../src/numerov.py):**
  - Implements Numerov integration, wavefunction normalization, and boundary derivative utilities.
  - **Main responsibilities:** build `q(x)` from a trial energy, march the second-order ODE on a uniform grid, normalize the final state, and estimate the derivative needed by inward-shooting parity conditions.

- **[`shooting.py`](../src/shooting.py):**
  - Implements parity-based shooting, mismatch functions, bracketing, and eigenvalue refinement for symmetric bound-state problems.
  - **Main responsibilities:** construct outward and inward startup values, evaluate mismatch functions, bracket sign changes, refine eigenvalues, reconstruct full wavefunctions, and return normalized state objects.

## Physics definitions

- **[`potentials.py`](../src/potentials.py):**
  - Defines the benchmark and exploration potentials used in the project, including square wells, the harmonic oscillator, the quartic double well, and scattering barriers.
  - This layer keeps the solver independent of the specific physical system.

## Experiments

- **[`experiments.py`](../src/experiments.py):**
  - Runs the main physical cases, performs the analysis steps used in the writeup, and saves plots and CSV outputs.
  - **Covered cases:**
  - Infinite square well
  - Harmonic oscillator
  - Quartic double well
  - Finite square well
  - Single-barrier scattering
  - Double-barrier resonant tunneling

## Analysis and visualization

- **[`analysis.py`](../src/analysis.py):**
  - Provides exact benchmark energies, convergence studies, error analysis, CSV export, and parameter sweeps.
  - **Main responsibilities:** supply analytic references where available, measure convergence with respect to grid spacing and box size, and write tabulated validation outputs used by the experiments.

- **[`plotting.py`](../src/plotting.py):**
  - Generates report-ready plots for spectra, wavefunctions, densities, convergence, tunneling, and scattering.
  - **Main responsibilities:** turn saved numerical outputs into consistent figures for the writeup and results directory without duplicating plotting logic in the experiment layer.

- **[`diagnostics.py`](../src/diagnostics.py):**
  - Builds the root-finding diagnostic figures, including global mismatch scans and zoomed bisection views.
  - **Main responsibilities:** keep experiment code focused on numerical setup while centralizing the visual checks used to inspect bracketing and root refinement.

## Extensions

- **[`scattering.py`](../src/scattering.py):**
  - Computes transmission, reflection, and resonant-tunneling behavior using a complex Numerov formulation.
  - **Main responsibilities:** impose asymptotic scattering boundary conditions, extract incident/reflected/transmitted amplitudes, and sweep energies for barrier problems.

- **[`RK4_harmonic_oscillator.py`](../src/RK4_harmonic_oscillator.py):**
  - Provides the RK4 harmonic-oscillator comparison used to benchmark Numerov against a general fourth-order ODE method.
  - **Main responsibilities:** solve the harmonic-oscillator shooting problem with RK4 on matched grids and return comparison data for the writeup figures.

## Supporting files and directories

- **[`run_solver.py`](../scripts/run_solver.py):**
  - Runs the full project workflow in the same order as the writeup.
  - **Main responsibilities:** clear and regenerate `results/`, execute the experiment suite, and finish with the lightweight regression subset used as a final sanity check.

- **[`test_solver.py`](../tests/test_solver.py):**
  - Contains the regression and numerical-validation tests.
  - **Main responsibilities:** check analytic benchmarks, convergence order, normalization, derivative stencils, parity-specific mismatch conditions, and basic scattering sanity. The top-level workflow runs a smaller subset through `run_all_tests()`, while `pytest` can execute the full module.

- **[`docs/`](../docs):**
  - Holds the user-facing documentation and the project writeup sources.
  - **Main responsibilities:** document how to run the project, explain the code organization and numerical method, and provide the final assignment writeup materials.
