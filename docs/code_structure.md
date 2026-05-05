# Code Structure

The project is organized into modular components so the numerical method, the
physics definitions, and the reporting pipeline remain separate.

## High-level workflow

The project runs in a top-down pipeline:

1. `scripts/run_solver.py` clears the generated `results/` directory.
2. It runs each experiment in the same order used in the writeup.
3. Each experiment solves a physical problem, saves figures and CSV outputs,
   and delegates plotting to the shared plotting and diagnostics helpers.
4. The lightweight regression tests in `tests/test_solver.py` run at the end as
   a numerical sanity check on the current codebase.

## Core solver

- `numerov.py`
  Implements Numerov integration, wavefunction normalization, and boundary
  derivative utilities.

  Main responsibilities:
  build `q(x)` from a trial energy, march the second-order ODE on a uniform
  grid, normalize the final state, and estimate the derivative needed by
  inward-shooting parity conditions.

- `shooting.py`
  Implements parity-based shooting, mismatch functions, bracketing, and
  eigenvalue refinement for symmetric bound-state problems.

  Main responsibilities:
  construct outward and inward startup values, evaluate mismatch functions,
  bracket sign changes, refine eigenvalues, reconstruct full wavefunctions, and
  return normalized state objects.

## Physics definitions

- `potentials.py`
  Defines the benchmark and exploration potentials used in the project,
  including square wells, the harmonic oscillator, the quartic double well, and
  scattering barriers.

  This layer keeps the solver independent of the specific physical system.

## Experiments

- `experiments.py`
  Runs the main physical cases, performs the analysis steps used in the
  writeup, and saves plots and CSV outputs.

  Covered cases:
  infinite square well, harmonic oscillator, quartic double well, finite square
  well, single-barrier scattering, and double-barrier resonant tunneling.

## Analysis and visualization

- `analysis.py`
  Provides exact benchmark energies, convergence studies, error analysis, CSV
  export, and parameter sweeps.

  This is the main scientific-validation layer rather than just output
  formatting.

- `plotting.py`
  Generates report-ready plots for spectra, wavefunctions, densities,
  convergence, tunneling, and scattering.

- `diagnostics.py`
  Builds the root-finding diagnostic figures, including global mismatch scans
  and zoomed bisection views.

  This keeps experiment code focused on numerical setup rather than
  plot-assembly details.

## Extensions

- `scattering.py`
  Computes transmission, reflection, and resonant-tunneling behavior using a
  complex Numerov formulation.

- `RK4_harmonic_oscillator.py`
  Provides the RK4 harmonic-oscillator comparison used to benchmark Numerov
  against a general fourth-order ODE method.

## Supporting files

- `scripts/run_solver.py`
  Runs the full project workflow in the same order as the writeup.

- `tests/test_solver.py`
  Contains the lightweight regression and numerical-validation tests.

  These tests cover analytic benchmarks, convergence-order checks,
  normalization, derivative-stencil correctness, parity-specific mismatch
  behavior, and scattering probability conservation.

- `docs/`
  Holds the user-facing documentation and the project writeup sources.
