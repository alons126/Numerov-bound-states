# Code Structure

The project is organized into modular components so the numerical method, the
physics definitions, and the reporting pipeline remain separate.

## Core solver

- `numerov.py`
  Implements Numerov integration, wavefunction normalization, and boundary
  derivative utilities.

- `shooting.py`
  Implements parity-based shooting, mismatch functions, bracketing, and
  eigenvalue refinement for symmetric bound-state problems.

## Physics definitions

- `potentials.py`
  Defines the benchmark and exploration potentials used in the project,
  including square wells, the harmonic oscillator, the quartic double well, and
  scattering barriers.

## Experiments

- `experiments.py`
  Runs the main physical cases, performs the analysis steps used in the
  writeup, and saves plots and CSV outputs.

## Analysis and visualization

- `analysis.py`
  Provides exact benchmark energies, convergence studies, error analysis, CSV
  export, and parameter sweeps.

- `plotting.py`
  Generates report-ready plots for spectra, wavefunctions, densities,
  convergence, tunneling, and scattering.

- `diagnostics.py`
  Builds the root-finding diagnostic figures, including global mismatch scans
  and zoomed bisection views.

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

- `docs/`
  Holds the user-facing documentation and the project writeup sources.
