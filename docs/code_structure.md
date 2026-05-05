}
# Code Structure

The project is organized into modular components:

## Core solver

- numerov.py  
  Numerov integration and wavefunction normalization

- shooting.py  
  Shooting method and eigenvalue search (bisection)

## Physics definitions

- potentials.py  
  Defines all potentials used in the project

## Experiments

- experiments.py  
  Runs all physical cases and generates results

## Analysis and visualization

- analysis.py  
  Convergence studies and error analysis

- plotting.py  
  Plot generation

- diagnostics.py  
  Root-finding and debugging diagnostics

## Extensions

- scattering.py  
  Barrier transmission and resonances

- RK4_harmonic_oscillator.py  
  RK4 comparison for the harmonic oscillato
# Code Structure

The project is organized into modular components.

## Core solver

- `numerov.py`  
  Implements Numerov integration, wavefunction normalization, and boundary derivative utilities.

- `shooting.py`  
  Implements parity-based shooting, boundary mismatch functions, bracketing, and bisection-based eigenvalue search.

## Physics definitions

- `potentials.py`  
  Defines all potentials used in the project, including validation potentials, double-well potentials, and scattering barriers.

## Experiments

- `experiments.py`  
  Runs the main physical cases, generates numerical results, and saves plots and CSV files.

## Analysis and visualization

- `analysis.py`  
  Provides exact benchmark energies, convergence studies, error analysis, and parameter sweeps.

- `plotting.py`  
  Generates report-ready figures for spectra, wavefunctions, densities, convergence, tunneling, and scattering.

- `diagnostics.py`  
  Generates root-finding diagnostics, including global mismatch scans and zoomed bisection views.

## Extensions

- `scattering.py`  
  Computes transmission, reflection, and resonant tunneling behavior for barrier potentials.

- `RK4_harmonic_oscillator.py`  
  Provides the RK4 harmonic-oscillator comparison used to benchmark Numerov against a general fourth-order ODE method.