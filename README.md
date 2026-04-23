# numerov-bound-states

Numerical solution of the 1D time-independent Schrödinger equation with the Numerov method and parity-based shooting.

## Features

- Numerov integration for symmetric 1D bound-state problems
- Shooting + bisection search for eigenvalues
- Validation on:
  - infinite square well
  - harmonic oscillator
- Physics exploration on:
  - quartic double well
  - finite square well
- Convergence studies versus grid spacing and domain size
- Double-well tunneling splitting analysis
- Plots and CSV output for report-ready figures
- Lightweight automated tests

## Project structure

- `src/`: core implementation
  - `numerov.py`: Numerov stepping and safe normalization
  - `shooting.py`: parity-based shooting and eigenvalue search
  - `potentials.py`: potential definitions
  - `analysis.py`: convergence and parameter-sweep helpers
  - `plotting.py`: figure generation
  - `experiments.py`: high-level numerical experiments
- `scripts/`
  - `run_solver.py`: runs the full project workflow
- `tests/`
  - `test_solver.py`: lightweight numerical sanity checks
- `docs/`
  - `report.tex`: project report skeleton

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run everything

```bash
python3 scripts/run_solver.py
```

This writes figures and CSV tables to `results/`.

If you encounter import errors, run with:

```bash
PYTHONPATH=. python3 scripts/run_solver.py
```

## What gets generated

Running the solver generates:

- validation tables for square well and harmonic oscillator
- wavefunction and density plots
- convergence plots versus `h` and `x_max`
- double-well splitting versus barrier parameter
- finite square well bound-state plots

## Notes

This implementation assumes symmetric potentials, so:

- even states satisfy `psi'(0)=0`
- odd states satisfy `psi(0)=0`

That makes the shooting problem simple and robust for a final project.

The code is written for clarity and analysis rather than maximum performance.
