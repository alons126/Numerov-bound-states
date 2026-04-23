# numerov-bound-states

A small project for solving the 1D time-independent Schrödinger equation with the Numerov method and a shooting procedure.

## Equation

In dimensionless units, the code solves

\[
\psi''(x) = 2\,[V(x) - E]\,\psi(x)
\]

for bound states in symmetric potentials.

## Implemented potentials

- Infinite square well
- Harmonic oscillator
- Finite square well
- Quartic double well

## Main features

- Numerov integration on the half-domain
- Parity-based shooting for symmetric potentials
- Bisection search for eigenvalues
- Wavefunction normalization
- Exact comparisons for the infinite square well and harmonic oscillator
- Convergence plots versus grid spacing and box size

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Project structure

- `src/`: core implementation
  - `numerov.py`: Numerov stepping and normalization
  - `shooting.py`: parity-based shooting and eigenvalue search
  - `potentials.py`: potential definitions
  - `analysis.py`: exact spectra and convergence helpers
  - `plotting.py`: figure generation
- `scripts/`
  - `run_solver.py`: runs the full project workflow
- `tests/`
  - `test_solver.py`: lightweight numerical sanity checks
- `docs/`
  - `report.tex`: project report

## Notes

This implementation assumes **symmetric potentials** so that:
- even states satisfy `psi'(0)=0`
- odd states satisfy `psi(0)=0`

That makes the shooting problem much simpler and robust for a course project.

For the infinite square well, the code uses a very large outside barrier to mimic the ideal wall numerically.
