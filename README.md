# Numerov-bound-states

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
python main.py
```

This writes figures and CSV tables to `results/`.

## File guide

- `main.py`: runs the project workflow
- `potentials.py`: potential definitions
- `numerov.py`: Numerov stepping and normalization
- `shooting.py`: parity-based shooting and eigenvalue search
- `analysis.py`: exact spectra and convergence helpers
- `plotting.py`: figures
- `tests.py`: lightweight numerical sanity checks

## Notes

This implementation assumes **symmetric potentials** so that:
- even states satisfy `psi'(0)=0`
- odd states satisfy `psi(0)=0`

That makes the shooting problem much simpler and robust for a course project.

For the infinite square well, the code uses a very large outside barrier to mimic the ideal wall numerically.
