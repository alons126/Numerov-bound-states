# Numerov Schrödinger Solver

**Author:** Alon Sportes
**Written for the course:** Numerical methods in astrophysics (0321488101)

This project solves the one-dimensional time-independent Schrödinger equation using the Numerov method combined with shooting and root finding.

It computes bound states for several potentials, validates against analytic solutions, performs convergence studies, and explores tunneling and scattering phenomena.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run the project

```bash
python scripts/run_solver.py
```

## Output

Results are saved in:

```text
results/
```

including:

- wavefunctions and spectra
- convergence plots
- tunneling and scattering results

## Project structure

See the documentation index:

[docs/index.md](docs/index.md)

The `docs/` directory contains the detailed run instructions, code structure,
numerical-method overview, and result-reproduction guide for the course project.
Additional information is explained in the code itself.
