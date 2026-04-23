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

## How it works

This project solves the one-dimensional time-independent Schrödinger equation

    ψ''(x) = 2 [V(x) − E] ψ(x)

using a combination of the Numerov method and a shooting procedure.

### 1. Discretization
A uniform spatial grid is created on a symmetric domain [−x_max, x_max]. For symmetric potentials, only the half-domain [0, x_max] is integrated, and the full wavefunction is reconstructed using parity.

### 2. Numerov integration
The differential equation is rewritten in the form

    ψ'' = q(x) ψ,  where q(x) = 2(V − E)

and integrated using the Numerov recurrence relation, which is a high-accuracy finite-difference scheme for second-order ODEs.

### 3. Shooting method
For a given trial energy E:
- the wavefunction is integrated outward from x = 0
- boundary conditions are chosen based on parity:
  - even states: ψ'(0) = 0
  - odd states:  ψ(0) = 0
- a mismatch function at the boundary x = x_max is computed

Eigenvalues are found by scanning over energies, detecting sign changes in the mismatch, and refining them with bisection.

### 4. Wavefunction reconstruction and normalization
The half-domain solution is reflected to the negative axis using parity. The resulting full wavefunction is then normalized using numerical integration.

### 5. Analysis and experiments
The solver is applied to several systems:
- infinite square well (validation against exact solution)
- harmonic oscillator (validation against exact solution)
- quartic double well (tunneling and energy splitting)
- finite square well (finite number of bound states)

Additional analysis includes:
- convergence studies versus grid spacing and domain size
- parameter sweeps (e.g., double-well barrier height)
- generation of plots and CSV tables for reporting

### 6. Workflow
The full workflow is executed by:

```
python3 scripts/run_solver.py
```

This script:
- runs all experiments
- saves figures and data to `results/`
- executes automated tests to verify correctness

## Notes

This implementation assumes symmetric potentials, so:

- even states satisfy `psi'(0)=0`
- odd states satisfy `psi(0)=0`

That makes the shooting problem simple and robust for a final project.

The code is written for clarity and analysis rather than maximum performance.
