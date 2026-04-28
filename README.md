# numerov-bound-states

Numerical solution of the 1D time-independent Schrödinger equation with the Numerov method and parity-based shooting.

## Features

- Numerov integration for symmetric 1D bound-state problems
- Shooting + bisection root-finding for eigenvalues (mismatch function formulation)
- Safeguarded root polishing after bisection for smaller final boundary residuals
- Validation on:
  - infinite square well
  - harmonic oscillator
- Physics exploration on (core bound-state focus with extensions):
  - quartic double well
  - finite square well
  - finite rectangular barrier (exploration of scattering behavior such as transmission/reflection validation)
  - double-barrier resonant tunneling (exploration of resonance peak structure)
- Convergence studies versus grid spacing and domain size
- Root-finding diagnostics (mismatch function plots and bisection traces)
- Double-well tunneling splitting analysis
- Separate grid-refinement and box-size studies for the quartic double well
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
  - `report.tex`: main project report
  - `code_walkthrough.md`: reviewer-oriented code map
  - `Project_outline_and_motivation.tex`: project overview and implementation notes

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
- RK4 versus Numerov comparison data for the harmonic oscillator
- double-well splitting versus barrier parameter
- double-well root-diagnostic plots
- finite square well bound-state plots
- transmission and reflection spectra for finite barriers
- double-barrier resonance peak tables and resonant-state plots

## How it works

This project solves the one-dimensional time-independent Schrödinger equation

    ψ''(x) = 2 [V(x) − E] ψ(x)

using a combination of the Numerov method and a shooting procedure.

### 1. Discretization
A uniform spatial grid is created on a symmetric domain $[-x_{\max}, x_{\max}]$. For symmetric potentials, only the half-domain $[0, x_{\max}]$ is integrated, and the full wavefunction is reconstructed using parity.

### 2. Numerov integration
The differential equation is rewritten in the form

    ψ'' = q(x) ψ,  where q(x) = 2(V − E)

and integrated using the Numerov recurrence relation, which is a high-accuracy finite-difference scheme for second-order ODEs.

A key implementation detail is that the accuracy of the method depends not only on the Numerov recurrence itself, but also on how boundary conditions are enforced. In particular, for inward shooting problems such as the harmonic oscillator, the derivative at the origin must be computed with a high-order stencil to preserve the overall accuracy of the scheme.

The startup values used for parity-based shooting also matter: the current implementation includes higher-order Taylor terms near the origin so the first Numerov step does not reduce the observed convergence order.

### 3. Shooting method

For a given trial energy $E$:
- the wavefunction is integrated according to the chosen boundary strategy
- boundary conditions are chosen based on parity:
  - even states: $\psi'(0) = 0$
  - odd states:  $\psi(0) = 0$
- a mismatch function $M(E)$ is constructed

For symmetric potentials solved on the half-domain:
- outward shooting uses the boundary mismatch at $x = x_{\max}$
- inward shooting (used for unbounded problems like the harmonic oscillator) enforces decay at large $|x|$ and evaluates the mismatch at $x = 0$

Eigenvalues are obtained by solving $M(E) = 0$. The solver scans over energies to locate sign changes (bracketing), refines roots using bisection, and then applies a short safeguarded polishing step inside the final bracket for the outward-shooting bound-state solver.

To make the algorithm transparent, the code can generate diagnostic plots of $M(E)$ versus $E$, where zero crossings correspond to physical eigenvalues.

### 4. Wavefunction reconstruction and normalization
The half-domain solution is reflected to the negative axis using parity. The resulting full wavefunction is then normalized using numerical integration.

### 5. Analysis and experiments
The solver is applied to several systems (with bound states as the primary focus and additional extensions for more complex behavior):
- infinite square well (validation against exact solution)
- harmonic oscillator (validation against exact solution, using inward shooting for stability)
- quartic double well (tunneling and energy splitting, with an analytic shift that places the well minima at zero without making the potential grid-dependent)
- finite square well (finite number of bound states)
- finite rectangular barrier (exploration of scattering behavior and probability-conservation validation)
- double barrier (resonant tunneling and peak structure as an extension)

Additional analysis includes:
- convergence studies versus grid spacing and domain size to verify numerical accuracy
- for the quartic double well, separate treatment of grid refinement and box truncation:
  - `h`-convergence uses successive refinements on a fixed domain
  - `x_max`-convergence compares against a larger-box reference while keeping spacing approximately fixed
- parameter sweeps (e.g., double-well barrier height)
- generation of plots and CSV tables for reporting
- estimation of convergence rates $\Delta E \propto h^p$ from log--log fits
- comparison between Numerov and RK4 integration methods for the harmonic oscillator to assess accuracy, efficiency, and method suitability
- conservation check $T + R \approx 1$ for scattering states
- identification of resonance peaks in double-barrier transmission

### 6. Workflow
The full workflow is executed by:

```
python3 scripts/run_solver.py
```

This script:
- runs all experiments
- executes automated tests to verify correctness
- saves figures and data to `results/`
- generates root-finding diagnostic plots for selected states

## Notes

This implementation assumes symmetric potentials, so:

- even states satisfy `psi'(0)=0`
- odd states satisfy `psi(0)=0`

That makes the shooting problem simple and robust for a final project.

For unbounded problems such as the harmonic oscillator, the domain is truncated to a finite interval and inward shooting is used to enforce the physically correct decaying behavior at large $|x|$.

The code is written for clarity and analysis rather than maximum performance.

A subtle but important numerical point is that the accuracy of eigenvalue shooting depends on both the integrator and the boundary mismatch evaluation. In this project, a higher-order derivative approximation was required to ensure that the Numerov method achieves its expected accuracy when compared with RK4. This serves as a practical example of how implementation details can influence the apparent performance of numerical algorithms.

Another subtle point appears in the quartic double well: shifting the potential by the sampled grid minimum makes the potential itself vary slightly with grid spacing. The current implementation instead uses the analytic minimum, which keeps convergence studies physically meaningful.
