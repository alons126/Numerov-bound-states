# Numerov Schrödinger Solver

**Author:** Alon Sportes  
**Written for the course:** Numerical methods in astrophysics (0321488101)

This project solves the one-dimensional time-independent Schrödinger equation using the Numerov method combined with shooting and root finding. It computes bound states for several potentials, validates against analytic solutions, performs convergence studies, and explores tunneling and scattering phenomena.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Tested on macOS with Python 3.14.4.

## Run the project

```bash
python3 scripts/run_solver.py
```

For more detailed execution notes, test commands, and output information, see [docs/how_to_run.md](docs/how_to_run.md).

## Output

Results are saved in `results/`, including:

- Wavefunctions and spectra
- Convergence plots
- Tunneling and scattering results

The submitted project files include the output used in the writeup. **NOTE:** `python3 scripts/run_solver.py` clears and regenerates `results/`, so running it will overwrite those saved outputs. At the end of the run, it also removes generated `test_env/` and project `__pycache__/` directories.

## Project structure

See the documentation index in [docs/index.md](docs/index.md). The `docs/` directory contains the detailed run instructions, code structure, numerical-method overview, and result-reproduction guide for the course project. Additional information is explained in the code itself.
