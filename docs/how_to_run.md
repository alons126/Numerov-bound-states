# How to Run

## Environment setup

Create a virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The project was tested on macOS with Python 3.14.4.

## Run the full workflow

Run all experiments and the lightweight automated tests with:

```bash
python3 scripts/run_solver.py
```

**NOTE:** This command clears and regenerates `results/`, so it will overwrite any saved output files in that directory. At the end of the run, it also removes generated `test_env/` and project `__pycache__/` directories.

If Python cannot resolve the local modules when running from your shell configuration, use:

```bash
PYTHONPATH=. python3 scripts/run_solver.py
```

To run the full test file directly with `pytest`, use:

```bash
python3 -m pytest tests/test_solver.py
```

## Output

All generated outputs are written under:

```text
results/
```

This includes:

- Bound-state energies and normalized wavefunctions
- Probability-density and spectrum plots
- Convergence studies versus grid spacing and box size
- Root-finding diagnostic plots
- Numerov versus RK4 comparison outputs for the harmonic oscillator
- Double-well splitting data
- Scattering transmission and reflection data
