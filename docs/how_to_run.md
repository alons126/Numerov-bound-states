# How to Run

## Environment setup

Create a virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run the full workflow

Run all experiments and the lightweight automated tests with:

```bash
python3 scripts/run_solver.py
```

If Python cannot resolve the local modules when running from your shell
configuration, use:

```bash
PYTHONPATH=. python3 scripts/run_solver.py
```

## Output

All generated outputs are written under:

```text
results/
```

This includes:

- bound-state energies and normalized wavefunctions
- probability-density and spectrum plots
- convergence studies versus grid spacing and box size
- root-finding diagnostic plots
- Numerov versus RK4 comparison outputs for the harmonic oscillator
- double-well splitting data
- scattering transmission and reflection data
