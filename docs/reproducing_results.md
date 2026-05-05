# Reproducing Results

Run:

```bash
python scripts/run_solver.py
```

This regenerates the numerical outputs used in the report.

## Main results

- Infinite square well: validation against the analytic spectrum
- Harmonic oscillator: validation and Numerov versus RK4 comparison
- Finite square well: bound states and convergence check
- Quartic double well: tunneling doublet and splitting study
- Single barrier: transmission, reflection, and flux conservation
- Double barrier: resonant tunneling and resonance-state plot

All outputs are saved under:

```text
results/
```

The `results/` directory is organized into experiment-specific subdirectories, matching the structure of the report.