# Reproducing Results

Run the full project workflow with:

```bash
python3 scripts/run_solver.py
```

This regenerates the numerical outputs used in the report and then runs the lightweight automated regression subset used by the workflow.

To run the full test module directly instead of the workflow subset, use:

```bash
python3 -m pytest tests/test_solver.py
```

## Generated experiment outputs

The workflow writes experiment-specific outputs under:

```text
results/
```

The directory structure is:

- `results/1_infinite_square_well_Numerov/`
- `results/2a_harmonic_oscillator_Numerov/`
- `results/2b_harmonic_oscillator_RK4/`
- `results/2c_harmonic_oscillator_Numerov_VS_RK4_comparison/`
- `results/3_double_well_Numerov/`
- `results/4_finite_square_well_Numerov/`
- `results/5_scattering_single_barrier_Numerov/`
- `results/6_scattering_double_barrier_Numerov/`

The execution order matches the report structure:

1. infinite square well
2. harmonic oscillator
3. quartic double well
4. finite square well
5. scattering and resonant tunneling

## Main results

- Infinite square well: analytic benchmark, parity-separated root diagnostics, and convergence versus grid spacing.

- Harmonic oscillator: analytic benchmark, inward-shooting diagnostics, convergence versus grid spacing and box size, and a matched-grid Numerov versus RK4 comparison.

- Quartic double well: low-lying states, tunneling splitting, successive-refinement convergence, and box-size sensitivity.

- Finite square well: bound states, near-threshold diagnostics, and a practical grid-refinement study against the finest saved run.

- Single barrier: transmission, reflection, and flux-conservation validation.

- Double barrier: resonant-tunneling transmission peaks and a resonant-state probability plot.

## File conventions

Each experiment directory contains the plots and CSV tables for that case. The filenames repeat the experiment prefix so the outputs remain self-describing when viewed outside the directory tree.

For the bound-state experiments, the saved files typically include:

- state-energy CSV tables
- state and probability-density plots
- convergence plots versus `h` and, where relevant, `x_max`
- global root-finding mismatch scans
- zoomed root-bracket plots with bisection history
