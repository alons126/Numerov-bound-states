# Reproducing Results

Run instructions are in [how_to_run.md](how_to_run.md). This page describes the generated outputs and how they map to the experiments.

## Generated experiment outputs

The workflow writes experiment-specific outputs under `results/`. The directory structure is:

- `results/1_infinite_square_well_Numerov/`
- `results/2a_harmonic_oscillator_Numerov/`
- `results/2b_harmonic_oscillator_RK4/`
- `results/2c_harmonic_oscillator_Numerov_VS_RK4_comparison/`
- `results/3_double_well_Numerov/`
- `results/4_finite_square_well_Numerov/`
- `results/5_scattering_single_barrier_Numerov/`
- `results/6_scattering_double_barrier_Numerov/`

The execution order broadly follows the structure of the project writeup.

## Main results

- **Infinite square well:** analytic benchmark, parity-separated root diagnostics, and convergence versus grid spacing.
- **Harmonic oscillator:** analytic benchmark, inward-shooting diagnostics, convergence versus grid spacing and box size, and a matched-grid comparison between Numerov and RK4.
- **Quartic double well:** low-lying states, tunneling splitting, successive-refinement convergence, and box-size sensitivity.
- **Finite square well:** bound states, near-threshold diagnostics, and a practical grid-refinement study against the finest saved run.
- **Single barrier:** transmission, reflection, and flux-conservation validation.
- **Double barrier:** resonant-tunneling transmission peaks and a resonant-state probability plot.

## File conventions

Each experiment directory contains the generated plots and CSV tables for that case. The filenames repeat the experiment prefix so the outputs remain self-describing when viewed outside the directory tree.

For the bound-state experiments, the saved files typically include:

- State-energy CSV tables
- State and probability-density plots
- Convergence plots versus $h$ and, where relevant, $x_{max}$
- Global root-finding mismatch scans
- Zoomed root-bracket plots with bisection history

For the scattering experiments, the saved files include transmission/reflection CSV tables and plots, while the double-barrier case also adds a transmission-peak table and a resonant-state probability-density plot.
