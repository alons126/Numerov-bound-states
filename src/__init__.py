"""
Package marker for the Numerov bound-state project.

Reviewer guide
--------------
The `src` package is intentionally split into layers:
- `numerov.py`: low-level recurrence, normalization, and derivative helpers
- `shooting.py`: bound-state eigenvalue search built on top of Numerov
- `potentials.py`: physical systems supplied as V(x)
- `analysis.py`: convergence, benchmarks, and parameter sweeps
- `plotting.py`: figure generation only
- `experiments.py`: high-level report cases and output generation
- `RK4_harmonic_oscillator.py`: RK4 harmonic-oscillator comparison module
- `scattering.py`: secondary extension for transmission and reflection

That structure matches the reviewer-facing documents and keeps physics,
numerics, analysis, and presentation concerns separate.
"""
